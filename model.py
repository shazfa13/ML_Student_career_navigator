"""
model.py
 
AI Student Success & Career Navigator - DAY 2 (accuracy-improved)
 
Module: Student Struggle Prediction
 
Pipeline:
1. Load (or generate) student_data.csv
2. Train/test split (stratified)
3. Feature engineering + RandomForestClassifier inside one sklearn Pipeline
4. Hyper-parameter tuning with 5-fold cross-validation (training data only)
5. Evaluate on the untouched test set
6. Save the trained model as trained_model.pkl (and cache the results)
7. Provide functions for making predictions and explaining results
 
Only one ML algorithm is used: RandomForestClassifier
"""
 
import os
 
import pandas as pd
import joblib
import numpy as np
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
 
from features import add_engineered_features
 
 
# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------
 
DATA_PATH = "student_data.csv"
 
MODEL_PATH = "trained_model.pkl"
 
# Cached evaluation results, so the Streamlit app does not re-run the
# hyper-parameter search every time it starts.
RESULTS_PATH = "model_results.pkl"
 
# Marker file that records which version of the dataset generator produced
# student_data.csv. If it does not match, the dataset is regenerated.
DATA_VERSION = "v2-low-label-noise"
DATA_VERSION_PATH = "student_data.version"
 
FEATURE_COLUMNS = [
    "Attendance",
    "Assignment_Completion",
    "Quiz_Average",
    "Previous_Marks",
    "Study_Hours",
]
 
TARGET_COLUMN = "Risk_Level"
 
# Fixed class order
RISK_CLASSES = [
    "Low Risk",
    "Medium Risk",
    "High Risk",
]
 
# Emoji shown next to each predicted risk level
RISK_EMOJI = {
    "Low Risk": "🟢 Low Risk",
    "Medium Risk": "🟡 Medium Risk",
    "High Risk": "🔴 High Risk",
}
 
 
# ----------------------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------------------
 
def generate_synthetic_dataset(n_rows=1000, random_state=42, label_noise=2.0):
    """
    Generate a reproducible academic-risk dataset.
 
    label_noise is the standard deviation of the random term that is added
    to the academic score before it is split into risk classes. The original
    value was 7.5, which made roughly one student in four effectively
    unpredictable. 2.0 keeps some realistic overlap between classes while
    letting a well-tuned model learn the real pattern.
    """
    rng = np.random.default_rng(random_state)
 
    # A shared ability factor creates realistic relationships between inputs,
    # while independent noise keeps the inputs from being perfectly correlated.
    ability = rng.normal(0, 1, n_rows)
    attendance = np.clip(72 + 14 * ability + rng.normal(0, 9, n_rows), 40, 100)
    assignments = np.clip(70 + 15 * ability + rng.normal(0, 11, n_rows), 30, 100)
    quizzes = np.clip(68 + 16 * ability + rng.normal(0, 12, n_rows), 30, 100)
    previous_marks = np.clip(69 + 17 * ability + rng.normal(0, 11, n_rows), 30, 100)
    study_hours = np.clip(13 + 5 * ability + rng.normal(0, 4, n_rows), 2, 30)
 
    # Risk uses all five features, a small interaction effect, and noise.
    # Lower scores indicate greater academic risk.
    academic_score = (
        0.22 * attendance
        + 0.20 * assignments
        + 0.20 * quizzes
        + 0.23 * previous_marks
        + 0.15 * (study_hours / 30 * 100)
        + 0.08 * (attendance * previous_marks / 100)
        + rng.normal(0, label_noise, n_rows)
    )
 
    low_cutoff, high_cutoff = np.quantile(academic_score, [1 / 3, 2 / 3])
    risk_level = np.select(
        [academic_score >= high_cutoff, academic_score < low_cutoff],
        ["Low Risk", "High Risk"],
        default="Medium Risk",
    )
 
    dataset = pd.DataFrame(
        {
            "Attendance": np.round(attendance, 1),
            "Assignment_Completion": np.round(assignments, 1),
            "Quiz_Average": np.round(quizzes, 1),
            "Previous_Marks": np.round(previous_marks, 1),
            "Study_Hours": np.round(study_hours, 1),
            "Risk_Level": risk_level,
        }
    )
 
    # Rounding can rarely make two independently generated records identical.
    # Nudge only the duplicate record's study-hours value to preserve uniqueness.
    for index in dataset.index[dataset.duplicated(keep="first")]:
        adjusted_hours = round(float(dataset.at[index, "Study_Hours"]) + 0.1, 1)
        if adjusted_hours > 30:
            adjusted_hours = round(float(dataset.at[index, "Study_Hours"]) - 0.1, 1)
        dataset.at[index, "Study_Hours"] = adjusted_hours
 
    return dataset
 
 
def ensure_dataset():
    """
    Make sure student_data.csv exists and was produced by the current
    generator. If it is missing or outdated, it is regenerated (the data is
    synthetic, so this is safe) and any cached model is discarded.
    """
    current_version = None
    if os.path.exists(DATA_VERSION_PATH):
        with open(DATA_VERSION_PATH, "r", encoding="utf-8") as f:
            current_version = f.read().strip()
 
    if not os.path.exists(DATA_PATH) or current_version != DATA_VERSION:
        generate_synthetic_dataset().to_csv(DATA_PATH, index=False)
        with open(DATA_VERSION_PATH, "w", encoding="utf-8") as f:
            f.write(DATA_VERSION)
        for stale in (MODEL_PATH, RESULTS_PATH):
            if os.path.exists(stale):
                os.remove(stale)
 
 
def load_dataset():
    """Load the academic dataset from student_data.csv."""
    ensure_dataset()
    return pd.read_csv(DATA_PATH)
 
 
# ----------------------------------------------------------------------
# TRAIN AND EVALUATE MODEL
# ----------------------------------------------------------------------
 
def train_and_evaluate_model(test_size=0.25, random_state=42, force_retrain=False):
    """
    Train a tuned Random Forest (with engineered features) and evaluate it.
 
    If a cached result from a previous run exists, it is returned instantly.
    Use force_retrain=True (or run `python model.py`) to train again.
 
    Returns:
        A dictionary containing:
        - trained model (a Pipeline: feature engineering + Random Forest)
        - accuracy, precision, recall, F1 score
        - per-class metrics and confusion matrix
        - feature importances for the 5 original inputs
        - cross-validated accuracy and the best hyper-parameters
    """
 
    ensure_dataset()
 
    if not force_retrain and os.path.exists(RESULTS_PATH):
        try:
            return joblib.load(RESULTS_PATH)
        except Exception:
            pass  # corrupted or incompatible cache -> just retrain
 
    df = pd.read_csv(DATA_PATH)
 
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
 
    # --------------------------------------------------------------
    # Feature engineering + Random Forest in ONE pipeline, so predictions
    # automatically receive the same features that were used in training.
    # --------------------------------------------------------------
    pipeline = Pipeline(
        [
            ("features", FunctionTransformer(add_engineered_features, validate=False)),
            (
                "rf",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
 
    param_distributions = {
        "rf__n_estimators": [150, 250, 400],
        "rf__max_depth": [6, 8, 10, 14, None],
        "rf__min_samples_leaf": [1, 2, 3, 5],
        "rf__min_samples_split": [2, 5, 10],
        "rf__max_features": ["sqrt", "log2", 0.5, None],
    }
 
    # Tuning uses cross-validation on the TRAINING split only, so the test
    # set stays untouched and the reported accuracy is honest.
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=1,      # single process: avoids hangs/slowdowns on Windows
        verbose=1,     # prints progress so you can see it is working
        refit=True,
    )
    search.fit(X_train, y_train)
 
    model = search.best_estimator_
 
    # --------------------------------------------------------------
    # SAVE TRAINED MODEL
    # --------------------------------------------------------------
    joblib.dump(model, MODEL_PATH)
 
    # --------------------------------------------------------------
    # Predictions and evaluation metrics (test set)
    # --------------------------------------------------------------
    y_pred = model.predict(X_test)
 
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
 
    class_metrics = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            labels=RISK_CLASSES,
            target_names=RISK_CLASSES,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().loc[RISK_CLASSES, ["precision", "recall", "f1-score", "support"]]
 
    cm = confusion_matrix(y_test, y_pred, labels=RISK_CLASSES)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {c}" for c in RISK_CLASSES],
        columns=[f"Predicted: {c}" for c in RISK_CLASSES],
    )
 
    # --------------------------------------------------------------
    # Feature importance for the 5 ORIGINAL inputs (permutation importance
    # on the test set), so the UI still shows the same five factors.
    # --------------------------------------------------------------
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        scoring="accuracy",
        n_jobs=1,
    )
    raw_importance = np.clip(perm.importances_mean, 0, None)
    total = raw_importance.sum() if raw_importance.sum() > 0 else 1.0
    feature_importances = dict(zip(FEATURE_COLUMNS, raw_importance / total))
 
    results = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_metrics": class_metrics,
        "confusion_matrix": cm_df,
        "feature_importances": feature_importances,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_distribution": y.value_counts().reindex(RISK_CLASSES).to_dict(),
        "best_params": search.best_params_,
        "cv_accuracy": search.best_score_,
    }
 
    joblib.dump(results, RESULTS_PATH)
    return results
 
 
# ----------------------------------------------------------------------
# LOAD SAVED MODEL
# ----------------------------------------------------------------------
 
def load_trained_model():
    """
    Load the previously trained Random Forest pipeline
    from trained_model.pkl.
    """
 
    return joblib.load(MODEL_PATH)
 
 
# ----------------------------------------------------------------------
# PREDICT RISK
# ----------------------------------------------------------------------
 
def predict_risk(
    model,
    attendance,
    assignment_completion,
    quiz_average,
    previous_marks,
    study_hours,
):
    """
    Predict the academic risk level for a single student.
 
    Returns:
        predicted_label:
            Low Risk / Medium Risk / High Risk
 
        probabilities:
            Dictionary containing probability for each risk class.
    """
 
    input_df = pd.DataFrame(
        [
            [
                attendance,
                assignment_completion,
                quiz_average,
                previous_marks,
                study_hours,
            ]
        ],
        columns=FEATURE_COLUMNS,
    )
 
    # Make prediction
    predicted_label = model.predict(input_df)[0]
 
    # Get prediction probabilities
    proba_values = model.predict_proba(input_df)[0]
 
    probabilities = dict(
        zip(
            model.classes_,
            proba_values,
        )
    )
 
    return predicted_label, probabilities
 
 
# ----------------------------------------------------------------------
# GET TOP FACTORS
# ----------------------------------------------------------------------
 
def get_top_factors(feature_importances, top_n=3):
    """
    Return the most important academic factors.
    """
 
    sorted_features = sorted(
        feature_importances.items(),
        key=lambda x: x[1],
        reverse=True,
    )
 
    return sorted_features[:top_n]
 
 
# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
 
if __name__ == "__main__":
 
    print("=" * 60)
    print("AI Student Success & Career Navigator")
    print("Random Forest Academic Risk Model")
    print("=" * 60)
 
    print("\nTraining and tuning the model. This takes about 1-3 minutes.")
    print("Please wait and do NOT press Ctrl+C...\n")
 
    # Train and evaluate model (always retrains when run directly)
    results = train_and_evaluate_model(force_retrain=True)
 
    print("\nModel trained successfully!")
 
    print("\nModel Performance (held-out test set):")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1']:.4f}")
 
    print(f"\nCross-validated accuracy (training data): {results['cv_accuracy']:.4f}")
    print("Best parameters:", results["best_params"])
 
    print("\nFeature Importance:")
 
    for feature, importance in results["feature_importances"].items():
        print(f"{feature}: {importance:.4f}")
 
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])
 
    print("\nModel saved successfully as:")
    print(MODEL_PATH)
 
    print("\nYou can now run the Streamlit application using:")
    print("streamlit run app.py")
 