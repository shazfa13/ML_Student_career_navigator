"""
model.py

AI Student Success & Career Navigator - DAY 2

Module: Student Struggle Prediction

This file handles the academic-risk Random Forest model.

Pipeline:
1. Load student_data.csv
2. Train/test split
3. Train RandomForestClassifier
4. Evaluate the model
5. Save the trained model as trained_model.pkl
6. Provide functions for making predictions and explaining results

Only one ML algorithm is used:
RandomForestClassifier
"""

import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------

DATA_PATH = "student_data.csv"

MODEL_PATH = "trained_model.pkl"

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

def load_dataset():
    """Load the academic dataset from student_data.csv."""
    return pd.read_csv(DATA_PATH)


def generate_synthetic_dataset(n_rows=1000, random_state=42):
    """Generate a reproducible, realistic academic-risk dataset."""
    rng = np.random.default_rng(random_state)

    # A shared ability factor creates realistic relationships between inputs,
    # while independent noise keeps the three classes from being perfect.
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
        + rng.normal(0, 7.5, n_rows)
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


# ----------------------------------------------------------------------
# TRAIN AND EVALUATE MODEL
# ----------------------------------------------------------------------

def train_and_evaluate_model(test_size=0.25, random_state=42):
    """
    Train a Random Forest Classifier and evaluate its performance.

    Returns:
        A dictionary containing:
        - trained model
        - accuracy
        - precision
        - recall
        - F1 score
        - confusion matrix
        - feature importances
    """

    # Load dataset
    df = load_dataset()

    # Separate features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # --------------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # --------------------------------------------------------------
    # SAVE TRAINED MODEL
    # --------------------------------------------------------------

    joblib.dump(model, MODEL_PATH)

    # --------------------------------------------------------------
    # Predictions
    # --------------------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

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

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=RISK_CLASSES,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {c}" for c in RISK_CLASSES],
        columns=[f"Predicted: {c}" for c in RISK_CLASSES],
    )

    # --------------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------------

    feature_importances = dict(
        zip(
            FEATURE_COLUMNS,
            model.feature_importances_,
        )
    )

    return {
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
    }


# ----------------------------------------------------------------------
# LOAD SAVED MODEL
# ----------------------------------------------------------------------

def load_trained_model():
    """
    Load the previously trained Random Forest model
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

    # Train and evaluate model
    results = train_and_evaluate_model()

    print("\nModel trained successfully!")

    print("\nModel Performance:")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1']:.4f}")

    print("\nFeature Importance:")

    for feature, importance in results["feature_importances"].items():
        print(f"{feature}: {importance:.4f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print("\nModel saved successfully as:")
    print(MODEL_PATH)

    print("\nYou can now run the Streamlit application using:")
    print("streamlit run app.py")
    