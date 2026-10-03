import pandas as pd
BASE_COLUMNS=["Attendance","Assignment_Completion","Quiz_Average","Previous_Marks","Study_Hours"]
def add_engineered_features(X):
    X=pd.DataFrame(X,columns=BASE_COLUMNS).copy()
    sp=X["Study_Hours"]/30*100
    sc=X[["Attendance","Assignment_Completion","Quiz_Average","Previous_Marks"]]
    X["Study_Pct"]=sp
    X["Academic_Mean"]=(sc.sum(axis=1)+sp)/5
    X["Score_Mean"]=sc.mean(axis=1)
    X["Weakest_Score"]=sc.min(axis=1)
    X["Score_Spread"]=sc.max(axis=1)-sc.min(axis=1)
    X["Attend_x_Marks"]=X["Attendance"]*X["Previous_Marks"]/100
    X["Effort_Gap"]=X["Assignment_Completion"]-X["Quiz_Average"]
    return X
 