import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from model.knn_model import load_model as knn
from model.logistic_model import load_model as lr
from model.decision_tree_model import load_model as dt
from model.naive_bayes_model import load_model as nb
from model.random_forest_model import load_model as rf
from model.xgboost_model import load_model as xgb

def train_and_evaluate(df, target_col):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": lr(),
        "Decision Tree": dt(),
        "kNN": knn(),
        "Naive Bayes": nb(),
        "Random Forest": rf(),
        "XGBoost": xgb()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred)
        })

    return pd.DataFrame(results)
