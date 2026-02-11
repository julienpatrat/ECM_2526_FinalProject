import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix
)

SPLIT_DIR = os.path.join("Data", "processed", "splits")
TARGET = "class"
MODEL_DIR = os.path.join("reports", "models")
REPORT_DIR = os.path.join("reports")
RANDOM_STATE = 42

def load_split(name: str) -> pd.DataFrame:
    path = os.path.join(SPLIT_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}. Run split_year1.py first.")
    return pd.read_csv(path)

def eval_model(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    roc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y, pred).tolist()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET].astype(int)
    X_val, y_val = val.drop(columns=[TARGET]), val[TARGET].astype(int)
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET].astype(int)

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    pipe.fit(X_train, y_train)

    # Évaluation à seuil 0.5 (baseline)
    results = {
        "val": eval_model(pipe, X_val, y_val, threshold=0.5),
        "test": eval_model(pipe, X_test, y_test, threshold=0.5),
    }

    # Sauvegarde
    model_path = os.path.join(MODEL_DIR, "logreg_baseline.joblib")
    joblib.dump(pipe, model_path)

    report_path = os.path.join(REPORT_DIR, "logreg_baseline_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("✅ Baseline trained.")
    print(f"- Model saved: {model_path}")
    print(f"- Metrics saved: {report_path}")
    print("VAL:", results["val"])
    print("TEST:", results["test"])

if __name__ == "__main__":
    main()
