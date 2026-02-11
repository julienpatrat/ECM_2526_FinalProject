import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib

SPLIT_DIR = os.path.join("Data", "processed", "us", "splits")
MODEL_DIR = os.path.join("models", "us")
REPORT_PATH = os.path.join("reports", "us_hgb_report.json")

TARGET = "class"
DROP_COLS = ["company_id", "year", TARGET]

def load_split(name):
    return pd.read_csv(os.path.join(SPLIT_DIR, f"{name}.csv"))

def metrics(y_true, p):
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "mean_prob": float(np.mean(p)),
        "positive_rate": float(np.mean(y_true)),
    }

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    X_train = train.drop(columns=DROP_COLS, errors="ignore")
    y_train = train[TARGET].astype(int)

    X_val = val.drop(columns=DROP_COLS, errors="ignore")
    y_val = val[TARGET].astype(int)

    X_test = test.drop(columns=DROP_COLS, errors="ignore")
    y_test = test[TARGET].astype(int)

    base = HistGradientBoostingClassifier(random_state=42)
    base.fit(X_train, y_train)

    calib = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calib.fit(X_val, y_val)

    p_val = calib.predict_proba(X_val)[:, 1]
    p_test = calib.predict_proba(X_test)[:, 1]

    report = {
        "model": "HistGradientBoosting + sigmoid calibration",
        "val": metrics(y_val, p_val),
        "test": metrics(y_test, p_test),
        "n_features": int(X_train.shape[1]),
        "sizes": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))}
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    joblib.dump(calib, os.path.join(MODEL_DIR, "hgb_calibrated.joblib"))

    print("✅ Saved:", REPORT_PATH)
    print("✅ Saved model:", os.path.join(MODEL_DIR, "hgb_calibrated.joblib"))
    print(report)

if __name__ == "__main__":
    main()
