import os
import json
import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score

SPLIT_DIR = os.path.join("Data", "processed", "splits")
BASE_MODEL_PATH = os.path.join("reports", "models", "logreg_baseline.joblib")
OUT_MODEL_PATH = os.path.join("reports", "models", "logreg_calibrated.joblib")
REPORT_PATH = os.path.join("reports", "logreg_calibration_report.json")
TARGET = "class"

def load_split(name: str):
    return pd.read_csv(os.path.join(SPLIT_DIR, f"{name}.csv"))

def report_probs(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_prob": float(p.mean()),
        "positive_rate": float(y.mean())
    }

def main():
    base = joblib.load(BASE_MODEL_PATH)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    X_val, y_val = val.drop(columns=[TARGET]), val[TARGET].astype(int).values
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET].astype(int).values

    # Calibration sur val (on ne touche pas au test pour calibrer)
    calibrator = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calibrator.fit(X_val, y_val)

    joblib.dump(calibrator, OUT_MODEL_PATH)

    p_val = calibrator.predict_proba(X_val)[:, 1]
    p_test = calibrator.predict_proba(X_test)[:, 1]

    out = {
        "calibration_method": "sigmoid",
        "val": report_probs(y_val, p_val),
        "test": report_probs(y_test, p_test)
    }

    os.makedirs("reports", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("✅ Calibrated model saved:", OUT_MODEL_PATH)
    print("✅ Report saved:", REPORT_PATH)
    print("VAL:", out["val"])
    print("TEST:", out["test"])

if __name__ == "__main__":
    main()
