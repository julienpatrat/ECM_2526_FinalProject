import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

SPLIT_DIR = os.path.join("Data", "processed", "splits_dedup")
TARGET = "class"
MODEL_DIR = os.path.join("reports", "models")
REPORT_PATH = os.path.join("reports", "hgb_report.json")
RANDOM_STATE = 42

def load_split(name: str) -> pd.DataFrame:
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET].astype(int).values
    X_val, y_val = val.drop(columns=[TARGET]), val[TARGET].astype(int).values
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET].astype(int).values

    # modèle non-linéaire + gestion missing via imputer
    base = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("hgb", HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            max_depth=None,
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31
        ))
    ])

    base.fit(X_train, y_train)

    # Calibration sur validation
    cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    cal.fit(X_val, y_val)

    joblib.dump(cal, os.path.join(MODEL_DIR, "hgb_calibrated.joblib"))

    p_val = cal.predict_proba(X_val)[:, 1]
    p_test = cal.predict_proba(X_test)[:, 1]

    out = {
        "model": "HistGradientBoosting + sigmoid calibration",
        "val": report_probs(y_val, p_val),
        "test": report_probs(y_test, p_test)
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("✅ Saved:", REPORT_PATH)
    print("VAL:", out["val"])
    print("TEST:", out["test"])

if __name__ == "__main__":
    main()
