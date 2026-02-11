import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    average_precision_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)

SPLIT_DIR = os.path.join("Data", "processed", "splits")
MODEL_PATH = os.path.join("reports", "models", "logreg_baseline.joblib")
TARGET = "class"

def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(SPLIT_DIR, f"{name}.csv"))

def metrics_at_threshold(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    return {
        "threshold": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist()
    }

def main():
    model = joblib.load(MODEL_PATH)

    val = load_split("val")
    test = load_split("test")

    X_val, y_val = val.drop(columns=[TARGET]), val[TARGET].astype(int).values
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET].astype(int).values

    proba_val = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    # métriques "ranking" indépendantes du seuil
    base = {
        "val": {
            "roc_auc": float(roc_auc_score(y_val, proba_val)),
            "pr_auc": float(average_precision_score(y_val, proba_val)),
        },
        "test": {
            "roc_auc": float(roc_auc_score(y_test, proba_test)),
            "pr_auc": float(average_precision_score(y_test, proba_test)),
        }
    }

    # grille de seuils
    thresholds = np.linspace(0.01, 0.99, 99)

    # 3 objectifs possibles
    best_f1 = None
    best_recall_80 = None  # meilleur precision sous contrainte recall >= 0.80
    best_precision_20 = None  # meilleur recall sous contrainte precision >= 0.20

    rows = []
    for thr in thresholds:
        m = metrics_at_threshold(y_val, proba_val, thr)
        rows.append(m)

        if best_f1 is None or m["f1"] > best_f1["f1"]:
            best_f1 = m

        if m["recall"] >= 0.80:
            if best_recall_80 is None or m["precision"] > best_recall_80["precision"]:
                best_recall_80 = m

        if m["precision"] >= 0.20:
            if best_precision_20 is None or m["recall"] > best_precision_20["recall"]:
                best_precision_20 = m

    # appliquer les seuils choisis sur TEST
    chosen = {
        "best_f1_on_val": {
            "val": best_f1,
            "test": metrics_at_threshold(y_test, proba_test, best_f1["threshold"]),
        },
        "best_precision_with_recall>=0.80_on_val": None if best_recall_80 is None else {
            "val": best_recall_80,
            "test": metrics_at_threshold(y_test, proba_test, best_recall_80["threshold"]),
        },
        "best_recall_with_precision>=0.20_on_val": None if best_precision_20 is None else {
            "val": best_precision_20,
            "test": metrics_at_threshold(y_test, proba_test, best_precision_20["threshold"]),
        }
    }

    out = {"base": base, "chosen_thresholds": chosen}
    os.makedirs("reports", exist_ok=True)
    with open("reports/threshold_tuning.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("✅ Saved: reports/threshold_tuning.json")
    print("Base:", base)
    print("Chosen:", chosen)

if __name__ == "__main__":
    main()
