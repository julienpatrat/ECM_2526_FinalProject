import os
import pandas as pd
import joblib

INPUT_CSV = os.path.join("Data", "processed", "year1_dedup.csv")
MODEL_PATH = os.path.join("reports", "models", "hgb_calibrated.joblib")
OUTPUT_CSV = os.path.join("Data", "processed", "year1_dedup_with_R_hgb.csv")
TARGET = "class"

def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(INPUT_CSV)
    X = df.drop(columns=[TARGET])
    R = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["R_hgb"] = R
    out.to_csv(OUTPUT_CSV, index=False)
    print("✅ Saved:", OUTPUT_CSV)
    print(out["R_hgb"].describe())

if __name__ == "__main__":
    main()
