import os
import pandas as pd
import joblib

INPUT_CSV = os.path.join("Data", "processed", "year1.csv")
MODEL_PATH = os.path.join("reports", "models", "logreg_calibrated.joblib")
OUTPUT_CSV = os.path.join("Data", "processed", "year1_with_R.csv")
TARGET = "class"

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing {INPUT_CSV}. Run prepare_year1.py first.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run calibrate_logreg.py first.")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(INPUT_CSV)

    X = df.drop(columns=[TARGET])
    R = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["R"] = R
    out.to_csv(OUTPUT_CSV, index=False)

    print("✅ Saved:", OUTPUT_CSV)
    print("R summary:")
    print(out["R"].describe())

if __name__ == "__main__":
    main()
