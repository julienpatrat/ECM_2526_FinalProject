import os
import pandas as pd
import joblib

IN_CSV = os.path.join("Data", "processed", "us", "us.csv")
OUT_CSV = os.path.join("Data", "processed", "us", "us_with_R_hgb.csv")
MODEL_PATH = os.path.join("models", "us", "hgb_calibrated.joblib")

TARGET = "class"

def main():
    df = pd.read_csv(IN_CSV)
    model = joblib.load(MODEL_PATH)

    X = df.drop(columns=["company_id", "year", TARGET], errors="ignore")
    df["R_hgb"] = model.predict_proba(X)[:, 1]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("✅ Saved:", OUT_CSV)
    print(df["R_hgb"].describe())

if __name__ == "__main__":
    main()
