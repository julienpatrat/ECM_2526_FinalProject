import os
import pandas as pd

INPUT = os.path.join("Data", "processed", "year1_with_R_hgb.csv")
OUTPUT = os.path.join("Data", "processed", "year1_scored_hgb.csv")

def bucket(r: float) -> str:
    if r < 0.01:
        return "LOW"
    elif r < 0.05:
        return "MEDIUM"
    elif r < 0.20:
        return "HIGH"
    else:
        return "CRITICAL"

def main():
    if not os.path.exists(INPUT):
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT)
    if "R_hgb" not in df.columns:
        raise ValueError("Column 'R_hgb' not found in input CSV.")

    df["risk_bucket"] = df["R_hgb"].apply(bucket)
    df.to_csv(OUTPUT, index=False)

    print("✅ Saved:", OUTPUT)
    print("\nBucket distribution (share):")
    print((df["risk_bucket"].value_counts(normalize=True)).to_string())

    print("\nDefault rate in each bucket (mean(class)):")
    print(df.groupby("risk_bucket")["class"].mean().rename("default_rate").to_string())

    print("\nCounts by bucket:")
    print(df["risk_bucket"].value_counts().to_string())

if __name__ == "__main__":
    main()
