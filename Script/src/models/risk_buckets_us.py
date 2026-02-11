import os, json
import pandas as pd

IN_CSV = os.path.join("Data", "processed", "us", "us_with_R_hgb.csv")
REPORT_PATH = os.path.join("reports", "us_risk_buckets_report.json")

def bucket(r):
    if r < 0.01: return "LOW"
    if r < 0.05: return "MEDIUM"
    if r < 0.20: return "HIGH"
    return "CRITICAL"

def main():
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(IN_CSV)
    df["risk_bucket"] = df["R_hgb"].apply(bucket)

    report = {
        "bucket_share": (df["risk_bucket"].value_counts(normalize=True)).to_dict(),
        "bucket_counts": (df["risk_bucket"].value_counts()).to_dict(),
        "default_rate_by_bucket": (df.groupby("risk_bucket")["class"].mean()).to_dict(),
        "overall_positive_rate": float(df["class"].mean()),
        "rows": int(len(df)),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("✅ Saved:", REPORT_PATH)
    print(report)

if __name__ == "__main__":
    main()
