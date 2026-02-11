import os, json
import pandas as pd

IN_CSV = os.path.join("Data", "processed", "us", "us_with_R_hgb.csv")
OUT_CSV = os.path.join("Data", "processed", "us", "company_lookup_us.csv")
OUT_JSON = os.path.join("Data", "processed", "us", "company_lookup_us.json")

def bucket(r):
    if r < 0.01: return "LOW"
    if r < 0.05: return "MEDIUM"
    if r < 0.20: return "HIGH"
    return "CRITICAL"

def main():
    df = pd.read_csv(IN_CSV)

    # 1 ligne par entreprise : dernière année observée
    df = df.sort_values(["company_id", "year"]).groupby("company_id", as_index=False).tail(1).copy()

    # créer un id numérique stable (rang)
    df = df.reset_index(drop=True)
    df["company_num_id"] = df.index + 1
    df["company_code"] = df["company_num_id"].apply(lambda x: f"US_{int(x):06d}")
    df["risk_bucket"] = df["R_hgb"].apply(bucket)

    lookup = df[["company_num_id", "company_code", "company_id", "year", "R_hgb", "risk_bucket"]].copy()
    lookup.to_csv(OUT_CSV, index=False)

    # JSON (clé = company_num_id)
    d = {
        str(int(r.company_num_id)): {
            "company_code": r.company_code,
            "company_name": r.company_id,
            "year": int(r.year),
            "R": float(r.R_hgb),
            "bucket": r.risk_bucket,
        }
        for r in lookup.itertuples(index=False)
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)

    print("✅ Saved:", OUT_CSV)
    print("✅ Saved:", OUT_JSON)
    print("Example:", list(d.items())[:1])

if __name__ == "__main__":
    main()
