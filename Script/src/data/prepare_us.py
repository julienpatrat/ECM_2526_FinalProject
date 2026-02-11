import os, json, re
import pandas as pd

RAW_PATH = os.path.join("Data", "source", "us", "american_bankruptcy_dataset.csv")
OUT_DIR = os.path.join("Data", "processed", "us")
OUT_CSV = os.path.join(OUT_DIR, "us.csv")
AUDIT_PATH = os.path.join("reports", "us_audit.json")

YEAR_CANDIDATES = ["year", "Year", "fyear", "FYEAR", "fiscal_year"]
TARGET_CANDIDATES = ["Bankruptcy", "bankruptcy", "target", "label", "class", "failed", "status", "status_label"]
COMPANY_ID_CANDIDATES = ["company_id", "Company_ID", "gvkey", "GVKEY", "permno", "PERMNO", "tic", "ticker", "Ticker", "id", "ID", "company_name"]

KEYWORDS_POS = re.compile(r"(bankrupt|bankruptcy|failed|failure|default|liquidat|insolv)", re.IGNORECASE)

def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def encode_target(series: pd.Series) -> pd.Series:
    # try numeric first
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().mean() > 0.95:
        # assume it's 0/1 already (or close)
        return s_num.fillna(0).astype(int).clip(0, 1)

    # else string mapping via keywords
    s_str = series.astype(str).fillna("")
    y = s_str.apply(lambda x: 1 if KEYWORDS_POS.search(x) else 0).astype(int)
    return y

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    cols = list(df.columns)

    year_col = pick_col(cols, YEAR_CANDIDATES)
    target_col = pick_col(cols, TARGET_CANDIDATES)
    company_col = pick_col(cols, COMPANY_ID_CANDIDATES)

    if year_col is None or target_col is None or company_col is None:
        raise ValueError(
            f"Could not detect required columns.\n"
            f"Found columns: {cols}\n"
            f"Detected year={year_col}, target={target_col}, company={company_col}\n"
        )

    df[year_col] = df[year_col].astype(int)
    df[company_col] = df[company_col].astype(str)

    # encode target properly
    df[target_col] = encode_target(df[target_col])

    meta_cols = [company_col, year_col, target_col]
    numeric_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    out = df[meta_cols + numeric_cols].copy()
    out = out.rename(columns={company_col: "company_id", year_col: "year", target_col: "class"})

    out.to_csv(OUT_CSV, index=False)

    audit = {
        "raw_path": RAW_PATH,
        "rows": int(out.shape[0]),
        "n_features": int(len(numeric_cols)),
        "year_min": int(out["year"].min()),
        "year_max": int(out["year"].max()),
        "positive_rate": float(out["class"].mean()),
        "target_value_counts": out["class"].value_counts().to_dict(),
        "company_id_col_used": company_col,
        "detected_cols": {"year": year_col, "target": target_col, "company_id": company_col},
        "feature_preview": numeric_cols[:15],
    }
    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("✅ Saved:", OUT_CSV)
    print("✅ Saved:", AUDIT_PATH)
    print("Target counts:", audit["target_value_counts"])
    print("Positive rate:", audit["positive_rate"])

if __name__ == "__main__":
    main()
