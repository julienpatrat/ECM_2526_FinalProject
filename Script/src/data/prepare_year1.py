import os
import json
import zipfile
from io import StringIO

import pandas as pd
from scipy.io import arff


ZIP_PATH = os.path.join("Data", "source", "polish+companies+bankruptcy+data.zip")
ARFF_NAME = "1year.arff"
OUTPUT_DIR = os.path.join("Data", "processed")


def load_year1_from_zip(zip_path: str, arff_name: str = "1year.arff") -> pd.DataFrame:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        candidates = [n for n in names if n.lower().endswith(arff_name.lower())]
        if not candidates:
            raise FileNotFoundError(f"{arff_name} not found in zip. Found: {names[:20]} ...")
        arff_path_in_zip = candidates[0]
        raw_bytes = z.read(arff_path_in_zip)

    text = raw_bytes.decode("utf-8", errors="replace")
    data, meta = arff.loadarff(StringIO(text))
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            )

    return df


def basic_audit(df: pd.DataFrame) -> dict:
    target_col = df.columns[-1]
    y = pd.to_numeric(df[target_col], errors="coerce")

    audit = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target_col": str(target_col),
        "overall_missing_rate": float(df.isna().mean().mean()),
        "missing_rate_by_col": (df.isna().mean().sort_values(ascending=False)).to_dict(),
        "target_value_counts": df[target_col].value_counts(dropna=False).to_dict(),
    }

    if y.notna().any():
        audit["target_positive_rate"] = float((y == 1).mean())

    return audit


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_year1_from_zip(ZIP_PATH, ARFF_NAME)

    out_csv = os.path.join(OUTPUT_DIR, "year1.csv")
    df.to_csv(out_csv, index=False)

    audit = basic_audit(df)
    out_audit = os.path.join(OUTPUT_DIR, "audit_year1.json")
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("✅ Done.")
    print(f"- CSV saved to: {out_csv}")
    print(f"- Audit saved to: {out_audit}")
    print(f"- Target column detected: {audit['target_col']}")
    if "target_positive_rate" in audit:
        print(f"- Positive (bankruptcy) rate: {audit['target_positive_rate']:.4f}")


if __name__ == "__main__":
    main()
