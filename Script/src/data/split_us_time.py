import os, json
import pandas as pd

IN_CSV = os.path.join("Data", "processed", "us", "us.csv")
OUT_DIR = os.path.join("Data", "processed", "us", "splits")
INFO_PATH = os.path.join("Data", "processed", "us", "split_info.json")

# Split temporel recommandé (train 1999-2011, val 2012-2014, test 2015-2018+)
TRAIN_MAX = 2011
VAL_MIN, VAL_MAX = 2012, 2014
TEST_MIN = 2015

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(IN_CSV)

    train = df[df["year"] <= TRAIN_MAX].copy()
    val = df[(df["year"] >= VAL_MIN) & (df["year"] <= VAL_MAX)].copy()
    test = df[df["year"] >= TEST_MIN].copy()

    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    info = {
        "in_csv": IN_CSV,
        "splits": {
            "train_years": f"..{TRAIN_MAX}",
            "val_years": f"{VAL_MIN}..{VAL_MAX}",
            "test_years": f"{TEST_MIN}..",
        },
        "sizes": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "positive_rate": {
            "train": float(train["class"].mean()) if len(train) else None,
            "val": float(val["class"].mean()) if len(val) else None,
            "test": float(test["class"].mean()) if len(test) else None,
        },
    }

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("✅ Saved splits to:", OUT_DIR)
    print(info)

if __name__ == "__main__":
    main()
