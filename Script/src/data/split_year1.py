import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = os.path.join("Data", "processed", "year1.csv")
OUT_DIR = os.path.join("Data", "processed", "splits")
TARGET = "class"
RANDOM_STATE = 42

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing {INPUT_CSV}. Run prepare_year1.py first.")

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # 80% train+val / 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Sur les 80%, on fait 75/25 => 60% train / 20% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_STATE
    )

    train = X_train.copy(); train[TARGET] = y_train.values
    val = X_val.copy();     val[TARGET] = y_val.values
    test = X_test.copy();   test[TARGET] = y_test.values

    train_path = os.path.join(OUT_DIR, "train.csv")
    val_path   = os.path.join(OUT_DIR, "val.csv")
    test_path  = os.path.join(OUT_DIR, "test.csv")

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    info = {
        "random_state": RANDOM_STATE,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "positive_rate": {
            "train": float(train[TARGET].mean()),
            "val": float(val[TARGET].mean()),
            "test": float(test[TARGET].mean()),
        }
    }
    with open(os.path.join(OUT_DIR, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("✅ Splits created:")
    print(train_path, val_path, test_path)
    print("Positive rates:", info["positive_rate"])

if __name__ == "__main__":
    main()
