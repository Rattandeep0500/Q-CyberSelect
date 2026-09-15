import pandas as pd
import json
from pathlib import Path

base = Path("data/unsw_nb15")

train_path = base / "UNSW_NB15_training-set.csv"
test_path = base / "UNSW_NB15_testing-set.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("TRAIN SHAPE:", train.shape)
print("TEST SHAPE:", test.shape)
print()
print("COLUMNS:")
for i, col in enumerate(train.columns, 1):
    print(f"{i:02d}. {col}")
print()
print("TARGET DISTRIBUTION:")
print(train["label"].value_counts().to_dict())
print()
print("ATTACK CATEGORIES:")
print(train["attack_cat"].value_counts(dropna=False).to_dict())
print()
print("DTYPES:")
print(train.dtypes.to_string())

profile = {
    "train_rows": int(train.shape[0]),
    "train_columns": int(train.shape[1]),
    "test_rows": int(test.shape[0]),
    "test_columns": int(test.shape[1]),
    "columns": train.columns.tolist(),
    "label_distribution": train["label"].value_counts().to_dict(),
    "attack_categories": train["attack_cat"].value_counts(dropna=False).to_dict()
}

Path("results").mkdir(exist_ok=True)

with open("results/unsw_nb15_profile.json", "w", encoding="utf-8") as f:
    json.dump(profile, f, indent=2, default=str)
