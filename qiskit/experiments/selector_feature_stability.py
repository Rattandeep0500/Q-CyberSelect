import json
from collections import Counter
import pandas as pd

df = pd.read_csv(
    "results/nested_selector_comparison.csv"
)

methods = df["method"].unique()

records = []

for method in methods:

    method_df = df[
        df["method"] == method
    ]

    counter = Counter()

    for value in method_df["selected"]:
        features = value.strip("[]").replace("'", "").split(",")

        for feature in features:
            feature = feature.strip()

            if feature:
                counter[feature] += 1

    folds = method_df["fold"].nunique()

    for feature, count in counter.items():
        records.append({
            "method": method,
            "feature": feature,
            "selected_count": count,
            "selection_frequency": count / folds
        })

result = pd.DataFrame(records)

result = result.sort_values(
    ["method", "selection_frequency"],
    ascending=[True, False]
)

print(result.to_string(index=False))

result.to_csv(
    "results/selector_feature_frequency.csv",
    index=False
)

summary = []

for method in methods:

    subset = result[
        result["method"] == method
    ]

    stable = subset[
        subset["selection_frequency"] == 1.0
    ]["feature"].tolist()

    summary.append({
        "method": method,
        "fully_stable_features": stable,
        "stable_feature_count": len(stable)
    })

print()
print("STABILITY SUMMARY")
print(json.dumps(summary, indent=2))

with open(
    "results/selector_stability_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )
