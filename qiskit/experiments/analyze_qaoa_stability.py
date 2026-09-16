import json
import itertools
import pandas as pd

with open(
    "results/qaoa_seed_stability.json",
    encoding="utf-8"
) as f:
    runs = json.load(f)

sets = [
    set(r["selected"])
    for r in runs
]

features = sorted(
    set().union(*sets)
)

frequency = []

for feature in features:
    count = sum(
        feature in s
        for s in sets
    )

    frequency.append({
        "feature": feature,
        "selected_count": count,
        "selection_frequency": count / len(sets)
    })

frequency_df = pd.DataFrame(frequency).sort_values(
    "selection_frequency",
    ascending=False
)

jaccard_values = []

for a, b in itertools.combinations(sets, 2):
    intersection = len(a & b)
    union = len(a | b)

    jaccard_values.append(
        intersection / union
    )

objectives = [
    r["objective"]
    for r in runs
]

summary = {
    "runs": len(runs),
    "features_per_run": len(next(iter(sets))),
    "mean_objective": sum(objectives) / len(objectives),
    "min_objective": min(objectives),
    "max_objective": max(objectives),
    "mean_pairwise_jaccard": sum(jaccard_values) / len(jaccard_values),
    "min_pairwise_jaccard": min(jaccard_values),
    "max_pairwise_jaccard": max(jaccard_values)
}

print("FEATURE SELECTION FREQUENCY")
print(frequency_df.to_string(index=False))

print()
print("STABILITY SUMMARY")

for key, value in summary.items():
    print(f"{key}: {value}")

frequency_df.to_csv(
    "results/qaoa_feature_selection_frequency.csv",
    index=False
)

with open(
    "results/qaoa_stability_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)
