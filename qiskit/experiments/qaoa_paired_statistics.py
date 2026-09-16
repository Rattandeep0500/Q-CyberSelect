import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel

df = pd.read_csv("results/nested_selector_comparison.csv")

methods = ["QAOA-4", "MI-4", "ANOVA-4", "L1-4", "RF-4"]

pivot = df.pivot(
    index="fold",
    columns="method",
    values="f1"
).sort_index()

records = []

for method in methods:
    if method == "QAOA-4":
        continue

    qaoa = pivot["QAOA-4"].to_numpy(dtype=float)
    baseline = pivot[method].to_numpy(dtype=float)

    differences = qaoa - baseline

    mean_difference = float(np.mean(differences))
    std_difference = float(np.std(differences, ddof=1))
    mean_abs_difference = float(np.mean(np.abs(differences)))

    if std_difference > 0:
        cohens_d = float(mean_difference / std_difference)
    else:
        cohens_d = 0.0

    paired_t = ttest_rel(qaoa, baseline)

    try:
        wilcoxon_result = wilcoxon(
            qaoa,
            baseline,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided"
        )
        wilcoxon_stat = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)
    except ValueError:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0

    records.append({
        "comparison": f"QAOA-4 vs {method}",
        "qaoa_mean_f1": float(np.mean(qaoa)),
        "baseline_mean_f1": float(np.mean(baseline)),
        "mean_f1_difference": mean_difference,
        "std_difference": std_difference,
        "mean_absolute_difference": mean_abs_difference,
        "cohens_d_paired": cohens_d,
        "paired_t_statistic": float(paired_t.statistic),
        "paired_t_pvalue": float(paired_t.pvalue),
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_p
    })

result = pd.DataFrame(records)

print("FOLD-LEVEL F1")
print(pivot.to_string())
print()
print("PAIRED STATISTICAL COMPARISONS")
print(result.to_string(index=False))

result.to_csv(
    "results/qaoa_paired_statistics.csv",
    index=False
)

with open(
    "results/qaoa_paired_statistics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        records,
        f,
        indent=2
    )

summary = {
    "n_folds": int(len(pivot)),
    "comparisons": records,
    "note": "Exploratory paired analysis based on three shared CV folds."
}

with open(
    "results/qaoa_paired_statistics_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )
