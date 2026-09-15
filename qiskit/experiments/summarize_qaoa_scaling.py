import json
import pandas as pd

with open("results/cyber_qaoa_exact_scaling.json", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for r in data:
    rows.append({
        "candidates": r["candidate_count"],
        "k": r["k"],
        "qaoa_objective": r["qaoa_objective"],
        "exact_objective": r["exact_objective"],
        "approximation_ratio": r["approximation_ratio"],
        "exact_match": r["exact_match"],
        "qaoa_time_seconds": r["qaoa_time_seconds"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    "results/cyber_qaoa_scaling_summary.csv",
    index=False
)

print(df.to_string(index=False))
