import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

graph = pd.read_csv(
    "results/unsw_feature_graph.csv",
    index_col=0
)

candidates = list(graph.columns[:8])
k = 4

qp = QuadraticProgram(name="cyber_dks_qaoa")

for feature in candidates:
    qp.binary_var(feature)

qp.linear_constraint(
    linear={feature: 1 for feature in candidates},
    sense="==",
    rhs=k,
    name="select_k"
)

quadratic = {}

for i in range(len(candidates)):
    for j in range(i + 1, len(candidates)):
        quadratic[(candidates[i], candidates[j])] = float(
            graph.loc[candidates[i], candidates[j]]
        )

qp.maximize(quadratic=quadratic)

start = time.perf_counter()

qaoa = QAOA(
    sampler=StatevectorSampler(seed=42),
    optimizer=COBYLA(maxiter=5),
    reps=1
)

result = MinimumEigenOptimizer(qaoa).solve(qp)

qaoa_time = time.perf_counter() - start

selected = [
    name.name
    for name, value in zip(qp.variables, result.x)
    if value > 0.5
]

exact_score = -float("inf")
exact_selected = None

for combo in __import__("itertools").combinations(range(len(candidates)), k):
    score = sum(
        graph.loc[candidates[i], candidates[j]]
        for i, j in __import__("itertools").combinations(combo, 2)
    )

    if score > exact_score:
        exact_score = float(score)
        exact_selected = combo

exact_features = [candidates[i] for i in exact_selected]

output = {
    "candidate_count": len(candidates),
    "k": k,
    "candidates": candidates,
    "qaoa_selected": selected,
    "qaoa_objective": float(result.fval),
    "qaoa_time_seconds": qaoa_time,
    "exact_selected": exact_features,
    "exact_objective": exact_score,
    "approximation_ratio": float(result.fval / exact_score),
    "exact_match": set(selected) == set(exact_features)
}

print(json.dumps(output, indent=2))

with open(
    "results/cyber_dks_qaoa.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(output, f, indent=2)
