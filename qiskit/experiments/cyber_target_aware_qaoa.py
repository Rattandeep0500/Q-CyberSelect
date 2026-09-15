import json
import time
import pandas as pd

from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

graph = pd.read_csv(
    "results/unsw_feature_graph.csv",
    index_col=0
)

relevance = pd.read_csv(
    "results/unsw_feature_relevance.csv",
    index_col=0
).iloc[:, 0]

candidates = relevance.sort_values(ascending=False).head(8).index.tolist()
k = 4
lambda_redundancy = 0.5

qp = QuadraticProgram(name="cyber_target_aware_qaoa")

for feature in candidates:
    qp.binary_var(feature)

qp.linear_constraint(
    linear={feature: 1 for feature in candidates},
    sense="==",
    rhs=k,
    name="select_k"
)

linear = {
    feature: float(relevance[feature])
    for feature in candidates
}

quadratic = {}

for i in range(len(candidates)):
    for j in range(i + 1, len(candidates)):
        a = candidates[i]
        b = candidates[j]
        quadratic[(a, b)] = -lambda_redundancy * float(
            graph.loc[a, b]
        )

qp.maximize(
    linear=linear,
    quadratic=quadratic
)

start = time.perf_counter()

qaoa = QAOA(
    sampler=StatevectorSampler(seed=42),
    optimizer=COBYLA(maxiter=10),
    reps=1
)

result = MinimumEigenOptimizer(qaoa).solve(qp)

elapsed = time.perf_counter() - start

selected = [
    name.name
    for name, value in zip(qp.variables, result.x)
    if value > 0.5
]

print("Target-aware Cyber QAOA")
print("Candidates:", candidates)
print("Selected:", selected)
print("Objective:", result.fval)
print("Time:", elapsed)

output = {
    "candidates": candidates,
    "k": k,
    "lambda_redundancy": lambda_redundancy,
    "selected": selected,
    "objective": float(result.fval),
    "time_seconds": elapsed
}

with open(
    "results/cyber_target_aware_qaoa.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(output, f, indent=2)
