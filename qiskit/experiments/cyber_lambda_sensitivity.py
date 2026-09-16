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

records = []

for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    qp = QuadraticProgram(name=f"lambda_{lam}")

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
            a = candidates[i]
            b = candidates[j]
            quadratic[(a, b)] = -lam * float(graph.loc[a, b])

    qp.maximize(
        linear={
            feature: float(relevance[feature])
            for feature in candidates
        },
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

    records.append({
        "lambda": lam,
        "selected": selected,
        "objective": float(result.fval),
        "time_seconds": elapsed
    })

df = pd.DataFrame(records)

print(df.to_string(index=False))

df.to_csv(
    "results/cyber_lambda_sensitivity.csv",
    index=False
)

with open(
    "results/cyber_lambda_sensitivity.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(records, f, indent=2)
