import json
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

candidates = relevance.sort_values(
    ascending=False
).head(8).index.tolist()

k = 4
seeds = [1, 7, 21, 42, 100]

records = []

qp = QuadraticProgram(name="seed_stability")

for feature in candidates:
    qp.binary_var(feature)

qp.linear_constraint(
    linear={feature: 1 for feature in candidates},
    sense="==",
    rhs=k,
    name="select_k"
)

qp.maximize(
    linear={feature: float(relevance[feature]) for feature in candidates}
)

for seed in seeds:

    qaoa = QAOA(
        sampler=StatevectorSampler(seed=seed),
        optimizer=COBYLA(maxiter=10),
        reps=1
    )

    result = MinimumEigenOptimizer(qaoa).solve(qp)

    selected = [
        name.name
        for name, value in zip(qp.variables, result.x)
        if value > 0.5
    ]

    records.append({
        "seed": seed,
        "selected": selected,
        "objective": float(result.fval)
    })

print(json.dumps(records, indent=2))

with open(
    "results/qaoa_seed_stability.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(records, f, indent=2)
