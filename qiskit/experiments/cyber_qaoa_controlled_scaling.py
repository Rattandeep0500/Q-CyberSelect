import itertools
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

k = 3
candidate_sizes = [6, 8]

records = []

for n in candidate_sizes:
    candidates = relevance.sort_values(ascending=False).head(n).index.tolist()

    qp = QuadraticProgram(name=f"cyber_dks_{n}")

    for feature in candidates:
        qp.binary_var(feature)

    qp.linear_constraint(
        linear={feature: 1 for feature in candidates},
        sense="==",
        rhs=k,
        name="select_k"
    )

    qp.maximize(
        quadratic={
            (candidates[i], candidates[j]): float(
                graph.loc[candidates[i], candidates[j]]
            )
            for i in range(n)
            for j in range(i + 1, n)
        }
    )

    start = time.perf_counter()

    qaoa = QAOA(
        sampler=StatevectorSampler(seed=42),
        optimizer=COBYLA(maxiter=3),
        reps=1
    )

    result = MinimumEigenOptimizer(qaoa).solve(qp)

    qaoa_time = time.perf_counter() - start

    qaoa_selected = [
        name.name
        for name, value in zip(qp.variables, result.x)
        if value > 0.5
    ]

    exact_score = -float("inf")
    exact_selected = None

    start = time.perf_counter()

    for combo in itertools.combinations(range(n), k):
        score = sum(
            graph.loc[candidates[i], candidates[j]]
            for i, j in itertools.combinations(combo, 2)
        )

        if score > exact_score:
            exact_score = float(score)
            exact_selected = combo

    exact_time = time.perf_counter() - start

    exact_features = [candidates[i] for i in exact_selected]

    records.append({
        "candidate_count": n,
        "k": k,
        "qaoa_selected": qaoa_selected,
        "qaoa_objective": float(result.fval),
        "exact_selected": exact_features,
        "exact_objective": exact_score,
        "approximation_ratio": float(result.fval / exact_score),
        "exact_match": set(qaoa_selected) == set(exact_features),
        "qaoa_time_seconds": qaoa_time,
        "exact_time_seconds": exact_time
    })

print(json.dumps(records, indent=2))

with open("results/cyber_qaoa_controlled_scaling.json", "w") as f:
    json.dump(records, f, indent=2)
