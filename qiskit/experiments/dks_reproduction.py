import itertools
import json
import time
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

data = load_wine()
X = StandardScaler().fit_transform(data.data[:, :6])
features = data.feature_names[:6]
W = abs(np.corrcoef(X, rowvar=False))
k = 3

qp = QuadraticProgram(name="dks_reproduction")

for feature in features:
    qp.binary_var(feature)

qp.linear_constraint(
    linear={feature: 1 for feature in features},
    sense="==",
    rhs=k,
    name="select_k"
)

qp.maximize(
    quadratic={
        (features[i], features[j]): float(W[i, j])
        for i in range(6)
        for j in range(i + 1, 6)
    }
)

start = time.perf_counter()

qaoa = QAOA(
    sampler=StatevectorSampler(seed=42),
    optimizer=COBYLA(maxiter=10),
    reps=1
)

qaoa_result = MinimumEigenOptimizer(qaoa).solve(qp)

qaoa_time = time.perf_counter() - start

qaoa_selected = [
    name.name
    for name, value in zip(qp.variables, qaoa_result.x)
    if value > 0.5
]

exact_score = -float("inf")
exact_selected = None

start = time.perf_counter()

for combo in itertools.combinations(range(6), k):
    score = sum(
        W[i, j]
        for i, j in itertools.combinations(combo, 2)
    )

    if score > exact_score:
        exact_score = score
        exact_selected = combo

exact_time = time.perf_counter() - start

exact_features = [features[i] for i in exact_selected]
ratio = qaoa_result.fval / exact_score

result = {
    "features": features,
    "k": k,
    "qaoa_selected": qaoa_selected,
    "qaoa_objective": float(qaoa_result.fval),
    "qaoa_time_seconds": qaoa_time,
    "exact_selected": exact_features,
    "exact_objective": float(exact_score),
    "exact_time_seconds": exact_time,
    "approximation_ratio": float(ratio)
}

print(json.dumps(result, indent=2))

with open(".\\results\\dks_reproduction.json", "w") as f:
    json.dump(result, f, indent=2)
