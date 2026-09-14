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

indices = [4, 5, 6, 8, 9, 10]
features = [data.feature_names[i] for i in indices]

X = StandardScaler().fit_transform(data.data[:, indices])
W = abs(np.corrcoef(X, rowvar=False))

threshold = 0.5
n = len(features)

edges = []

for i in range(n):
    for j in range(i + 1, n):
        if W[i, j] > threshold:
            edges.append((i, j))

qp = QuadraticProgram(name="mis_qaoa")

for feature in features:
    qp.binary_var(feature)

for i, j in edges:
    qp.linear_constraint(
        linear={
            features[i]: 1,
            features[j]: 1
        },
        sense="<=",
        rhs=1,
        name=f"edge_{i}_{j}"
    )

qp.maximize(
    linear={feature: 1 for feature in features}
)

start = time.perf_counter()

qaoa = QAOA(
    sampler=StatevectorSampler(seed=42),
    optimizer=COBYLA(maxiter=10),
    reps=1
)

result = MinimumEigenOptimizer(qaoa).solve(qp)

qaoa_time = time.perf_counter() - start

qaoa_selected = [
    name.name
    for name, value in zip(qp.variables, result.x)
    if value > 0.5
]

best_size = -1
best_solution = None

start = time.perf_counter()

for r in range(n + 1):
    for combo in itertools.combinations(range(n), r):
        valid = True

        for i, j in edges:
            if i in combo and j in combo:
                valid = False
                break

        if valid and r > best_size:
            best_size = r
            best_solution = combo

exact_time = time.perf_counter() - start

exact_selected = [features[i] for i in best_solution]

print("MIS QAOA")
print("Features:", features)
print("Edges:", [(features[i], features[j]) for i, j in edges])
print("QAOA selected:", qaoa_selected)
print("QAOA size:", len(qaoa_selected))
print("Exact selected:", exact_selected)
print("Exact size:", best_size)
print("QAOA objective:", result.fval)
print("QAOA time:", qaoa_time)
print("Exact time:", exact_time)
print("Optimal size match:", len(qaoa_selected) == best_size)

output = {
    "features": features,
    "edges": [(features[i], features[j]) for i, j in edges],
    "qaoa_selected": qaoa_selected,
    "qaoa_size": len(qaoa_selected),
    "exact_selected": exact_selected,
    "exact_size": best_size,
    "qaoa_objective": float(result.fval),
    "qaoa_time_seconds": qaoa_time,
    "exact_time_seconds": exact_time,
    "optimal_size_match": len(qaoa_selected) == best_size
}

with open(".\\results\\mis_reproduction.json", "w") as f:
    json.dump(output, f, indent=2)
