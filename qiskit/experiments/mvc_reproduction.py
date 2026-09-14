import itertools
import json
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
W = abs(__import__("numpy").corrcoef(X, rowvar=False))

threshold = 0.5
n = len(features)

edges = [
    (i, j)
    for i in range(n)
    for j in range(i + 1, n)
    if W[i, j] > threshold
]

qp = QuadraticProgram(name="mvc_qaoa")

for feature in features:
    qp.binary_var(feature)

for i, j in edges:
    qp.linear_constraint(
        linear={
            features[i]: 1,
            features[j]: 1
        },
        sense=">=",
        rhs=1,
        name=f"cover_{i}_{j}"
    )

qp.minimize(
    linear={feature: 1 for feature in features}
)

qaoa = QAOA(
    sampler=StatevectorSampler(seed=42),
    optimizer=COBYLA(maxiter=10),
    reps=1
)

result = MinimumEigenOptimizer(qaoa).solve(qp)

qaoa_selected = [
    name.name
    for name, value in zip(qp.variables, result.x)
    if value > 0.5
]

best_size = n + 1
best_solution = None

for r in range(n + 1):
    for combo in itertools.combinations(range(n), r):
        if all(i in combo or j in combo for i, j in edges):
            if r < best_size:
                best_size = r
                best_solution = combo

exact_selected = [features[i] for i in best_solution]

output = {
    "features": features,
    "edges": [(features[i], features[j]) for i, j in edges],
    "qaoa_selected": qaoa_selected,
    "qaoa_size": len(qaoa_selected),
    "exact_selected": exact_selected,
    "exact_size": best_size,
    "qaoa_objective": float(result.fval),
    "optimal_size_match": len(qaoa_selected) == best_size
}

print(json.dumps(output, indent=2))

with open(".\\results\\mvc_reproduction.json", "w") as f:
    json.dump(output, f, indent=2)
