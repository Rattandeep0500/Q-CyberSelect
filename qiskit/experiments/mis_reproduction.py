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

qp = QuadraticProgram(name="mis_qaoa")

for feature in features:
    qp.binary_var(feature)

for i in range(6):
    for j in range(i + 1, 6):
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
best_selected = None

for r in range(7):
    for combo in itertools.combinations(range(6), r):
        valid = True

        for i, j in itertools.combinations(combo, 2):
            if W[i, j] > 0:
                valid = False
                break

        if valid and r > best_size:
            best_size = r
            best_selected = combo

exact_selected = [features[i] for i in best_selected]

output = {
    "qaoa_selected": qaoa_selected,
    "qaoa_size": len(qaoa_selected),
    "qaoa_objective": float(result.fval),
    "qaoa_time_seconds": qaoa_time,
    "exact_selected": exact_selected,
    "exact_size": best_size,
    "match": set(qaoa_selected) == set(exact_selected)
}

print(json.dumps(output, indent=2))

with open(".\\results\\mis_reproduction.json", "w") as f:
    json.dump(output, f, indent=2)
