import itertools
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

qp = QuadraticProgram(name="dks_qaoa")

for feature in features:
    qp.binary_var(feature)

qp.linear_constraint(
    linear={feature: 1 for feature in features},
    sense="==",
    rhs=k,
    name="select_k"
)

quadratic = {}

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        quadratic[(features[i], features[j])] = float(W[i, j])

qp.maximize(quadratic=quadratic)

qaoa = QAOA(
    sampler=StatevectorSampler(seed=42),
    optimizer=COBYLA(maxiter=10),
    reps=1
)

result = MinimumEigenOptimizer(qaoa).solve(qp)

selected = [
    name.name
    for name, value in zip(qp.variables, result.x)
    if value > 0.5
]

exact_score = -float("inf")
exact_selected = None

for combo in itertools.combinations(range(len(features)), k):
    score = sum(
        W[i, j]
        for i, j in itertools.combinations(combo, 2)
    )

    if score > exact_score:
        exact_score = score
        exact_selected = combo

exact_features = [features[i] for i in exact_selected]

print("Densest-k-Subgraph")
print("QAOA:", selected)
print("QAOA objective:", result.fval)
print("Exact:", exact_features)
print("Exact objective:", exact_score)
print("Approximation ratio:", result.fval / exact_score)
