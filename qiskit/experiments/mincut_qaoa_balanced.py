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

qp = QuadraticProgram(name="balanced_mincut")

for feature in features:
    qp.binary_var(feature)

linear = {feature: 0.0 for feature in features}
quadratic = {}

for i in range(6):
    for j in range(i + 1, 6):
        w = float(W[i, j])
        linear[features[i]] += w
        linear[features[j]] += w
        quadratic[(features[i], features[j])] = -2.0 * w

qp.linear_constraint(
    linear={feature: 1 for feature in features},
    sense="==",
    rhs=3,
    name="balanced_partition"
)

qp.minimize(
    linear=linear,
    quadratic=quadratic
)

qaoa = QAOA(
    sampler=StatevectorSampler(seed=42),
    optimizer=COBYLA(maxiter=5),
    reps=1
)

result = MinimumEigenOptimizer(qaoa).solve(qp)

print("Balanced MinCut QAOA")
print("Partition A:")

for name, value in zip(qp.variables, result.x):
    if value < 0.5:
        print(name.name)

print("Partition B:")

for name, value in zip(qp.variables, result.x):
    if value >= 0.5:
        print(name.name)

print("Objective:", result.fval)
