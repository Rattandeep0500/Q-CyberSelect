import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

data = load_wine()
X = StandardScaler().fit_transform(data.data)
y = data.target
features = data.feature_names

mi = mutual_info_classif(X, y, random_state=42)
mi = mi / mi.max()
corr = abs(np.corrcoef(X, rowvar=False))

qp = QuadraticProgram(name="target_aware_qaoa")

for feature in features:
    qp.binary_var(feature)

qp.linear_constraint(
    linear={feature: 1 for feature in features},
    sense="==",
    rhs=3,
    name="select_k"
)

quadratic = {}

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        quadratic[(features[i], features[j])] = -0.5 * float(corr[i, j])

qp.maximize(
    linear={
        features[i]: float(mi[i])
        for i in range(len(features))
    },
    quadratic=quadratic
)

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

print("Target-aware QAOA")
print("Selected features:", selected)
print("Objective:", result.fval)
