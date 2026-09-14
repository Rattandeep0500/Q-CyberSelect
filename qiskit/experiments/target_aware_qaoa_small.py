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

mi = mutual_info_classif(X, y, random_state=42)
top = np.argsort(mi)[::-1][:6]

features = [data.feature_names[i] for i in top]
X_top = X[:, top]
corr = abs(np.corrcoef(X_top, rowvar=False))
mi_top = mi[top]
mi_top = mi_top / mi_top.max()

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

for i in range(6):
    for j in range(i + 1, 6):
        quadratic[(features[i], features[j])] = float(
            -0.3 * corr[i, j]
        )

qp.maximize(
    linear={
        features[i]: float(mi_top[i])
        for i in range(6)
    },
    quadratic=quadratic
)

qaoa = QAOA(
    sampler=StatevectorSampler(seed=42),
    optimizer=COBYLA(maxiter=5),
    reps=1
)

result = MinimumEigenOptimizer(qaoa).solve(qp)

selected = [
    name.name
    for name, value in zip(qp.variables, result.x)
    if value > 0.5
]

print("Target-aware QAOA")
print("Candidates:", features)
print("Selected:", selected)
print("Objective:", result.fval)
