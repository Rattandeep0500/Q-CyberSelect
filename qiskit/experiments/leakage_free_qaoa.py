import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

data = load_wine()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42,
    stratify=data.target
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mi = mutual_info_classif(X_train_scaled, y_train, random_state=42)

top = np.argsort(mi)[::-1][:6]

features = [data.feature_names[i] for i in top]
mi_top = mi[top]
mi_top = mi_top / mi_top.max()

corr = abs(np.corrcoef(X_train_scaled[:, top], rowvar=False))

qp = QuadraticProgram(name="leakage_free_qaoa")

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

indices = [data.feature_names.index(f) for f in selected]

model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled[:, indices], y_train)

pred = model.predict(X_test_scaled[:, indices])

print("Leakage-free QAOA")
print("Candidates:", features)
print("Selected:", selected)
print("Accuracy:", accuracy_score(y_test, pred))
print("F1:", f1_score(y_test, pred, average="weighted"))
