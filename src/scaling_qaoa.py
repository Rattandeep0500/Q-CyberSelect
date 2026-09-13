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
ranking = np.argsort(mi)[::-1]

for n in [6, 7, 8, 9, 10]:
    indices = ranking[:n]
    features = [data.feature_names[i] for i in indices]

    corr = abs(np.corrcoef(X[:, indices], rowvar=False))
    mi_top = mi[indices] / mi[indices].max()

    qp = QuadraticProgram(name=f"qaoa_n{n}")

    for feature in features:
        qp.binary_var(feature)

    qp.linear_constraint(
        linear={feature: 1 for feature in features},
        sense="==",
        rhs=3,
        name="select_k"
    )

    quadratic = {}

    for i in range(n):
        for j in range(i + 1, n):
            weight = (
                0.7 * ((mi_top[i] + mi_top[j]) / 2)
                + 0.3 * corr[i, j]
            )
            quadratic[(features[i], features[j])] = float(weight)

    qp.maximize(quadratic=quadratic)

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

    print(f"n={n}")
    print("QAOA:", selected)
    print("Objective:", result.fval)
    print()
