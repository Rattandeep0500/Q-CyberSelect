import numpy as np
import networkx as nx
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

corr = abs(np.corrcoef(X, rowvar=False))

G = nx.Graph()
G.add_nodes_from(features)

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        if corr[i, j] > 0.5:
            G.add_edge(features[i], features[j], weight=float(corr[i, j]))

qp = QuadraticProgram(name="small_dks")

for feature in features:
    qp.binary_var(feature)

qp.linear_constraint(
    linear={feature: 1 for feature in features},
    sense="==",
    rhs=3,
    name="select_k"
)

qp.maximize(
    quadratic={
        (u, v): float(weight)
        for u, v, weight in G.edges(data="weight")
    }
)

qaoa = QAOA(
    sampler=StatevectorSampler(seed=42),
    optimizer=COBYLA(maxiter=10),
    reps=1
)

result = MinimumEigenOptimizer(qaoa).solve(qp)

print("Selected features:")
for name, value in zip(qp.variables, result.x):
    if value > 0.5:
        print(name.name)

print("Objective:", result.fval)
