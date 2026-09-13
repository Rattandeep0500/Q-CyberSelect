import numpy as np
import networkx as nx
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

corr = abs(np.corrcoef(X, rowvar=False))
mi = mutual_info_classif(X, y, random_state=42)
mi = mi / mi.max()

G = nx.Graph()
G.add_nodes_from(features)

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        relevance = (mi[i] + mi[j]) / 2
        redundancy = corr[i, j]
        weight = 0.7 * relevance + 0.3 * redundancy
        G.add_edge(features[i], features[j], weight=float(weight))

k = 3

qp = QuadraticProgram(name="relevance_graph_qaoa")

for feature in features:
    qp.binary_var(feature)

qp.linear_constraint(
    linear={feature: 1 for feature in features},
    sense="==",
    rhs=k,
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

selected = []

for name, value in zip(qp.variables, result.x):
    if value > 0.5:
        selected.append(name.name)

print("Selected features:")
for feature in selected:
    print(feature)

print("Objective:", result.fval)
