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
X = StandardScaler().fit_transform(data.data)
features = data.feature_names

corr = abs(np.corrcoef(X, rowvar=False))

G = nx.Graph()
G.add_nodes_from(features)

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        G.add_edge(features[i], features[j], weight=float(corr[i, j]))

qp = QuadraticProgram(name="mincut_gtfs")

for feature in features:
    qp.binary_var(feature)

quadratic = {}

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        w = corr[i, j]
        quadratic[(features[i], features[j])] = 2.0 * float(w)

qp.minimize(
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

print("MinCut QAOA")
print("Selected:", selected)
print("Objective:", result.fval)
