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

corr = abs(__import__("numpy").corrcoef(X, rowvar=False))

G = nx.Graph()
G.add_nodes_from(features)

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        if corr[i, j] > 0.5:
            G.add_edge(features[i], features[j], weight=float(corr[i, j]))

k = 5

qp = QuadraticProgram(name="wine_densest_k_subgraph")

for feature in features:
    qp.binary_var(feature)

qp.linear_constraint(
    linear={feature: 1 for feature in features},
    sense="==",
    rhs=k,
    name="select_k_features"
)

qp.maximize(
    quadratic={
        (u, v): float(weight)
        for u, v, weight in G.edges(data="weight")
    }
)

sampler = StatevectorSampler(seed=42)

qaoa = QAOA(
    sampler=sampler,
    optimizer=COBYLA(maxiter=100),
    reps=2
)

optimizer = MinimumEigenOptimizer(qaoa)

result = optimizer.solve(qp)

print("Selected features:")
selected = []

for name, value in zip(qp.variables, result.x):
    if value > 0.5:
        selected.append(name.name)

for feature in selected:
    print(feature)

print("\nNumber selected:", len(selected))
print("Objective:", result.fval)
