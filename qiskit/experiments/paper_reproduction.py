import networkx as nx
from qiskit_optimization import QuadraticProgram

features = ["f1", "f2", "f3", "f4"]

edges = [
    ("f1", "f2", 0.90),
    ("f1", "f3", 0.70),
    ("f2", "f3", 0.80),
    ("f2", "f4", 0.20),
    ("f3", "f4", 0.30),
]

G = nx.Graph()
G.add_nodes_from(features)
G.add_weighted_edges_from(edges)

for u, v, data in G.edges(data=True):
    print(f"{u} -- {v}: {data['weight']}")

k = 2

qp = QuadraticProgram(name="densest_k_subgraph")

for feature in features:
    qp.binary_var(name=feature)

quadratic = {}

for u, v, weight in G.edges(data="weight"):
    quadratic[(u, v)] = float(weight)

qp.linear_constraint(
    linear={feature: 1 for feature in features},
    sense="==",
    rhs=k,
    name="select_k_features"
)

qp.maximize(
    linear={},
    quadratic=quadratic
)

print(qp)
