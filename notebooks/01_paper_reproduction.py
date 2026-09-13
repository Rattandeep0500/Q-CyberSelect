import numpy as np
import networkx as nx
from qiskit import __version__ as qiskit_version
from qiskit_aer import Aer
from qiskit_optimization import __version__ as optimization_version

print("Q-CyberSelect")
print("=" * 40)
print("Qiskit:", qiskit_version)
print("Qiskit Optimization:", optimization_version)
print("Aer:", Aer)
print("NetworkX:", nx.__version__)
print("NumPy:", np.__version__)

G = nx.Graph()

features = ["f1", "f2", "f3", "f4"]

G.add_nodes_from(features)

G.add_weighted_edges_from([
    ("f1", "f2", 0.90),
    ("f1", "f3", 0.70),
    ("f2", "f3", 0.80),
    ("f2", "f4", 0.20),
    ("f3", "f4", 0.30),
])

print("\nFeature graph")
print("-" * 40)

for node in G.nodes:
    print(node)

print("\nEdges")
for u, v, data in G.edges(data=True):
    print(f"{u} -- {v}  weight={data['weight']}")

print("\nEnvironment and graph: READY")
