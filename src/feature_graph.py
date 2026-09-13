import numpy as np
import networkx as nx
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

data = load_wine()

X = StandardScaler().fit_transform(data.data)
feature_names = data.feature_names

corr = np.corrcoef(X, rowvar=False)
G = nx.Graph()
G.add_nodes_from(feature_names)

for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):
        weight = abs(corr[i, j])
        if weight > 0.5:
            G.add_edge(feature_names[i], feature_names[j], weight=float(weight))

print("Features:", len(feature_names))
print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())

print("\nFeature graph:")
for u, v, data in G.edges(data=True):
    print(f"{u} -- {v}: {data['weight']:.4f}")
