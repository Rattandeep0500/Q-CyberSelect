import numpy as np
import networkx as nx
from sklearn.datasets import load_wine
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

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

print("Feature relevance:")
for feature, score in zip(features, mi):
    print(f"{feature}: {score:.4f}")

print("\nTop graph edges:")
edges = sorted(G.edges(data="weight"), key=lambda x: x[2], reverse=True)

for u, v, weight in edges[:10]:
    print(f"{u} -- {v}: {weight:.4f}")
