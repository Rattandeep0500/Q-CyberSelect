import itertools
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

data = load_wine()
X = StandardScaler().fit_transform(data.data[:, :6])
features = data.feature_names[:6]

W = abs(np.corrcoef(X, rowvar=False))

best = float("inf")
best_partition = None

for combo in itertools.combinations(range(6), 3):
    A = set(combo)
    B = set(range(6)) - A

    cut = sum(
        W[i, j]
        for i in A
        for j in B
    )

    if cut < best:
        best = cut
        best_partition = combo

A = [features[i] for i in best_partition]
B = [features[i] for i in range(6) if i not in best_partition]

print("Exact Balanced MinCut")
print("Partition A:", A)
print("Partition B:", B)
print("Objective:", best)
