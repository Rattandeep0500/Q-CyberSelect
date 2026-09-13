import itertools
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

data = load_wine()
X = StandardScaler().fit_transform(data.data)
y = data.target

mi = mutual_info_classif(X, y, random_state=42)
top = np.argsort(mi)[-6:]
features = [data.feature_names[i] for i in top]

corr = abs(np.corrcoef(X[:, top], rowvar=False))
mi_top = mi[top]
mi_top = mi_top / mi_top.max()

def edge_weight(i, j):
    relevance = (mi_top[i] + mi_top[j]) / 2
    redundancy = corr[i, j]
    return 0.7 * relevance + 0.3 * redundancy

best = None

for combo in itertools.combinations(range(6), 3):
    score = 0

    for i, j in itertools.combinations(combo, 2):
        score += edge_weight(i, j)

    if best is None or score > best[0]:
        best = (score, combo)

print("Exact best 3-feature solution:")
print()

for i in best[1]:
    print(features[i])

print()
print("Exact objective:", best[0])
