import itertools
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

data = load_wine()
X = StandardScaler().fit_transform(data.data)
y = data.target

mi = mutual_info_classif(X, y, random_state=42)
top = np.argsort(mi)[::-1][:6]

features = [data.feature_names[i] for i in top]
mi_top = mi[top]
mi_top = mi_top / mi_top.max()
corr = abs(np.corrcoef(X[:, top], rowvar=False))

def objective(combo):
    score = sum(mi_top[i] for i in combo)
    for i, j in itertools.combinations(combo, 2):
        score -= 0.3 * corr[i, j]
    return score

best_score = -float("inf")
best_combo = None

for combo in itertools.combinations(range(6), 3):
    score = objective(combo)
    if score > best_score:
        best_score = score
        best_combo = combo

selected = [features[i] for i in best_combo]

print("Exact target-aware optimum")
print("Candidates:", features)
print("Selected:", selected)
print("Objective:", best_score)
