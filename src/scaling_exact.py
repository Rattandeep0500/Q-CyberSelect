import itertools
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

data = load_wine()
X = StandardScaler().fit_transform(data.data)
y = data.target

mi = mutual_info_classif(X, y, random_state=42)
ranking = np.argsort(mi)[::-1]

for n in [6, 7, 8, 9, 10]:
    indices = ranking[:n]
    features = [data.feature_names[i] for i in indices]

    corr = abs(np.corrcoef(X[:, indices], rowvar=False))
    mi_top = mi[indices] / mi[indices].max()

    def weight(i, j):
        return 0.7 * ((mi_top[i] + mi_top[j]) / 2) + 0.3 * corr[i, j]

    best_score = -1
    best_combo = None

    for combo in itertools.combinations(range(n), 3):
        score = sum(
            weight(i, j)
            for i, j in itertools.combinations(combo, 2)
        )

        if score > best_score:
            best_score = score
            best_combo = combo

    print(f"n={n}  optimum={best_score:.6f}  selected={[features[i] for i in best_combo]}")
