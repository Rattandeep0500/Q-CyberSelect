import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

BASE = Path("data/unsw_nb15")

df = pd.read_csv(BASE / "UNSW_NB15_training-set.csv")

target = "label"
drop_cols = ["id", "attack_cat", target]

categorical = ["proto", "service", "state"]
features = [c for c in df.columns if c not in drop_cols]
numeric = [c for c in features if c not in categorical]

relevance = pd.read_csv(
    "results/unsw_feature_relevance.csv",
    index_col=0
).iloc[:, 0]

graph = pd.read_csv(
    "results/unsw_feature_graph.csv",
    index_col=0
)

candidates = relevance.sort_values(ascending=False).head(8).index.tolist()
k = 4

X = df[features]
y = df[target].astype(int)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]

records = []

for lam in lambdas:

    selected = []

    scores = []

    for train_idx, val_idx in cv.split(X, y):

        fold = X.iloc[train_idx]
        fold_y = y.iloc[train_idx]

        fold_relevance = {}

        for feature in candidates:
            a = fold[feature]

            if feature in categorical:
                a = pd.factorize(a)[0]

            corr = np.corrcoef(
                pd.to_numeric(a, errors="coerce"),
                fold_y
            )[0, 1]

            if np.isnan(corr):
                corr = 0.0

            fold_relevance[feature] = abs(corr)

        fold_relevance = pd.Series(fold_relevance)

        ranked = fold_relevance.sort_values(ascending=False)

        fold_candidates = ranked.head(8).index.tolist()

        best_score = -float("inf")
        best_combo = None

        for combo in __import__("itertools").combinations(
            fold_candidates,
            k
        ):

            score = sum(
                fold_relevance[f]
                for f in combo
            )

            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    a = combo[i]
                    b = combo[j]

                    av = pd.factorize(fold[a])[0]
                    bv = pd.factorize(fold[b])[0]

                    c = np.corrcoef(av, bv)[0, 1]

                    if np.isnan(c):
                        c = 0.0

                    score -= lam * abs(c)

            if score > best_score:
                best_score = score
                best_combo = combo

        selected.append(best_combo)

        fold_train = X.iloc[train_idx][list(best_combo)]
        fold_val = X.iloc[val_idx][list(best_combo)]

        fold_categorical = [
            c for c in best_combo
            if c in categorical
        ]

        fold_numeric = [
            c for c in best_combo
            if c in numeric
        ]

        transformers = []

        if fold_numeric:
            transformers.append((
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                fold_numeric
            ))

        if fold_categorical:
            transformers.append((
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                fold_categorical
            ))

        model = Pipeline([
            ("prep", ColumnTransformer(transformers)),
            ("clf", LogisticRegression(
                max_iter=500,
                solver="liblinear"
            ))
        ])

        model.fit(
            fold_train,
            y.iloc[train_idx]
        )

        pred = model.predict(fold_val)

        scores.append(
            f1_score(
                y.iloc[val_idx],
                pred,
                zero_division=0
            )
        )

    records.append({
        "lambda": lam,
        "mean_cv_f1": float(np.mean(scores)),
        "std_cv_f1": float(np.std(scores)),
        "fold_f1": [float(x) for x in scores],
        "selected_per_fold": [
            list(x) for x in selected
        ]
    })

result = pd.DataFrame(records)

print(result[
    [
        "lambda",
        "mean_cv_f1",
        "std_cv_f1",
        "fold_f1"
    ]
].to_string(index=False))

result.to_csv(
    "results/cyber_lambda_cv_selection.csv",
    index=False
)

with open(
    "results/cyber_lambda_cv_selection.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(records, f, indent=2)
