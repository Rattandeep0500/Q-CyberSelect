import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

base = "data/unsw_nb15/UNSW_NB15_training-set.csv"

df = pd.read_csv(base)

target = "label"
drop = ["id", "attack_cat", target]

features = [c for c in df.columns if c not in drop]
categorical = ["proto", "service", "state"]
numeric = [c for c in features if c not in categorical]

candidate_features = [
    "sttl",
    "dttl",
    "ct_state_ttl",
    "dload",
    "dinpkt",
    "dpkts",
    "state",
    "rate"
]

X = df[features]
y = df[target].astype(int)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

records = []

for fold_id, (train_idx, val_idx) in enumerate(
    cv.split(X, y),
    start=1
):

    Xtr = X.iloc[train_idx]
    Xva = X.iloc[val_idx]
    ytr = y.iloc[train_idx]
    yva = y.iloc[val_idx]

    relevance = {}

    for feature in candidate_features:

        if feature in categorical:
            a = pd.factorize(Xtr[feature])[0]
        else:
            a = pd.to_numeric(
                Xtr[feature],
                errors="coerce"
            ).fillna(
                Xtr[feature].median()
            )

        corr = np.corrcoef(
            a,
            ytr
        )[0, 1]

        if np.isnan(corr):
            corr = 0.0

        relevance[feature] = abs(corr)

    relevance = pd.Series(relevance)

    corr_matrix = pd.DataFrame(index=candidate_features, columns=candidate_features)

    for a in candidate_features:
        for b in candidate_features:

            if a == b:
                corr_matrix.loc[a, b] = 0.0
                continue

            if a in categorical:
                av = pd.factorize(Xtr[a])[0]
            else:
                av = pd.to_numeric(
                    Xtr[a],
                    errors="coerce"
                ).fillna(
                    Xtr[a].median()
                )

            if b in categorical:
                bv = pd.factorize(Xtr[b])[0]
            else:
                bv = pd.to_numeric(
                    Xtr[b],
                    errors="coerce"
                ).fillna(
                    Xtr[b].median()
                )

            c = np.corrcoef(av, bv)[0, 1]

            if np.isnan(c):
                c = 0.0

            corr_matrix.loc[a, b] = abs(c)

    qp = QuadraticProgram(
        name=f"cv_fold_{fold_id}"
    )

    for feature in candidate_features:
        qp.binary_var(feature)

    qp.linear_constraint(
        linear={
            feature: 1
            for feature in candidate_features
        },
        sense="==",
        rhs=4,
        name="select_k"
    )

    qp.maximize(
        linear={
            feature: float(relevance[feature])
            for feature in candidate_features
        },
        quadratic={
            (candidate_features[i], candidate_features[j]):
                -0.25 * float(
                    corr_matrix.iloc[i, j]
                )
            for i in range(len(candidate_features))
            for j in range(i + 1, len(candidate_features))
        }
    )

    qaoa = QAOA(
        sampler=StatevectorSampler(seed=42 + fold_id),
        optimizer=COBYLA(maxiter=10),
        reps=1
    )

    result = MinimumEigenOptimizer(qaoa).solve(qp)

    selected = [
        name.name
        for name, value in zip(
            qp.variables,
            result.x
        )
        if value > 0.5
    ]

    categorical_selected = [
        c for c in selected
        if c in categorical
    ]

    numeric_selected = [
        c for c in selected
        if c in numeric
    ]

    transformers = []

    if numeric_selected:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_selected
        ))

    if categorical_selected:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_selected
        ))

    model = Pipeline([
        (
            "preprocessor",
            ColumnTransformer(transformers)
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        )
    ])

    model.fit(
        Xtr[selected],
        ytr
    )

    pred = model.predict(
        Xva[selected]
    )

    score = f1_score(
        yva,
        pred,
        zero_division=0
    )

    records.append({
        "fold": fold_id,
        "selected_features": selected,
        "f1": float(score)
    })

print(
    json.dumps(
        records,
        indent=2
    )
)

with open(
    "results/qaoa_cv_stability.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        records,
        f,
        indent=2
    )

print()
print("Mean CV F1:",
      np.mean([r["f1"] for r in records]))

print("Selected sets:")
for r in records:
    print(
        r["fold"],
        r["selected_features"]
    )
