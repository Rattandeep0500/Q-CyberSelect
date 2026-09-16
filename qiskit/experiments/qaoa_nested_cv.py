import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

df = pd.read_csv(
    "data/unsw_nb15/UNSW_NB15_training-set.csv"
)

target = "label"
drop = ["id", "attack_cat", target]

features = [
    c for c in df.columns
    if c not in drop
]

X = df[features].copy()
y = df[target].astype(int)

categorical = ["proto", "service", "state"]
numeric = [c for c in features if c not in categorical]

for col in categorical:
    X[col] = pd.factorize(X[col])[0].astype(float)

for col in numeric:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

num_imputer = SimpleImputer(strategy="median")
X[numeric] = num_imputer.fit_transform(X[numeric])

cat_imputer = SimpleImputer(strategy="most_frequent")
X[categorical] = cat_imputer.fit_transform(X[categorical])

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

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

records = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y),
    start=1
):

    Xtr = X.iloc[train_idx]
    Xva = X.iloc[val_idx]
    ytr = y.iloc[train_idx]
    yva = y.iloc[val_idx]

    scaler = StandardScaler()

    Xtr_scaled = scaler.fit_transform(Xtr)
    Xva_scaled = scaler.transform(Xva)

    relevance = {}

    for i, feature in enumerate(features):
        score = np.corrcoef(
            Xtr_scaled[:, i],
            ytr
        )[0, 1]

        if np.isnan(score):
            score = 0.0

        relevance[feature] = abs(score)

    relevance = pd.Series(relevance)

    candidates = (
        relevance
        .sort_values(ascending=False)
        .head(8)
        .index
        .tolist()
    )

    candidate_idx = [
        features.index(f)
        for f in candidates
    ]

    corr = abs(
        np.corrcoef(
            Xtr_scaled[:, candidate_idx],
            rowvar=False
        )
    )

    qp = QuadraticProgram(
        name=f"nested_qaoa_fold_{fold}"
    )

    for feature in candidates:
        qp.binary_var(feature)

    qp.linear_constraint(
        linear={
            feature: 1
            for feature in candidates
        },
        sense="==",
        rhs=4,
        name="select_k"
    )

    qp.maximize(
        linear={
            feature: float(relevance[feature])
            for feature in candidates
        },
        quadratic={
            (
                candidates[i],
                candidates[j]
            ): -0.25 * float(
                corr[i, j]
            )
            for i in range(8)
            for j in range(i + 1, 8)
        }
    )

    qaoa = QAOA(
        sampler=StatevectorSampler(
            seed=100 + fold
        ),
        optimizer=COBYLA(
            maxiter=8
        ),
        reps=1
    )

    result = MinimumEigenOptimizer(
        qaoa
    ).solve(qp)

    selected = [
        name.name
        for name, value in zip(
            qp.variables,
            result.x
        )
        if value > 0.5
    ]

    selected_idx = [
        features.index(f)
        for f in selected
    ]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(
        Xtr_scaled[:, selected_idx],
        ytr
    )

    pred = model.predict(
        Xva_scaled[:, selected_idx]
    )

    prob = model.predict_proba(
        Xva_scaled[:, selected_idx]
    )[:, 1]

    records.append({
        "fold": fold,
        "candidates": candidates,
        "selected": selected,
        "f1": float(
            f1_score(
                yva,
                pred,
                zero_division=0
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                yva,
                pred
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                yva,
                prob
            )
        )
    })

summary = {
    "mean_f1": float(
        np.mean([r["f1"] for r in records])
    ),
    "std_f1": float(
        np.std(
            [r["f1"] for r in records],
            ddof=1
        )
    ),
    "mean_balanced_accuracy": float(
        np.mean(
            [r["balanced_accuracy"] for r in records]
        )
    ),
    "mean_roc_auc": float(
        np.mean(
            [r["roc_auc"] for r in records]
        )
    )
}

print(json.dumps(records, indent=2))
print()
print(json.dumps(summary, indent=2))

with open(
    "results/qaoa_nested_cv.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "folds": records,
            "summary": summary
        },
        f,
        indent=2
    )
