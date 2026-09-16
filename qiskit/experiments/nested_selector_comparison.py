import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

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

categorical = ["proto", "service", "state"]

X = df[features].copy()
y = df[target].astype(int)

for col in categorical:
    X[col] = pd.factorize(X[col])[0].astype(float)

for col in features:
    if col not in categorical:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(X),
    columns=features
)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

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

    mi = mutual_info_classif(
        Xtr_scaled,
        ytr,
        random_state=42
    )

    fvals, _ = f_classif(
        Xtr_scaled,
        ytr
    )

    l1 = LogisticRegression(
        l1_ratio=1.0,
        solver="saga",
        max_iter=1000,
        random_state=42
    )

    l1.fit(
        Xtr_scaled,
        ytr
    )

    l1_importance = np.abs(
        l1.coef_
    ).mean(axis=0)

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf.fit(
        Xtr_scaled,
        ytr
    )

    rf_importance = rf.feature_importances_

    selectors = {
        "MI-4": np.argsort(mi)[-4:],
        "ANOVA-4": np.argsort(fvals)[-4:],
        "L1-4": np.argsort(l1_importance)[-4:],
        "RF-4": np.argsort(rf_importance)[-4:]
    }

    qaoa_candidates = [
        "sttl",
        "ct_state_ttl",
        "dload",
        "ct_dst_sport_ltm",
        "dmean",
        "rate",
        "swin",
        "dwin"
    ]

    qaoa_idx = [
        features.index(f)
        for f in qaoa_candidates
    ]

    qaoa_corr = abs(
        np.corrcoef(
            Xtr_scaled[:, qaoa_idx],
            rowvar=False
        )
    )

    qaoa_relevance = pd.Series(
        mi[qaoa_idx],
        index=qaoa_candidates
    )

    qp = QuadraticProgram(
        name=f"qaoa_nested_fold_{fold}"
    )

    for feature in qaoa_candidates:
        qp.binary_var(feature)

    qp.linear_constraint(
        linear={
            f: 1
            for f in qaoa_candidates
        },
        sense="==",
        rhs=4,
        name="select_k"
    )

    qp.maximize(
        linear={
            f: float(qaoa_relevance[f])
            for f in qaoa_candidates
        },
        quadratic={
            (
                qaoa_candidates[i],
                qaoa_candidates[j]
            ): -0.25 * float(
                qaoa_corr[i, j]
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

    qaoa_result = MinimumEigenOptimizer(
        qaoa
    ).solve(qp)

    qaoa_names = [
        name.name
        for name, value in zip(
            qp.variables,
            qaoa_result.x
        )
        if value > 0.5
    ]

    selectors["QAOA-4"] = np.array([
        features.index(f)
        for f in qaoa_names
    ])

    for method, indices in selectors.items():

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

        model.fit(
            Xtr_scaled[:, indices],
            ytr
        )

        pred = model.predict(
            Xva_scaled[:, indices]
        )

        prob = model.predict_proba(
            Xva_scaled[:, indices]
        )[:, 1]

        records.append({
            "fold": fold,
            "method": method,
            "features": 4,
            "selected": [
                features[i]
                for i in indices
            ],
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

result = pd.DataFrame(records)

summary = (
    result
    .groupby("method")
    .agg(
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        mean_balanced_accuracy=(
            "balanced_accuracy",
            "mean"
        ),
        mean_roc_auc=("roc_auc", "mean")
    )
    .reset_index()
)

print(result.to_string(index=False))
print()
print("SUMMARY")
print(summary.to_string(index=False))

result.to_csv(
    "results/nested_selector_comparison.csv",
    index=False
)

summary.to_csv(
    "results/nested_selector_summary.csv",
    index=False
)

with open(
    "results/nested_selector_comparison.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        records,
        f,
        indent=2
    )
