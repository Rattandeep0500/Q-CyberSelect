import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

df = pd.read_csv(
    "data/unsw_nb15/UNSW_NB15_training-set.csv"
)

target = "label"
drop = ["id", "attack_cat", target]

all_features = [
    c for c in df.columns
    if c not in drop
]

qaoa_features = [
    "sttl",
    "ct_state_ttl",
    "dload",
    "rate"
]

categorical_all = ["proto", "service", "state"]

y = df[target].astype(int)
X = df[all_features]

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

    for method, features in [
        ("QAOA-4", qaoa_features),
        ("All-42", all_features)
    ]:

        categorical = [
            c for c in features
            if c in categorical_all
        ]

        numeric = [
            c for c in features
            if c not in categorical_all
        ]

        transformers = []

        if numeric:
            transformers.append((
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numeric
            ))

        if categorical:
            transformers.append((
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore"
                    ))
                ]),
                categorical
            ))

        model = Pipeline([
            (
                "preprocessor",
                ColumnTransformer(transformers)
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced"
                )
            )
        ])

        model.fit(
            X.iloc[train_idx][features],
            y.iloc[train_idx]
        )

        pred = model.predict(
            X.iloc[val_idx][features]
        )

        prob = model.predict_proba(
            X.iloc[val_idx][features]
        )[:, 1]

        records.append({
            "fold": fold,
            "method": method,
            "feature_count": len(features),
            "f1": f1_score(
                y.iloc[val_idx],
                pred,
                zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y.iloc[val_idx],
                pred
            ),
            "roc_auc": roc_auc_score(
                y.iloc[val_idx],
                prob
            )
        })

result = pd.DataFrame(records)

print(result.to_string(index=False))

summary = (
    result
    .groupby("method")
    .agg(
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_roc_auc=("roc_auc", "mean")
    )
    .reset_index()
)

print()
print("SUMMARY")
print(summary.to_string(index=False))

result.to_csv(
    "results/qaoa_vs_all_cv.csv",
    index=False
)

summary.to_csv(
    "results/qaoa_vs_all_cv_summary.csv",
    index=False
)
