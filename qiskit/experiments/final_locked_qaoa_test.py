import json
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)

base = Path("data/unsw_nb15")

train = pd.read_csv(base / "UNSW_NB15_training-set.csv")
test = pd.read_csv(base / "UNSW_NB15_testing-set.csv")

with open(
    "results/cyber_lambda_cv_selection.json",
    encoding="utf-8"
) as f:
    cv_results = json.load(f)

best_lambda = max(
    cv_results,
    key=lambda x: x["mean_cv_f1"]
)["lambda"]

with open(
    "results/cyber_lambda_sensitivity.json",
    encoding="utf-8"
) as f:
    lambda_results = json.load(f)

selected_entry = next(
    x for x in lambda_results
    if x["lambda"] == best_lambda
)

features = selected_entry["selected"]

categorical_all = ["proto", "service", "state"]

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
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    ))

results = []

models = [
    (
        "LogisticRegression",
        LogisticRegression(
            max_iter=500,
            solver="liblinear"
        )
    ),
    (
        "RandomForest",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    )
]

for model_name, model in models:

    pipeline = Pipeline([
        (
            "preprocessor",
            ColumnTransformer(transformers)
        ),
        ("model", model)
    ])

    pipeline.fit(
        train[features],
        train["label"]
    )

    pred = pipeline.predict(
        test[features]
    )

    prob = pipeline.predict_proba(
        test[features]
    )[:, 1]

    results.append({
        "lambda": best_lambda,
        "model": model_name,
        "feature_count": len(features),
        "features": features,
        "accuracy": accuracy_score(
            test["label"], pred
        ),
        "balanced_accuracy": balanced_accuracy_score(
            test["label"], pred
        ),
        "precision": precision_score(
            test["label"], pred, zero_division=0
        ),
        "recall": recall_score(
            test["label"], pred, zero_division=0
        ),
        "f1": f1_score(
            test["label"], pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            test["label"], prob
        ),
        "pr_auc": average_precision_score(
            test["label"], prob
        ),
        "mcc": matthews_corrcoef(
            test["label"], pred
        )
    })

output = pd.DataFrame(results)

print("LOCKED LAMBDA:", best_lambda)
print("FEATURES:", features)
print()
print(output.to_string(index=False))

output.to_csv(
    "results/final_locked_qaoa_test.csv",
    index=False
)

with open(
    "results/final_locked_qaoa_test.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        indent=2
    )

with open(
    "results/locked_model_selection.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "selection_rule": "highest mean 3-fold training CV F1",
            "selected_lambda": best_lambda,
            "selected_features": features
        },
        f,
        indent=2
    )
