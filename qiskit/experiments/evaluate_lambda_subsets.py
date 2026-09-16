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

BASE = Path("data/unsw_nb15")

train = pd.read_csv(BASE / "UNSW_NB15_training-set.csv")
test = pd.read_csv(BASE / "UNSW_NB15_testing-set.csv")

with open(
    "results/cyber_lambda_sensitivity.json",
    encoding="utf-8"
) as f:
    lambda_results = json.load(f)

categorical_all = ["proto", "service", "state"]
target = "label"

def evaluate(lam, features, model_name, model):
    categorical = [c for c in features if c in categorical_all]
    numeric = [c for c in features if c not in categorical_all]

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

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("model", model)
    ])

    pipeline.fit(train[features], train[target])

    pred = pipeline.predict(test[features])
    prob = pipeline.predict_proba(test[features])[:, 1]

    return {
        "lambda": lam,
        "model": model_name,
        "feature_count": len(features),
        "features": features,
        "accuracy": accuracy_score(test[target], pred),
        "balanced_accuracy": balanced_accuracy_score(test[target], pred),
        "precision": precision_score(test[target], pred, zero_division=0),
        "recall": recall_score(test[target], pred, zero_division=0),
        "f1": f1_score(test[target], pred, zero_division=0),
        "roc_auc": roc_auc_score(test[target], prob),
        "pr_auc": average_precision_score(test[target], prob),
        "mcc": matthews_corrcoef(test[target], pred)
    }

records = []

for entry in lambda_results:
    lam = entry["lambda"]
    features = entry["selected"]

    records.append(
        evaluate(
            lam,
            features,
            "LogisticRegression",
            LogisticRegression(
                max_iter=500,
                solver="liblinear"
            )
        )
    )

    records.append(
        evaluate(
            lam,
            features,
            "RandomForest",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        )
    )

df = pd.DataFrame(records)

print(df[
    [
        "lambda",
        "model",
        "feature_count",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc"
    ]
].to_string(index=False))

df.to_csv(
    "results/cyber_lambda_model_comparison.csv",
    index=False
)

with open(
    "results/cyber_lambda_model_comparison.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(records, f, indent=2)
