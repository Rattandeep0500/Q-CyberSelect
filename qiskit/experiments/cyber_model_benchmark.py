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

with open("results/cyber_dks_qaoa.json", encoding="utf-8") as f:
    qaoa = json.load(f)

qaoa_features = qaoa["qaoa_selected"]

relevance = pd.read_csv(
    "results/unsw_feature_relevance.csv",
    index_col=0
).iloc[:, 0]

k = len(qaoa_features)

mi_features = relevance.sort_values(ascending=False).head(k).index.tolist()

drop = ["id", "attack_cat", "label"]

all_features = [c for c in train.columns if c not in drop]

categorical_all = ["proto", "service", "state"]
numeric_all = [c for c in all_features if c not in categorical_all]

def build_pipeline(features, model):
    categorical = [c for c in features if c in categorical_all]
    numeric = [c for c in features if c in numeric_all]

    transformers = []

    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numeric
            )
        )

    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical
            )
        )

    preprocessor = ColumnTransformer(transformers)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

def evaluate(method, features, model_name, model):
    pipeline = build_pipeline(features, model)

    pipeline.fit(
        train[features],
        train["label"]
    )

    pred = pipeline.predict(test[features])
    prob = pipeline.predict_proba(test[features])[:, 1]

    return {
        "method": method,
        "model": model_name,
        "features": len(features),
        "accuracy": accuracy_score(test["label"], pred),
        "balanced_accuracy": balanced_accuracy_score(test["label"], pred),
        "precision": precision_score(test["label"], pred, zero_division=0),
        "recall": recall_score(test["label"], pred, zero_division=0),
        "f1": f1_score(test["label"], pred, zero_division=0),
        "roc_auc": roc_auc_score(test["label"], prob),
        "pr_auc": average_precision_score(test["label"], prob),
        "mcc": matthews_corrcoef(test["label"], pred)
    }

results = []

methods = [
    ("QAOA-DkS", qaoa_features),
    ("MI-top-k", mi_features),
    ("All-features", all_features)
]

models = [
    ("LogisticRegression", LogisticRegression(max_iter=500, solver="liblinear")),
    ("RandomForest", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
]

for method, features in methods:
    for model_name, model in models:
        results.append(
            evaluate(
                method,
                features,
                model_name,
                model
            )
        )

df = pd.DataFrame(results)

print(df.to_string(index=False))

df.to_csv(
    "results/cyber_model_benchmark.csv",
    index=False
)

with open(
    "results/cyber_model_benchmark.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        indent=2
    )
