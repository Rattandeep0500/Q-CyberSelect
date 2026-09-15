import json
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef

base = Path("data/unsw_nb15")

train = pd.read_csv(base / "UNSW_NB15_training-set.csv")
test = pd.read_csv(base / "UNSW_NB15_testing-set.csv")

with open("results/cyber_target_aware_qaoa.json", encoding="utf-8") as f:
    qaoa = json.load(f)

features = qaoa["selected"]

categorical_all = ["proto", "service", "state"]

def evaluate(model_name, model):
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

    pipeline.fit(train[features], train["label"])

    pred = pipeline.predict(test[features])
    prob = pipeline.predict_proba(test[features])[:, 1]

    return {
        "method": "Target-aware-QAOA",
        "model": model_name,
        "features": len(features),
        "selected": features,
        "accuracy": accuracy_score(test["label"], pred),
        "balanced_accuracy": balanced_accuracy_score(test["label"], pred),
        "precision": precision_score(test["label"], pred, zero_division=0),
        "recall": recall_score(test["label"], pred, zero_division=0),
        "f1": f1_score(test["label"], pred, zero_division=0),
        "roc_auc": roc_auc_score(test["label"], prob),
        "pr_auc": average_precision_score(test["label"], prob),
        "mcc": matthews_corrcoef(test["label"], pred)
    }

results = [
    evaluate(
        "LogisticRegression",
        LogisticRegression(
            max_iter=500,
            solver="liblinear"
        )
    ),
    evaluate(
        "RandomForest",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    )
]

for r in results:
    print(r)

with open(
    "results/cyber_target_aware_model_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=2)

pd.DataFrame(results).to_csv(
    "results/cyber_target_aware_model_results.csv",
    index=False
)
