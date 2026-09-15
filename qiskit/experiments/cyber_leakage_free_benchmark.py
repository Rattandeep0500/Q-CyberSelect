import json
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE = Path("data/unsw_nb15")

train = pd.read_csv(BASE / "UNSW_NB15_training-set.csv")
test = pd.read_csv(BASE / "UNSW_NB15_testing-set.csv")

with open("results/cyber_dks_qaoa.json", encoding="utf-8") as f:
    qaoa = json.load(f)

qaoa_features = qaoa["qaoa_selected"]
k = len(qaoa_features)

relevance = pd.read_csv(
    "results/unsw_feature_relevance.csv",
    index_col=0
).iloc[:, 0]

mi_features = relevance.sort_values(ascending=False).head(k).index.tolist()

target = "label"
drop = ["id", "attack_cat", target]

all_features = [c for c in train.columns if c not in drop]

categorical_all = ["proto", "service", "state"]
numeric_all = [c for c in all_features if c not in categorical_all]

def evaluate(name, selected_features):
    categorical = [c for c in selected_features if c in categorical_all]
    numeric = [c for c in selected_features if c in numeric_all]

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

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=300,
            solver="liblinear"
        ))
    ])

    model.fit(
        train[selected_features],
        train[target]
    )

    pred = model.predict(test[selected_features])
    prob = model.predict_proba(test[selected_features])[:, 1]

    return {
        "method": name,
        "feature_count": len(selected_features),
        "features": selected_features,
        "accuracy": accuracy_score(test[target], pred),
        "balanced_accuracy": balanced_accuracy_score(test[target], pred),
        "precision": precision_score(test[target], pred, zero_division=0),
        "recall": recall_score(test[target], pred, zero_division=0),
        "f1": f1_score(test[target], pred, zero_division=0),
        "roc_auc": roc_auc_score(test[target], prob)
    }

results = []

results.append(
    evaluate("QAOA-DkS", qaoa_features)
)

results.append(
    evaluate("MI-top-k", mi_features)
)

results.append(
    evaluate("All-features", all_features)
)

df = pd.DataFrame(results)

print(df[
    [
        "method",
        "feature_count",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]
].to_string(index=False))

df.to_json(
    "results/cyber_leakage_free_benchmark.json",
    orient="records",
    indent=2
)

df.to_csv(
    "results/cyber_leakage_free_benchmark.csv",
    index=False
)
