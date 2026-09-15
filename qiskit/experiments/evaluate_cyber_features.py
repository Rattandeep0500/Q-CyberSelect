import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

base = Path("data/unsw_nb15")

train = pd.read_csv(base / "UNSW_NB15_training-set.csv")
test = pd.read_csv(base / "UNSW_NB15_testing-set.csv")

with open("results/cyber_dks_qaoa.json", encoding="utf-8") as f:
    qaoa_result = json.load(f)

qaoa_features = qaoa_result["qaoa_selected"]

drop_cols = ["id", "attack_cat", "label"]

X_train = train.drop(columns=drop_cols).copy()
X_test = test.drop(columns=drop_cols).copy()
y_train = train["label"].astype(int)
y_test = test["label"].astype(int)

categorical = ["proto", "service", "state"]
numeric = [c for c in X_train.columns if c not in categorical]

for col in categorical:
    categories = pd.Index(X_train[col].dropna().unique())
    train_map = {value: i for i, value in enumerate(categories)}

    X_train[col] = X_train[col].map(train_map).fillna(-1).astype(int)
    X_test[col] = X_test[col].map(train_map).fillna(-1).astype(int)

scaler = StandardScaler()

X_train[numeric] = scaler.fit_transform(X_train[numeric])
X_test[numeric] = scaler.transform(X_test[numeric])

def evaluate(name, features):
    train_part = X_train[features]
    test_part = X_test[features]

    model = LogisticRegression(
        max_iter=500,
        n_jobs=-1
    )

    model.fit(train_part, y_train)

    pred = model.predict(test_part)
    prob = model.predict_proba(test_part)[:, 1]

    return {
        "method": name,
        "features": len(features),
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob)
    }

candidate_count = min(4, len(qaoa_features))

mi_scores = {}

for col in X_train.columns:
    mi_scores[col] = abs(np.corrcoef(X_train[col], y_train)[0, 1])

mi_features = sorted(
    mi_scores,
    key=mi_scores.get,
    reverse=True
)[:candidate_count]

results = []

results.append(
    evaluate(
        "QAOA-DkS",
        qaoa_features
    )
)

results.append(
    evaluate(
        "Classical-correlation",
        mi_features
    )
)

results.append(
    evaluate(
        "All-features",
        list(X_train.columns)
    )
)

df = pd.DataFrame(results)

print()
print(df.to_string(index=False))

df.to_csv(
    "results/cyber_classifier_comparison.csv",
    index=False
)
