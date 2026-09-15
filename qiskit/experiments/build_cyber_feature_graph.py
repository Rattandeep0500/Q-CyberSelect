import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

BASE = Path("data/unsw_nb15")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

train = pd.read_csv(BASE / "UNSW_NB15_training-set.csv")
test = pd.read_csv(BASE / "UNSW_NB15_testing-set.csv")

target = train["label"].astype(int)

drop_cols = ["id", "attack_cat", "label"]

X_train = train.drop(columns=drop_cols)
X_test = test.drop(columns=drop_cols)

categorical = ["proto", "service", "state"]
numeric = [c for c in X_train.columns if c not in categorical]

for col in categorical:
    train_codes = pd.Categorical(X_train[col]).codes
    test_codes = pd.Categorical(
        X_test[col],
        categories=pd.Categorical(X_train[col]).categories
    ).codes

    X_train[col] = train_codes
    X_test[col] = test_codes

discretizer = KBinsDiscretizer(
    n_bins=10,
    encode="ordinal",
    strategy="quantile",
    subsample=None
)

X_train_num = discretizer.fit_transform(X_train[numeric])
X_test_num = discretizer.transform(X_test[numeric])

X_train_graph = pd.DataFrame(
    X_train_num,
    columns=numeric,
    index=X_train.index
).astype(int)

X_test_graph = pd.DataFrame(
    X_test_num,
    columns=numeric,
    index=X_test.index
).astype(int)

for col in categorical:
    X_train_graph[col] = X_train[col].astype(int)
    X_test_graph[col] = X_test[col].astype(int)

X_train_graph = X_train_graph[X_train.columns]
X_test_graph = X_test_graph[X_test.columns]

sample_n = min(50000, len(X_train_graph))
sample_idx = np.random.default_rng(42).choice(
    len(X_train_graph),
    size=sample_n,
    replace=False
)

X_sample = X_train_graph.iloc[sample_idx].reset_index(drop=True)
y_sample = target.iloc[sample_idx].to_numpy()

def entropy(values):
    counts = np.bincount(values[values >= 0])
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())

def normalized_mi(a, b):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)

    mask = (a >= 0) & (b >= 0)
    a = a[mask]
    b = b[mask]

    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return 0.0

    mi = mutual_info_score(a, b)
    ha = entropy(a)
    hb = entropy(b)

    if ha == 0 or hb == 0:
        return 0.0

    return float(mi / np.sqrt(ha * hb))

relevance = {}

for col in X_sample.columns:
    relevance[col] = normalized_mi(
        X_sample[col].to_numpy(),
        y_sample
    )

relevance = pd.Series(relevance).sort_values(ascending=False)

top_k_candidates = 12
candidates = relevance.head(top_k_candidates).index.tolist()

weights = pd.DataFrame(
    np.zeros((len(candidates), len(candidates))),
    index=candidates,
    columns=candidates
)

for i, a in enumerate(candidates):
    for j in range(i + 1, len(candidates)):
        b = candidates[j]
        value = normalized_mi(
            X_sample[a].to_numpy(),
            X_sample[b].to_numpy()
        )
        weights.loc[a, b] = value
        weights.loc[b, a] = value

relevance.to_csv(OUT / "unsw_feature_relevance.csv", header=["relevance"])
weights.to_csv(OUT / "unsw_feature_graph.csv")

metadata = {
    "dataset": "UNSW-NB15",
    "train_shape": list(train.shape),
    "test_shape": list(test.shape),
    "predictor_count": len(X_train.columns),
    "predictors": X_train.columns.tolist(),
    "categorical_features": categorical,
    "numeric_features": numeric,
    "graph_method": "normalized mutual information",
    "numeric_discretization": "10 quantile bins",
    "graph_sample_size": int(sample_n),
    "candidate_count": int(top_k_candidates),
    "candidates": candidates
}

with open(OUT / "cyber_feature_graph_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Predictors:", len(X_train.columns))
print("Graph sample:", sample_n)
print()
print("Top cybersecurity features:")
for feature, score in relevance.head(12).items():
    print(f"{feature}: {score:.6f}")

print()
print("Candidate graph size:", len(candidates))
print("Saved:")
print("results/unsw_feature_relevance.csv")
print("results/unsw_feature_graph.csv")
print("results/cyber_feature_graph_metadata.json")
