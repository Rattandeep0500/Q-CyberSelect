import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

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

for col in features:
    if col in categorical:
        X[col] = pd.factorize(X[col])[0]
    else:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
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

    imputer = SimpleImputer(
        strategy="median"
    )

    Xtr_imp = imputer.fit_transform(Xtr)
    Xva_imp = imputer.transform(Xva)

    scaler = StandardScaler()

    Xtr_scaled = scaler.fit_transform(Xtr_imp)
    Xva_scaled = scaler.transform(Xva_imp)

    selectors = {}

    mi = mutual_info_classif(
        Xtr_scaled,
        ytr,
        random_state=42
    )

    selectors["MI"] = np.argsort(mi)[-4:]

    f_values, _ = f_classif(
        Xtr_scaled,
        ytr
    )

    selectors["ANOVA"] = np.argsort(f_values)[-4:]

    l1 = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=500
    )

    l1.fit(
        Xtr_scaled,
        ytr
    )

    importance = np.abs(
        l1.coef_
    ).mean(axis=0)

    selectors["L1"] = np.argsort(
        importance
    )[-4:]

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf.fit(
        Xtr_imp,
        ytr
    )

    selectors["RandomForest"] = np.argsort(
        rf.feature_importances_
    )[-4:]

    for method, indices in selectors.items():

        model = LogisticRegression(
            max_iter=500,
            solver="liblinear"
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

        selected_names = [
            features[i]
            for i in indices
        ]

        records.append({
            "fold": fold,
            "method": method,
            "feature_count": 4,
            "features": selected_names,
            "f1": f1_score(
                yva,
                pred,
                zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(
                yva,
                pred
            ),
            "roc_auc": roc_auc_score(
                yva,
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
        mean_balanced_accuracy=(
            "balanced_accuracy",
            "mean"
        ),
        mean_roc_auc=("roc_auc", "mean")
    )
    .reset_index()
)

print()
print("SUMMARY")
print(summary.to_string(index=False))

result.to_csv(
    "results/classical_feature_selection_cv.csv",
    index=False
)

summary.to_csv(
    "results/classical_feature_selection_summary.csv",
    index=False
)
