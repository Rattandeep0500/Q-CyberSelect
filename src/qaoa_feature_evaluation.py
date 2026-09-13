import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

data = load_wine()
X = data.data
y = data.target

sets = {
    "QAOA-3": [
        "flavanoids",
        "proline",
        "od280/od315_of_diluted_wines"
    ],
    "QAOA-7": [
        "flavanoids",
        "od280/od315_of_diluted_wines",
        "total_phenols"
    ],
    "All-13": data.feature_names
}

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

for name, selected in sets.items():
    indices = [data.feature_names.index(f) for f in selected]

    train = X_train[:, indices]
    test = X_test[:, indices]

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    model = LogisticRegression(max_iter=500)
    model.fit(train, y_train)

    pred = model.predict(test)

    print(name)
    print("Features:", len(selected))
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1:", f1_score(y_test, pred, average="weighted"))
    print()
