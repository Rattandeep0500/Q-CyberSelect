import itertools
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

data = load_wine()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42,
    stratify=data.target
)

best_f1 = -1
best_acc = -1
best_features = None

for combo in itertools.combinations(range(X_train.shape[1]), 3):
    train = X_train[:, combo]
    test = X_test[:, combo]

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    model = LogisticRegression(max_iter=500)
    model.fit(train, y_train)

    pred = model.predict(test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")

    if f1 > best_f1:
        best_f1 = f1
        best_acc = acc
        best_features = combo

selected = [data.feature_names[i] for i in best_features]

print("Exact predictive best 3-feature subset")
print("Features:", selected)
print("Accuracy:", best_acc)
print("F1:", best_f1)
