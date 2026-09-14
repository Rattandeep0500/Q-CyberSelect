from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

data = load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

def evaluate(X_train, X_test, name):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print(name)
    print("Features:", X_train.shape[1])
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1:", f1_score(y_test, pred, average="weighted"))
    print()

selected = ["total_phenols", "flavanoids", "proanthocyanins"]
indices = [data.feature_names.index(f) for f in selected]

evaluate(
    X_train[:, indices],
    X_test[:, indices],
    "QAOA-selected"
)

evaluate(
    X_train,
    X_test,
    "All features"
)
