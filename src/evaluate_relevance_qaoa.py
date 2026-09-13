from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

data = load_wine()

X = data.data
y = data.target

features = [
    "od280/od315_of_diluted_wines",
    "proline",
    "flavanoids"
]

indices = [data.feature_names.index(f) for f in features]

X_train, X_test, y_train, y_test = train_test_split(
    X[:, indices],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("QAOA relevance-aware")
print("Features:", features)
print("Accuracy:", accuracy_score(y_test, pred))
print("F1:", f1_score(y_test, pred, average="weighted"))
