from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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

selector = SelectKBest(mutual_info_classif, k=3)

X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

scaler = StandardScaler()
X_train_selected = scaler.fit_transform(X_train_selected)
X_test_selected = scaler.transform(X_test_selected)

model = LogisticRegression(max_iter=500)
model.fit(X_train_selected, y_train)

pred = model.predict(X_test_selected)

selected = [
    data.feature_names[i]
    for i, value in enumerate(selector.get_support())
    if value
]

print("Classical MI-selected")
print("Features:", selected)
print("Accuracy:", accuracy_score(y_test, pred))
print("F1:", f1_score(y_test, pred, average="weighted"))
