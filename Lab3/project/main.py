import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_random_forest(X_train, y_train, X_test, y_test, params=None):
    model = RandomForestClassifier(**params) if params else RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


param_dist = {
    'n_estimators': 200,
    'max_features': 'sqrt',
    'max_depth': 14,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'bootstrap': False
}

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

results = []
results_base = []

for train_size in range(10, 90, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size / 100, random_state=42)

    accuracy = train_random_forest(X_train, y_train, X_test, y_test, params=param_dist)
    results.append((train_size, accuracy))

    accuracy_base = train_random_forest(X_train, y_train, X_test, y_test)
    results_base.append((train_size, accuracy_base))

train_sizes, accuracies = zip(*results)
_, accuracies_base = zip(*results_base)

plt.plot(train_sizes, accuracies, '-o', label='Random Forest (Selected Parameters)')
plt.plot(train_sizes, accuracies_base, '-o', label='Random Forest (Default Parameters)')
plt.xlabel('Train Size (%)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Train Size')
plt.legend()
plt.show()
