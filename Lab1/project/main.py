import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = np.array(X, dtype=np.float32) / 255
y = np.array(y, dtype=int)

matrices = [np.zeros((28, 28)) for _ in range(10)]
# 42/None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

for img, label in zip(X_train, y_train):
    img = img.reshape(28, 28)
    matrices[label] += img

predictions = []
for img in X_test:
    img = img.reshape(28, 28)
    scores = [np.sum(matrix * img) for matrix in matrices]
    predictions.append(np.argmax(scores))

accuracy = np.mean(predictions == y_test)
print(f'Точность: {accuracy * 100:.2f}%')


train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
accuracies = []
# 42/None
for train_size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=None)
    matrices = [np.zeros((28, 28)) for _ in range(10)]
    for img, label in zip(X_train, y_train):
        img = img.reshape(28, 28)
        matrices[label] += img
    predictions = []
    for img in X_test:
        img = img.reshape(28, 28)
        scores = [np.sum(matrix * img) for matrix in matrices]
        predictions.append(np.argmax(scores))
    accuracy = np.mean(predictions == y_test)
    accuracies.append(accuracy)

plt.plot(train_sizes, accuracies)
plt.xlabel('Размер обучающей выборки')
plt.ylabel('Точность')
plt.show()
