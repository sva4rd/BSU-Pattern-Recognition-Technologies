from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=None)

clf = svm.SVC(decision_function_shape='ovr', kernel='rbf', C=10)
gammas = np.linspace(0.00001, 0.01, 100)
scores = []
for gamma in gammas:
    clf.gamma = gamma
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

plt.plot(gammas, scores)
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
