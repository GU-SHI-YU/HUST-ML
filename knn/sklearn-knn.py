import time
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=600, test_size=100)

accuracies = []
x = []

for p in [2,5,10,20]:
    accuracies.clear()
    x.clear()
    for k in range(1, 500):
        model = KNeighborsClassifier(n_neighbors = k, p = p)

        model.fit(X_train, y_train)
        # evaluate the model and print the accuracies list
        score = model.score(X_test, y_test)
        score = round(score * 100, 2)
        accuracies.append(score)
        k = float(k)
        k = round(k, 2)
        x.append(k)
    lab = 'p=' + str(p)
    plt.plot(x, accuracies, label = lab)
legend(loc = 'upper right')
plt.savefig('res/k.png')
plt.show()