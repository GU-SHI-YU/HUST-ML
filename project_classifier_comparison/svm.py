#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state



X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=6000, test_size=1000)
X_train.shape



param_grid = {"kernel":['linear', 'poly', 'rbf', 'sigmoid'], "max_iter":[-1, 50, 100, 200]}

svc = SVC()

clf = GridSearchCV(estimator=svc, param_grid=param_grid, n_jobs=-1)

clf.fit(X_train, y_train)

s = str(clf.best_params_)

with open("svcParam", "w") as f:
    f.write(s)





