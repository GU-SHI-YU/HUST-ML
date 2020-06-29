#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression
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



param_grid = {"penalty":['l1', 'l2', 'elasticnet'], 
              "max_iter":[-1, 50, 100, 200], 
              "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

LR = LogisticRegression()

clf = GridSearchCV(estimator=LR, param_grid=param_grid, n_jobs=-1)

clf.fit(X_train, y_train)

s = str(clf.best_params_)

with open("LRParam", "w") as f:
    f.write(s)




