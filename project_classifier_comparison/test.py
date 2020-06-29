#!/usr/bin/env python
# coding: utf-8



import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.metrics import classification_report



X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=60000, test_size=10000)



label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



begin = time.time()
k = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=-1)
k.fit(X_train, y_train)
k_y_pred = k.predict(X_test)
print(classification_report(y_test, k_y_pred, target_names=label_names))
end = time.time()
print("time:", end-begin)



begin = time.time()
LR = LogisticRegression(penalty='l2', solver='saga', max_iter=50, n_jobs=-1)
LR.fit(X_train, y_train)
LR_y_pred = LR.predict(X_test)
print(classification_report(y_test, LR_y_pred, target_names=label_names))
end = time.time()
print("time:", end-begin)


# In[12]:


begin = time.time()
GNB = MultinomialNB(alpha=0.8)
GNB.fit(X_train, y_train)
GNB_y_pred = GNB.predict(X_test)
print(classification_report(y_test, GNB_y_pred, target_names=label_names))
end = time.time()
print("time:", end-begin)


# In[13]:


begin = time.time()
svc = SVC(kernel='rbf', max_iter=200)
svc.fit(X_train, y_train)
svc_y_pred = svc.predict(X_test)
print(classification_report(y_test, svc_y_pred, target_names=label_names))
end = time.time()
print("time:", end-begin)


# In[4]:


for i in range(60000):
    for j in range(784):
        if(X_train[i][j] < 50):
            X_train[i][j] = 1
        else:
            X_train[i][j] = 0


# In[5]:


for i in range(10000):
    for j in range(784):
        if(X_test[i][j] < 50):
            X_test[i][j] = 1
        else:
            X_test[i][j] = 0


# In[42]:


begin = time.time()
k = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=-1)
k.fit(X_train, y_train)
k_y_pred = k.predict(X_test)
s = str(classification_report(y_test, k_y_pred, target_names=label_names))
end = time.time()
with open("../res/kRes2", "w") as f:
    f.write(s)
    f.write(str(end - begin))


# In[43]:


begin = time.time()
LR = LogisticRegression(penalty='l2', solver='saga', max_iter=50, n_jobs=-1)
LR.fit(X_train, y_train)
LR_y_pred = LR.predict(X_test)
s = str(classification_report(y_test, LR_y_pred, target_names=label_names))
end = time.time()
with open("../res/LRRes2", "w") as f:
    f.write(s)
    f.write(str(end - begin))


# In[44]:


begin = time.time()
svc = SVC(kernel='rbf', max_iter=200)
svc.fit(X_train, y_train)
svc_y_pred = svc.predict(X_test)
s = str(classification_report(y_test, svc_y_pred, target_names=label_names))
end = time.time()
with open("../res/svmRes2", "w") as f:
    f.write(s)
    f.write(str(end - begin))


# In[48]:


begin = time.time()
GNB = MultinomialNB(alpha=0.8)
GNB.fit(X_train, y_train)
GNB_y_pred = GNB.predict(X_test)
s = str(classification_report(y_test, GNB_y_pred, target_names=label_names))
end = time.time()
with open("../res/NBRes2", "w") as f:
    f.write(s)
    f.write(str(end - begin))


# In[46]:


X_test[0]


# In[ ]:




