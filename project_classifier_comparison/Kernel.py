#!/usr/bin/env python
# coding: utf-8


from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def K(X,Y=None,metric='poly',coef0=1,gamma=None,degree=3):
    if metric == 'poly':
        k = pairwise_kernels(X,Y=Y,metric=metric,coef0=coef0,gamma=gamma,degree=degree)
    elif metric == 'linear':
        k = pairwise_kernels(X,Y=Y,metric=metric)
    elif metric == 'sigmoid':
        k = pairwise_kernels(X,Y=Y,metric=metric,coef0=coef0,gamma=gamma)
    elif metric == 'rbf':
        k = pairwise_kernels(X,Y=Y,metric=metric,gamma=gamma)
    return k

class KernelKNN():  
    def __init__(self,metric='poly',coef0=1,gamma=None,degree=3,
                 n_neighbors=5,weights='distance'): #uniform
        self.m = metric
        self.coef = coef0
        self.gamma = gamma
        self.degree = degree
        self.n = n_neighbors
        self.weights = weights
    
    def fit(self,X,Y=None):
    # Polynomial kernel: K(a,b) = (coef0+gamma<a,b>)**degree
    # Sigmoid kernel: K(a,b) = tanh(coef+gamma<a,b>)
    # Linear kernel: K(a,b) = <a,b>
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.x = X
        self.y = Y
        classes = np.unique(Y)
        label = np.zeros((len(Y),len(classes)))
        for i in range(len(Y)):
            for ii in range(len(classes)):
                if Y[i] == classes[ii]:
                    label[i][ii] = 1
        self.classes = classes
        self.label = label

        Kaa = []
        for i in range(len(X)):
            Kaa.append(K(X[i,:].reshape(1,-1),metric=self.m,
                         coef0=self.coef,gamma=self.gamma,degree=self.degree))
        self.Kaa = np.asarray(Kaa).ravel().reshape(len(Kaa),1)
        return self
    
    def predict_proba(self,X):
        X = np.asarray(X)
        Kbb = []
        Kab = K(self.x,X,metric=self.m,
                coef0=self.coef,gamma=self.gamma,degree=self.degree)

        for i in range(len(X)):
            Kbb.append(K(X[i,:].reshape(1,-1),metric=self.m,
                         coef0=self.coef,gamma=self.gamma,degree=self.degree))
        self.Kbb = np.asarray(Kbb).ravel()
        d = np.array(self.Kaa-2*Kab+self.Kbb, dtype=float) #shape: (n_train,n_test)

        n_d = [] #neighbors' distance matrix
        index = []
        for i in range(d.shape[1]):
            index.append(np.argsort(d[:,i])[:self.n])
            n_d.append(d[index[i],i])
        n_d = np.asmatrix(n_d) + 1e-20
        
        w = np.asarray((1/n_d) / np.sum(1/n_d,axis=1)) 
            #weights matrix, shape: (n_test,n_neighbors)
        w_neighbor = w.reshape((w.shape[0],1,w.shape[1]))
            #neighbors' weights matrix, shape: (n_test,1,n_neighbors)
        
        prob = []
        label_neighbor = self.label[index] 
            #neighbors' index, shape: (n_test,n_neighbors,n_classes)
        for i in range(len(w_neighbor)):
            prob.append(np.dot(w_neighbor[i,:,:],label_neighbor[i,:,:]).ravel())
        prob = np.asarray(prob)
        self.prob = prob
        return prob
    
    def predict(self):
        
        #prob = predict_proba(self,X)
        yhat = self.classes[np.argmax(self.prob,axis=1)]
        return yhat




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
    X, y, train_size=600, test_size=100)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)



import matplotlib.pyplot as plt

acc=[]
acc_p=[]
acc_l=[]
acc_r=[]
acc_s=[]

for k in range(1, 500):
    KNN = KNeighborsClassifier(n_neighbors=k, p=2, n_jobs=-1)
    KNN.fit(X_train, y_train)
    y_pred=KNN.predict(X_test)
    acc.append(accuracy_score(y_test, y_pred))
    KNN_p = KernelKNN(n_neighbors=k, metric='poly', gamma=1/784)
    KNN_p.fit(X_train, y_train)
    KNN_p.predict_proba(X_test)
    y_pred = KNN_p.predict()
    acc_p.append(accuracy_score(y_test, y_pred))
    KNN_l = KernelKNN(n_neighbors=k, metric='linear', gamma=1/784)
    KNN_l.fit(X_train, y_train)
    KNN_l.predict_proba(X_test)
    y_pred = KNN_l.predict()
    acc_l.append(accuracy_score(y_test, y_pred))
    KNN_r = KernelKNN(n_neighbors=k, metric='rbf', gamma=1/784)
    KNN_r.fit(X_train, y_train)
    KNN_r.predict_proba(X_test)
    y_pred = KNN_r.predict()
    acc_r.append(accuracy_score(y_test, y_pred))
    KNN_s = KernelKNN(n_neighbors=k, metric='sigmoid', gamma=1/784)
    KNN_s.fit(X_train, y_train)
    KNN_s.predict_proba(X_test)
    y_pred = KNN_s.predict()
    acc_s.append(accuracy_score(y_test, y_pred))



plt.xlabel('k')
plt.ylabel('Acuuracy')
plt.title('Accuracy Curves of Kelnels')
plt.plot(acc, label='none')
plt.plot(acc_p, label='poly')
plt.plot(acc_s, label='sigmoid')
plt.plot(acc_r, label='rbf')
plt.plot(acc_l, label='linear')
plt.legend()




