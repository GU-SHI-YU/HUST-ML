#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt

pos = [0, 1, 2, 3]
acc = [0.97, 0.93, 0.97, 0.83]
name = ['KNN', 'LogisticRegression', 'SVM', 'Naive Bayes']
rects = plt.bar(pos, acc)
for rect in rects:
    plt.text(rect.get_x()+0.4, rect.get_height(), rect.get_height(), ha='center', va='bottom')
plt.title('Accuracies of Classifiers')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.xticks(pos, name)
plt.show()


plt.cla()
acc = [298.00, 119.04, 366.97, 0.57]
rects = plt.bar(pos, acc)
for rect in rects:
    plt.text(rect.get_x()+0.4, rect.get_height(), rect.get_height(), ha='center', va='bottom')
plt.title('Predicting Time of Classifiers')
plt.xlabel('Classifier')
plt.ylabel('Time/s')
plt.xticks(pos, name)
plt.show()




