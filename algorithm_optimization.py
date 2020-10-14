# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:44:25 2019

@author: Evan Macdonald
"""

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import Kintec_Functions as kf

comp = 'evanm'

#%% Import and pre-process data
data = kf.get_training_data(comp)   

order = 2
cutoff = 10
buffLen = 40
numvars = 16

X, Y = kf.pre_process_data(data,order,cutoff,buffLen,numvars)


#%% Select what features to use
# see 'Features Legend.xlsx' for details
select_feats = np.asarray([0,7,16,23, #FSR1
                           1,8,17,24, #FSR2
                           2,9,18,25, #FSR3
                           3,10,19,26, #FSR4
                           4,11,20,27, #FSR5
                           5,12,21,28, #FSR6
                           6,13,22,29, #FSR7
                           14,15,30,31 #Acc
                           ])
X_red = np.take(X,select_feats,axis=1)

#%% Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_red, Y, test_size=0.3, random_state=0)

#%% model setup
model = svm.SVC(C=105, kernel='poly', gamma=0.05)
#model = LogisticRegression(multi_class='multinomial',solver='newton-cg', C=5000, tol=0.01)

scores = cross_validate(model, X_train, y_train, cv=5)

print("Training Accuracy: %0.5f (+/- %0.5f)" % (scores['test_score'].mean(), scores['test_score'].std() * 2))

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
target_names = ['Sitting','Standing','Walking']
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print()
#%% grid search parameters for SVM
#print("Optimizing parameters")
#model = svm.SVC()
#tuning_parameters = [{'kernel': ['poly'], 'gamma': [0.06], 'C': [105]}]
#
#clf = GridSearchCV(model, tuning_parameters, cv=5,scoring='accuracy')
#clf.fit(X_train, y_train)
#
#print("Best parameters set found on development set:")
#print()
#print(clf.best_params_)
#print()
#print("Grid scores on development set:")
#print()
#means = clf.cv_results_['mean_test_score']
#stds = clf.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#    print("%0.4f (+/-%0.04f) for %r"
#          % (mean, std * 2, params))
#print()
#
#print("Detailed classification report:")
#print()
#print("The model is trained on the full development set.")
#print("The scores are computed on the full evaluation set.")
#print()
#y_true, y_pred = y_test, clf.predict(X_test)
#target_names = ['Sitting','Standing','Walking']
#print(classification_report(y_true, y_pred,target_names=target_names))
#print()

#%% optimize MLR
#print("Optimizing parameters")
#model = LogisticRegression(multi_class='multinomial')
#tuning_parameters = [{'solver': ['newton-cg'],'C': [5200,5500,5800,6000], 'tol': [0.1,0.01]}]
#
#clf = GridSearchCV(model, tuning_parameters, cv=5, scoring='accuracy')
#clf.fit(X_train, y_train)
#
#print("Best parameters set found on development set:")
#print()
#print(clf.best_params_)
#print()
#print("Grid scores on development set:")
#print()
#means = clf.cv_results_['mean_test_score']
#stds = clf.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#    print("%0.4f (+/-%0.04f) for %r"
#          % (mean, std * 2, params))
#print()
#
#print("Detailed classification report:")
#print()
#print("The model is trained on the full development set.")
#print("The scores are computed on the full evaluation set.")
#print()
#y_true, y_pred = y_test, clf.predict(X_test)
#target_names = ['Sitting','Standing','Walking']
#print(classification_report(y_true, y_pred,target_names=target_names))
#print()
