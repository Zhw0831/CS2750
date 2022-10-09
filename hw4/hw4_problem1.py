#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:02:35 2022

@author: milos
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### ML Classification models
from sklearn.linear_model import LogisticRegression  # Log reg
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


dataframe1 = pd.read_csv('pima_train.csv')
array1 = dataframe1.values
X_train = array1[:,0:8]
Y_train= array1[:,8:10]

dataframe2 = pd.read_csv('pima_test.csv')
array2 = dataframe2.values
X_test = array2[:,0:8]
Y_test= array2[:,8]
targets=['class 0', 'class 1']

# Logistic regression
print("LogReg")

# initialize a LogReg model
logreg = LogisticRegression()
# train the Logreg model
logreg.fit(X_train, Y_train)
# make class prediction for the LogReg model
prediction_LR = logreg.predict(X_test)

# Test Accuracy
print("Test score: {:.2f}".format(logreg.score(X_test, Y_test)))

#Test Confusion matrix
conf_matrix = confusion_matrix(Y_test, prediction_LR)
print('confusion matrix for test data:\n',conf_matrix)
# extract components of the confusion matrix
tn, fp, fn, tp = confusion_matrix(Y_test, prediction_LR).ravel()

# calculate the misclassification error, SENS (recall),SPEC, PPV (precision), NPV on the test data
mis_err_test = (fp+fn)/(tn+fp+fn+tp)
sens_test = tp/(tp+fn)
spec_test = tn/(tn+fp)
ppv_test = tp/(tp+fp)
npv_test = tn/(tn+fn)

print('misclassification error for test data:\n',mis_err_test)
print('SENS for test data:\n',sens_test)
print('SPEC for test data:\n',spec_test)
print('PPV for test data:\n',ppv_test)
print('NPV for test data:\n',npv_test)

# predict the probability for class 1 (not just class label)
probs_LR=logreg.predict_proba(X_test)


# calculate AUROC
Auroc_score=roc_auc_score(Y_test, probs_LR[:,1])
print("AUROC score: {:.2f}".format(Auroc_score))
# ROC curve 
# curve=roc_curve(Y_test, logreg.predict_proba(X_test)[:,1])
# ROC curve components
fpr, tpr, thresholdsROC = roc_curve(Y_test, probs_LR[:,1])

# Draw the ROC curve
plt.figure(1)
# ROC curve components
fpr, tpr, thresholdsROC = roc_curve(Y_test, probs_LR[:,1])
#plot
plt.plot(fpr,tpr)
plt.title("ROC curve")
plt.show()

# Draw the PR curve
plt.figure(2)

auprc = average_precision_score(Y_test, probs_LR[:,1])
print("AUPRC score: {:.2f}".format(auprc))

# Components of the Precision recall curve
precision, recall, thresholdsPR = precision_recall_curve(Y_test, probs_LR[:,1])
# plot
plt.plot(recall,precision)
plt.title("PR curve")
plt.show()

# for train data

# make class prediction for the LogReg model
prediction_LR_train = logreg.predict(X_train)

# Test Accuracy
print("Test score for train data: {:.2f}".format(logreg.score(X_train, Y_train)))

#Test Confusion matrix
conf_matrix_train = confusion_matrix(Y_train, prediction_LR_train)
print('confusion matrix for train data:\n',conf_matrix_train)
# extract components of the confusion matrix
tn_t, fp_t, fn_t, tp_t = confusion_matrix(Y_train, prediction_LR_train).ravel()

# calculate the misclassification error, SENS (recall),SPEC, PPV (precision), NPV on the train data
mis_err_train = (fp_t+fn_t)/(tn_t+fp_t+fn_t+tp_t)
sens_train = tp_t/(tp_t+fn_t)
spec_train = tn_t/(tn_t+fp_t)
ppv_train = tp_t/(tp_t+fp_t)
npv_train = tn_t/(tn_t+fn_t)

print('misclassification error for train data:\n',mis_err_train)
print('SENS for train data:\n',sens_train)
print('SPEC for train data:\n',spec_train)
print('PPV for train data:\n',ppv_train)
print('NPV for train data:\n',npv_train)