#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 00:36:18 2022

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

# split the dataset into two subset based on the class value
zero_set = dataframe1[dataframe1['class'] == 0]
one_set = dataframe1[dataframe1['class'] == 1]

exp_attr = [1, 5, 7, 8]
normal_attr = [2, 3, 4, 6]

# sample: the dataset values, attr_num: determine which distribution to use
def fit_NB(sample, attr_num):
    # [1 5 7 8] - exponential distribution
    if attr_num in exp_attr:
        miu = sum(sample)/len(sample)
        return miu
    # [2 3 4 6] - normal distribution
    elif attr_num in normal_attr:
        sum_gaussian = sum(sample)
        mean_ml = sum_gaussian/len(sample)

        variance_sum = 0
        for i in range(0,len(sample)):
            variance_sum += (sample[i]-mean_ml)**2
        # unbiased
        variance_ml = variance_sum/(len(sample)-1)

        return [mean_ml,variance_ml]

p_0 = len(zero_set)/len(X_train)
p_1 = len(one_set)/len(X_train)

def predict_NB(conditional_0, conditional_1):
    sum_0 = 0
    sum_1 = 0
    for i in range(len(conditional_0)):
        sum_0 += np.log(conditional_0[i])
        sum_1 += np.log(conditional_1[i])

    g_0 = sum_0 + np.log(p_0)
    g_1 = sum_1 + np.log(p_1)

    if(g_1>=g_0):
        return 1
    else:
        return 0       


def predict_prob(x_0,x_1):
    mult_0 = 1
    mult_1 = 1
    for i in range(len(x_0)):
        mult_0 *= x_0[i]
    for i in range(len(x_1)):
        mult_1 *= x_1[i]
    
    rslt = (mult_1*p_1)/(mult_0*p_0 + mult_1*p_1)

    return rslt


# apply the functions on the data

# parameters estimation
# class 0
zero_para = []
for i in range(8):
    sample = zero_set.values[:,i]
    zero_para.append(fit_NB(sample, i+1))
    print('parameter estimate for class 0, attribute ', str(i+1))
    print('\n',zero_para[i],'\n')

# class 1
one_para = []
for i in range(8):
    sample = one_set.values[:,i]
    one_para.append(fit_NB(sample,i+1))
    print('parameter estimate for class 1, attribute ', str(i+1))
    print('\n',one_para[i],'\n')

# classify the test data
pred = []
probs_test = []

for i in range(len(X_test)):
    zero_prob = []
    one_prob = []
    x = X_test[i,:]
    for j in range(len(x)):
        # exponential distribution attributes
        if j+1 in exp_attr:
            miu_0 = zero_para[j]
            miu_1 = one_para[j]

            val_0 = (1/miu_0)*(np.exp(-x[j]/miu_0)) 
            zero_prob.append(val_0)            
            
            val_1 = (1/miu_1)*(np.exp(-x[j]/miu_1))
            one_prob.append(val_1)
        # normal distribution attributes
        elif j+1 in normal_attr:
            mean_0 = zero_para[j][0]
            var_0 = zero_para[j][1]
            mean_1 = one_para[j][0]
            var_1 = one_para[j][1]
                
            val_0 = (1/(np.sqrt(var_0)*np.sqrt(2*np.pi)))*(np.exp(-((x[j]-mean_0)**2/(2*(var_0)))))      
            zero_prob.append(val_0)            
            
            val_1 = (1/(np.sqrt(var_1)*np.sqrt(2*np.pi)))*(np.exp(-((x[j]-mean_1)**2/(2*(var_1)))))         
            one_prob.append(val_1)

    label = predict_NB(zero_prob,one_prob)
    prob = predict_prob(zero_prob,one_prob)
    pred.append(label)
    probs_test.append(prob)


#Test Confusion matrix
conf_matrix = confusion_matrix(Y_test, pred)
print('confusion matrix for test data:\n',conf_matrix)
# extract components of the confusion matrix
tn, fp, fn, tp = confusion_matrix(Y_test, pred).ravel()

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

# plot the ROC curve
Auroc_score=roc_auc_score(Y_test, probs_test)
print("AUROC score: {:.2f}".format(Auroc_score))

plt.figure(1)
fpr, tpr, thresholdsROC = roc_curve(Y_test, probs_test)
plt.plot(fpr,tpr)
plt.title("ROC curve")
plt.show()

# plot the PR curve
auprc = average_precision_score(Y_test, probs_test)
print("AUPRC score: {:.2f}".format(auprc))

plt.figure(2)
precision, recall, thresholdsPR = precision_recall_curve(Y_test, probs_test)
plt.plot(recall,precision)
plt.title("PR curve")
plt.show()









    

