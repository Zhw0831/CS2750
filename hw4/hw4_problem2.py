#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 00:42:08 2022

@author: milos
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### ML Classification models
from sklearn.linear_model import LogisticRegression  # Log reg
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_recall_curve


dataframe = pd.read_csv('pima.csv')
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
targets=['class 0', 'class 1']

# split the dataset into two subset based on the class value
zero_set = dataframe[dataframe['class'] == 0]
one_set = dataframe[dataframe['class'] == 1]

# plot histogram for two classes
for i in range(8):
    # set up for plotting side by side
    fig, ((ax1,ax2)) = plt.subplots(1, 2)
    # plot the histogram of the zero-class values on the left side
    ax1.hist(zero_set.values[:,i],bins=20,color='orange')
    ax1.set_title('class 0')
    # plot the histogram of the one-class values on the left side
    ax2.hist(one_set.values[:,i],bins=20,color='b')
    ax2.set_title('class 1')

    plt.suptitle("histogram of " + names[i])

    plt.show()

