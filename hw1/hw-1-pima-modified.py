#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:05:14 2022

@author: milos
"""

# Homework assignment 1 files
# pima dataset problems
  
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# -------------------------------------

# Problem 2

#load pima dataset 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
print('X dimensions:',X.shape)
print('Y dimensions:',Y.shape)
print(X[0:2,:])
print(Y[0:2])

# Problem 2 a
print('the min of each attribute:')
print(np.min(X, axis=0))

print('the max of each attribute:')
print(np.max(X, axis=0))

#Problem 2 b
print('the mean of each attribute:')
print(np.mean(X, axis=0))

print('the standard deviation of each attribute:')
print(np.std(X, axis=0))

# Problem 2 c 
# split the dataset into two subset based on the class value
zero_set = dataframe[dataframe['class'] == 0]
one_set = dataframe[dataframe['class'] == 1]

# calculate the mean ans std of the zero class set
print('the mean for class 0 attributes: \n' + str(np.mean(zero_set,axis=0)))
print('the std for class 0 attributes: \n' + str(np.std(zero_set,axis=0)))

# calculate the mean ans std of the one class set
print('the mean for class 1 attributes: \n' + str(np.mean(one_set,axis=0)))
print('the std for class 1 attributes: \n' + str(np.std(one_set,axis=0)))

# Problem 2 d
print('the correlation between each attribute and class label:')
for i in range(8):
    print(np.corrcoef(X[:,i],Y))

# Problem 2 e

# initialize the maximum correlation value and corresponding attributes for later use
max = 0
corr1 = names[0]
corr2 = names[0]

# for each attribute, calculate the correlation of it with other attributes
# this loop makes sure every two attributes are only measured once
for i in range(7):
    for j in range(i+1,8):
        print('The mutual correlation between ' + names[i] + ' and ' + names[j] + ' is ')
        corr = np.corrcoef(X[:,i],X[:,j])
        if abs(corr[0,1]) >= max:
            # change the maximum correlation value to the just-seen larger value
            max = corr[0,1]
            # update the corresponding attributes, too
            corr1 = names[i]
            corr2 = names[j]
        # only want the mutual correlation between these two attributes once
        print(corr[0,1])

print('The largest mutual correlation is ' + str(max) + ' between ' + corr1 + ' and ' + corr2)

# Problem 2 f
for i in range(9):
    plt.figure()
    # generate histograms for rach of the attributes with bins 20
    plt.hist(array[:,i],bins = 20)
    plt.suptitle("histogram of " + names[i])
    plt.show()

# Problem 2 g
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

# Problem 2 h
def scatter_plot(attr1, attr2):
    # values on x-axis are from the column index attr1 in X
    x = X[:,attr1]
    # values on y-axis are from the column index attr2 in X
    y = X[:,attr2]
    # plot the scatter plot
    plt.scatter(x,y)
    # edit x-y labels
    plt.xlabel(names[attr1])
    plt.ylabel(names[attr2])
    plt.show()

for i in range(7):
    for j in range(i+1,8):
        scatter_plot(i,j)

# ------------------------------------------------------

# Problem 3 b
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# fit() computes mean and std for later scaling
scaler.fit(X)
# transform() performs centering and scaling
print('the first 5 normalized values in the attribute pres:')
print(scaler.transform(X)[0:5,2])

# Problem 3 c
from sklearn.preprocessing import KBinsDiscretizer
# assume 10 equal-sized bins
discre = KBinsDiscretizer(n_bins = 10)
# fit() fits the estimator to "pres" data
pres_val = X[:,2].reshape(-1,1)
discre.fit(pres_val)
# transform() discretizes the data
print('the first 5 discretized values in the attribute pres:')
print(discre.transform(pres_val)[0:5])

# -------------------------------------------------------

# Problem 4 a
# split the train and test datasets with the test size of 0.33 and random seed of 7
train1, test1 = train_test_split(array, test_size=0.33, random_state=7)
# print out the size of both sets
print('the size of the training set (a):')
print(len(train1))
print('the size of the test set (a):')
print(len(test1))

# Problem 4 b
# split the train and test datasets with the test size of 0.33 and random seed of 3
train2, test2 = train_test_split(array, test_size=0.33, random_state=3)
# print out the size of both sets
print('the size of the training set (b):')
print(len(train2))
print('the size of the test set (b):')
print(len(test2))

# compare the instances of datasets generated in (a) and (b)
print('check if the traning set in (a) and (b) are the same:')
print((train1 in train2) and (train2 in train1))
print('check if the test set in (a) and (b) are the same:')
print((test1 in test2) and (test2 in test1))

# Problem 4 c
# split the train and test datasets with the test size of 0.25 and random seed of 7
train3, test3 = train_test_split(array, test_size=0.25, random_state=7)
# print out the size of both sets
print('the size of the training set (c):')
print(len(train3))
print('the size of the test set (c):')
print(len(test3))
