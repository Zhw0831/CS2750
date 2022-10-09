#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:17:41 2022

@author: milos
"""

# Problem 2. Part 2. Quadratic model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import mean_squared_error

# load the boston housing dataset
housing = datasets.fetch_openml(name='boston',version=1)
housing.details 
X=housing.data
Y=housing.target

# train/test split
X_train=X[:-100]
X_test=X[-100:]
Y_train=Y[:-100]
Y_test=Y[-100:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)
# Print regression coefficients
print('Coefficients: \n', regr.coef_)


# Make predictions on the testing set
Y_test_pred = regr.predict(X_test)
# The mean squared error on the test set
print('Mean squared testing error for linear model: %.2f'
      % mean_squared_error(Y_test, Y_test_pred))

# Make predictions on the training set
Y_train_pred = regr.predict(X_train)
# The mean squared error on the train set
print('Mean squared training error for linear model: %.2f'
      % mean_squared_error(Y_train, Y_train_pred))

# Part 2.2
# (a)
def quadratic_expansion(x):
      v =[]

      for i in range(0,len(x)):
            for j in range(i,len(x)):
                  combination = float(x[i])*float(x[j])
                  v.append(combination)
      return v

# (b)
# take one example from our dataset to calculate the new dimension
vector = X_train.values[0,:]
print(np.shape(quadratic_expansion(vector)))

# (c)
X_trans_train = quadratic_expansion(vector)
for i in range(1,len(X_train)):
      old_x_train = X_train.values[i,:]
      new_x_train = quadratic_expansion(old_x_train)
      X_trans_train = np.vstack([X_trans_train, new_x_train])

X_trans_test = quadratic_expansion(X_test.values[0,:])
for i in range(1,len(X_test)):
      old_x_test = X_test.values[i,:]
      new_x_test = quadratic_expansion(old_x_test)
      X_trans_test = np.vstack([X_trans_test, new_x_test])

regr = linear_model.LinearRegression()
regr.fit(X_trans_train, Y_train)

# Make predictions on the testing set
Y_test_pred_trans = regr.predict(X_trans_test)
# The mean squared error on the test set
print('Mean squared testing error for quadratic model: %.2f'
      % mean_squared_error(Y_test, Y_test_pred_trans))

# Make predictions on the training set
Y_train_pred_trans = regr.predict(X_trans_train)
# The mean squared error on the train set
print('Mean squared training error for quadratic model: %.2f'
      % mean_squared_error(Y_train, Y_train_pred_trans))




       
                    
