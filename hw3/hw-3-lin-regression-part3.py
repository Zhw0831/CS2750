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

# normalize the input
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# fit() computes mean and std for later scaling
scaler.fit(X)
# transform() performs centering and scaling
X_norm = scaler.transform(X)

# train/test split
X_train=X_norm[:-100]
X_test=X_norm[-100:]
Y_train=Y[:-100]
Y_test=Y[-100:]

# --------------------------------------------------
# implements the stochastic gradient descent procedure

# all weights initially start with 1
weights = np.ones([1,len(X_train[0,:])+1])

# add the bias to the input vector
bias_1 = np.ones([len(X_train[:,0]),1])
X_train = np.hstack([bias_1,X_train])
bias_2 = np.ones([len(X_test[:,0]),1])
X_test = np.hstack([bias_2,X_test])


counter = 0

while(counter<1000):
    i = counter%(len(X_train[:,0]))
    # select the next data point
    x_i = X_train[i,:]
    y_i = Y_train[i]
    # set the learning rate
    learning_rate = 0.05/(counter+1)
    # learning_rate = 0.05
    # learning_rate = 0.01
    # learning_rate = 2/(np.sqrt(len(X_train[:,0])))
    # calculate the linear model f
    f = np.dot(weights, x_i)
    # update the weight vector
    weights = weights + learning_rate*(y_i - f)*x_i
    # update the counter
    counter += 1

print(weights)

# apply the model to dataset and calculate training/testing errors
model_train = np.dot(weights,X_train.T)
print('Mean squared train error for online gradient descent: %.2f'
      % mean_squared_error(Y_train, model_train.T))

model_test = np.dot(weights,X_test.T)
print('Mean squared test error for online gradient descent: %.2f'
      % mean_squared_error(Y_test, model_test.T))






