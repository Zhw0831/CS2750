#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:15:17 2022

@author: milos
"""

# Problem 1 - nonparametric density estimation


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# loads the true pdf values on interval [0,1]
dataframe = pd.read_csv("true_pdf.csv")
true_dist = dataframe.values
#plots the true distribution
plt.plot(true_dist[:,0],true_dist[:,1])

# load the training data
datainstances = pd.read_csv("data-NDE.csv")
data=datainstances.values
plt.hist(data,10)

# Part (b)
def Parzen_window_pdf(x, D, h):
    K = 0
    # the number of all data instances
    N = len(D)
    # volume
    V = h

    for i in D:
        # flag indicates if the point is within the window
        flag = 1
        # find the absolute distance between the point to the center x
        dist = np.abs(i-x)
        # if the point is outside the window, set flag to 0
        if dist/h > 1/2:
            flag = 0
        K += flag

    return K/(N*V)

# Part (c)
p_sample_1 = []
p_sample_2 = []
p_sample_3 = []

for t in true_dist[:,0]:
    p_1 = Parzen_window_pdf(t,data,0.02)
    p_sample_1.append(p_1)

    p_2 = Parzen_window_pdf(t,data,0.05)
    p_sample_2.append(p_2)

    p_3 = Parzen_window_pdf(t,data,0.1)
    p_sample_3.append(p_3)

x = true_dist[:,0]
y = true_dist[:,1]

plt.figure()
plt.plot(x,y)
plt.plot(x,p_sample_1)
plt.legend(['true density','h=0.02'])
plt.suptitle('Parzen window h = 0.02')
plt.show()

plt.figure()
plt.plot(x,y)
plt.plot(x,p_sample_2)
plt.legend(['true density','h=0.05'])
plt.suptitle('Parzen window h = 0.05')
plt.show()

plt.figure()
plt.plot(x,y)
plt.plot(x,p_sample_3)
plt.legend(['true density','h=0.1'])
plt.suptitle('Parzen window h = 0.1')
plt.show()

#  # Part (d)
def Gaussian_kernel_pdf(x, D, h):
    total = 0
    N = len(D)
    pi = np.pi
    c = 1/((2*pi*(h**2))**(1/2))

    for i in D:
        kernel = c*(np.exp(-((x-i)**2)/(2*(h**2))))
        total += kernel

    return total/N 

p_sample_g1 = []
p_sample_g2 = []
p_sample_g3 = []
for t in true_dist[:,0]:
   p_g1 = Gaussian_kernel_pdf(t,data,0.1)
   p_sample_g1.append(p_g1)

   p_g2 = Gaussian_kernel_pdf(t,data,0.16)
   p_sample_g2.append(p_g2)

   p_g3 = Gaussian_kernel_pdf(t,data,0.25)
   p_sample_g3.append(p_g3)

plt.figure()
plt.plot(x,y)
plt.plot(x,p_sample_g1)
plt.legend(['true density','h=0.1'])
plt.suptitle('smooth Gaussian h = 0.1')
plt.show()

plt.figure()
plt.plot(x,y)
plt.plot(x,p_sample_g2)
plt.legend(['true density','h=0.16'])
plt.suptitle('smooth Gaussian h = 0.16')
plt.show()

plt.figure()
plt.plot(x,y)
plt.plot(x,p_sample_g3)
plt.legend(['true density','h=0.25'])
plt.suptitle('smooth Gaussian h = 0.25')
plt.show()
    

# Part (e)
def knn_estimate_pdf(x,D,k):
    distance = []
    N = len(D)

    for i in D:
        dist = np.abs(x-i)
        distance.append(dist)

    # sort the distance vector in ascending order
    distance.sort()
    # pick the kth distance from the center point
    r = distance[k-1]
    if(r==0):
        V = 1 
    else: 
        V = 2*r
    return k/(N*V)

p_sample_k1 = []
p_sample_k2 = []
p_sample_k3 = []

for t in true_dist[:,0]:
    p_k1 = knn_estimate_pdf(t,data,1)
    p_sample_k1.append(float(p_k1))

    p_k2 = knn_estimate_pdf(t,data,3)
    p_sample_k2.append(float(p_k2))

    p_k3 = knn_estimate_pdf(t,data,5)
    p_sample_k3.append(float(p_k3))


plt.figure()
plt.plot(x,y)
plt.plot(x,p_sample_k1)
plt.legend(['true density','k=1'])
plt.suptitle('knn estimator k = 1')
plt.show()

plt.figure()
plt.plot(x,y)
plt.plot(x,p_sample_k2)
plt.legend(['true density','k=3'])
plt.suptitle('knn estimator k = 3')
plt.show()

plt.figure()
plt.plot(x,y)
plt.plot(x,p_sample_k3)
plt.legend(['true density','k=5'])
plt.suptitle('knn estimator k = 5')
plt.show()





