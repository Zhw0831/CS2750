#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 08:50:08 2022

@author: milos
"""

#### stats basics: calculate sample mean, standard error and a confidence interval

## sample
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt

#### the function calculates basic sample stats: sample mean, standard error and a confidence interval
def confidence_interval(sample, confidence_level):
    # the function calculates the sample mean, sample standard error, and confidence interval
    degrees_freedom = sample.size - 1
    sample_mean = np.mean(sample)
    sample_standard_error = scipy.stats.sem(sample)
    # compute confidence interval for `sample`
    confidence_interval = scipy.stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
    print('Sample mean', sample_mean)
    print('Sample standard error:', sample_standard_error)
    print('Confidence interval:',confidence_interval)
    return sample_mean, sample_standard_error, confidence_interval

# loading the data from the text file
dataframe = pd.read_csv("mean_study_data.txt")
array = dataframe.values
sample = array[:,0]
# set confidence_level
confidence_level = 0.95
# calculate basic sample stats for all examples
samp_mean, samp_stde, conf_interval=confidence_interval(sample,confidence_level)

print(samp_mean, samp_stde, conf_interval)


# part 1
# read in the data file
file = pd.read_csv("mean_study_data.txt")
values = file.values[:,0]
# calculate mean and std by numpy
print('the mean for data: ' + str(np.mean(values,axis=0)))
print('the std for data: ' + str(np.std(values,axis=0)))

# part 2
def subsample(data,k):
    return data.sample(n=k)

# part 3
means_25 = []

for i in range(1,1001):
    sub_sample_25 = subsample(file, 25)
    sub_mean_25 = np.mean(sub_sample_25,axis=0)
    means_25.append(sub_mean_25)

plt.figure()
plt.hist(np.array(means_25),bins = 20)
plt.suptitle("histogram of 1000 subsample (size 25) means")
plt.show()

# part 5
means_40 = []

for i in range(1,1001):
    sub_sample_40 = subsample(file, 40)
    sub_mean_40 = np.mean(sub_sample_40,axis=0)
    means_40.append(sub_mean_40)

plt.figure()
plt.hist(np.array(means_40),bins = 20)
plt.suptitle("histogram of 1000 subsample (size 40) means")
plt.show()

# part 6
print("for the first 25 examples:")
first_25 = file.head(25).values[:,0]
samp_mean_25, samp_stde_25, conf_interval_25=confidence_interval(first_25,confidence_level)
print(samp_mean_25, samp_stde_25, conf_interval_25)



