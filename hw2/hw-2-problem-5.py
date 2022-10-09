import numpy as np
import scipy
from scipy import stats
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv('gaussian1.txt')
sample = file.values[:,0]

# Part (a)
plt.figure()
 # generate histograms for the gaussian data with bins 10
plt.hist(sample,bins = 10)
plt.suptitle("histogram of gaussian1.txt data")
plt.show()

# Part (b)
# calculate ML estimate of mean and variance
sum_gaussian = sum(sample)
mean_ml = sum_gaussian/len(sample)

variance_sum = 0
for i in range(0,len(sample)):
    variance_sum += (sample[i]-mean_ml)**2

variance_ml = variance_sum/(len(sample)-1)

print('ML estimate of mean is ', str(mean_ml))
print('ML estimate of unbiased variance is ', str(variance_ml))

std_ml = np.sqrt(variance_ml)

x = np.linspace(mean_ml-3*std_ml,mean_ml+3*std_ml,1000)


fig, ax = plt.subplots(1,1)
ax.plot(x,stats.norm.pdf(x,loc=mean_ml, scale=std_ml))

ax.set_title('Gaussian distribution of the data')
plt.show()