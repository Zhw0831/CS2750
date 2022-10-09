import numpy as np
import scipy
from scipy import stats
from scipy.stats import poisson
from scipy.stats import gamma
import pandas as pd
import matplotlib.pyplot as plt

# Part (a)
lambda_1 = 2
lambda_2 = 6

x_1 = range(10)
x_2 = range(15)

fig, ((ax1,ax2)) = plt.subplots(1,2)

ax1.plot(stats.poisson.pmf(x_1,mu=lambda_1),marker='o')
ax2.plot(stats.poisson.pmf(x_2,mu=lambda_2),marker='o')

ax1.set_title('Poisson distribution (lambda=2)')
ax2.set_title('Poisson distribution (lambda=6)')

plt.show()

# Part (b)
# use the equation derived from part 1
file = pd.read_csv('poisson.txt')
sample = file.values[:,0]
lambda_ml = sum(sample)/len(sample)
print('The ML estimate of lambda is ', str(lambda_ml))

fig, ax = plt.subplots(1,1)
x_3 = range(15)
ax.plot(stats.poisson.pmf(x_3,mu=lambda_ml),marker='o')
ax.set_title('Poisson distribution (lambda=5.24)')

plt.show()

# Part (c)
x_4 = np.linspace(0,25,100)
y1 = stats.gamma.pdf(x_4,a=1,scale=2)
y2 = stats.gamma.pdf(x_4,a=3,scale=5)

plt.plot(x_4, y1, label='a=1, b=2')
plt.plot(x_4, y2, label='a=3, b=5')
plt.legend()
plt.show()

# Part (d)
a1 = 1+sum(sample)
b1 = 2/((len(sample)*2)+1)
a2 = 3+sum(sample)
b2 = 5/((len(sample)*5)+1)

y3 = stats.gamma.pdf(x_4,a=a1,scale=b1)
y4 = stats.gamma.pdf(x_4,a=a2,scale=b2)

fig, ax = plt.subplots(1,1)
ax.plot(x_4, y3)

ax.set_title('posterior distributions for (orginally) a=1,b=2')
plt.show()

fig, ax = plt.subplots(1,1)
ax.plot(x_4, y4)

ax.set_title('posterior distributions for (orginally) a=3,b=5')
plt.show()

