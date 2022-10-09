import numpy as np
import scipy
from scipy import stats
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('coin.txt')
sample = data.values[:,0]

count_head = 0
count_tail = 0

# count the number of heads and tails
for i in sample:
    if(i==1):
        count_head += 1
    elif(i==0):
        count_tail += 1

# Part (a)
ml_head = count_head/len(sample)
print("The ML estimate of theta is " + str(ml_head))

# Part (b)
fig, ((ax1,ax2)) = plt.subplots(1, 2)

x = np.arange(0., 1., 0.01)

a_prior = 1
b_prior = 1

ax1.plot(x, beta.pdf(x, a_prior, b_prior))
ax1.set_title('beta distribution of the prior')

# posterior is still a beta distribution with parameters N1+a and N2+b
a_post = a_prior + count_head
b_post = b_prior + count_tail

ax2.plot(x, beta.pdf(x, a_post, b_post))
ax2.set_title('beta distribution of the posterior')

plt.suptitle('beta distribution with prior of beta(theta|1,1)')
plt.show()

# Part (c)
# MAP
fig, ax = plt.subplots(1,1)
ax.plot(x, beta.pdf(x, a_post, b_post))

ax.plot(0.65, beta.pdf(0.65,a_post,b_post),marker="o", markersize=5)
ax.annotate(text='MAP at theta=0.65',xy=(0.65,beta.pdf(0.65,a_post,b_post)))

ax.set_title('beta distribution of the posterior with prior of beta(theta|1,1)')
plt.show()

# Expected value
fig, ax = plt.subplots(1,1)
ax.plot(x, beta.pdf(x, a_post, b_post))

ax.plot(33/51, beta.pdf(33/51,a_post,b_post),marker="o", markersize=5)
ax.annotate(text='Expected at theta=33/51',xy=(33/51,beta.pdf(33/51,a_post,b_post)))

ax.set_title('beta distribution of the posterior with prior of beta(theta|1,1)')
plt.show()

# Part d
fig, ((ax1,ax2)) = plt.subplots(1, 2)

a_prior_d = 4
b_prior_d = 2

ax1.plot(x, beta.pdf(x, a_prior_d, b_prior_d))
ax1.set_title('beta distribution of the prior')

# posterior is still a beta distribution with parameters N1+a and N2+b
a_post_d = a_prior_d + count_head
b_post_d = b_prior_d + count_tail

ax2.plot(x, beta.pdf(x, a_post_d, b_post_d))
ax2.set_title('beta distribution of the posterior')

plt.suptitle('beta distribution with prior of beta(theta|4,2)')
plt.show()

# MAP
fig, ax = plt.subplots(1,1)
ax.plot(x, beta.pdf(x, a_post_d, b_post_d))

ax.plot(17/26, beta.pdf(17/26,a_post_d,b_post_d),marker="o", markersize=5)
ax.annotate(text='MAP at theta=17/26',xy=(17/26,beta.pdf(17/26,a_post_d,b_post_d)))

ax.set_title('beta distribution of the posterior with prior of beta(theta|4,2)')
plt.show()

# Expected value
fig, ax = plt.subplots(1,1)
ax.plot(x, beta.pdf(x, a_post_d, b_post_d))

ax.plot(69/106, beta.pdf(69/106,a_post_d,b_post_d),marker="o", markersize=5)
ax.annotate(text='Expected at theta=69/106',xy=(69/106,beta.pdf(69/106,a_post_d,b_post_d)))

ax.set_title('beta distribution of the posterior with prior of beta(theta|4,2)')
plt.show()


