import pickle
import pandas as pd 
import numpy as np

class EM():
    def __init__(self):
        # read in the test file
        self.df = pd.read_csv(r'C:\Users\Zhen Wu\Desktop\CS2750\hw8\data\training_data.csv').values
        # initialize Q
        self.Q_val = -1e10
        # initialize the parameters
        np.random.seed(2)
        self.pi = np.random.random(4) # class priors
        np.random.seed(2)
        self.theta = np.random.random([6,4,5]) # random probabilities for conditionals
        for i in range(len(self.theta)):
            self.theta[i] = [n/sum(n) for n in self.theta[i]] # normalize for each class


    def expectation_term_1(self,data_entry,j):
        # this part credits to TA Mesut
        cgivend = np.ones(4) 
        for n in range(len(data_entry)): 
            if data_entry[n] != 0: 
                cgivend *= self.theta[n,:,data_entry[n]-1] 
        cgivend *= self.pi 
        cgivend /= np.sum(cgivend) 
        return cgivend[j-1]


    def expectation_term_2(self,data_entry,i,j,k):
        # if the value is not k and not missing, do not bother
        if(data_entry[i-1]!=0 and data_entry[i-1]!=k):
            return 0
        # if the value is valid
        else:
            first_multiplier = self.expectation_term_1(data_entry,j)
            if(data_entry[i-1]==0): # if missing, need to multiply by the estimate
                second_multiplier = self.theta[i-1][j-1][k-1]
                first_multiplier *= second_multiplier

            return first_multiplier

    
    def compute_N_terms_1(self,j):
        N_j = 0

        for n in range(len(self.df)):
            data_entry = self.df[n,:]
            N_j += self.expectation_term_1(data_entry,j)
        
        return N_j

    def compute_N_terms_2(self,i,j,k):
        N_ijk = 0

        for n in range(len(self.df)):
            data_entry = self.df[n,:]
            N_ijk += self.expectation_term_2(data_entry,i,j,k)
        
        return N_ijk

    def compute_para_pi(self,j):
        # compute pi
        pi_denominator = 0
        pi_numerator = 0
        for n in range(4):
            pi_component = self.compute_N_terms_1(n+1)
            pi_denominator += pi_component
            if(n+1==j):
                pi_numerator = pi_component
        pi_para = pi_numerator/pi_denominator

        return pi_para

    def compute_para_theta(self,i,j,k):
        # compute theta
        theta_denominator = 0
        theta_numerator = 0
        for n in range(5):
            theta_component = self.compute_N_terms_2(i,j,n+1)
            theta_denominator += theta_component
            if(n+1==k):
                theta_numerator = theta_component
        theta_para = theta_numerator/theta_denominator

        return theta_para
    
    def compute_Q(self):
        # initialize matrices for new parameters
        # if the new Q is larger than the previous one, set the old parameters to these
        pi_update = np.zeros(4)
        theta_update = np.zeros([6,4,5])

        # compute the first part involving pi
        first_part = 0
        # for each of the 4 classes
        for n in range(1,5):
            N_j = self.compute_N_terms_1(n)
            pi_new = self.compute_para_pi(n)
            log_pi = np.log(pi_new)
            first_part += N_j*log_pi

            # update the parameters
            pi_update[n-1] = pi_new
        
        # compute the second part involving theta
        second_part = 0
        # for each of the classes (4 classes in total)
        for j in range(1,5):
            # for each of the attributes (6 attributes in total)
            for i in range(1,7):
                for k in range(1,6):
                    N_ijk = self.compute_N_terms_2(i,j,k)
                    theta_new = self.compute_para_theta(i,j,k)
                    log_theta = np.log(theta_new)
                    second_part += N_ijk*log_theta

                    # update the parameters
                    theta_update[i-1][j-1][k-1] = theta_new
        
        # now compute Q
        Q_new = first_part + second_part

        if(Q_new > self.Q_val and Q_new-self.Q_val > 0.001):
            setattr(self,'pi',pi_update)
            setattr(self,'theta',theta_update)

        return Q_new

# apply and save the model
a = EM()
flag = 0
while(flag==0):
    Q_old = a.Q_val
    Q_new = a.compute_Q()
    # if only very small difference, stop the iterations
    if(Q_new-Q_old) <= 0.001:
        flag = 1
    if(Q_new > Q_old):
        a.Q_val = Q_new

print(a.pi)

pickle.dump(a,open(r'C:\Users\Zhen Wu\Desktop\CS2750\hw8\data\em','wb'))


            

        


    

