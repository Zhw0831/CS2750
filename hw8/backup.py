import pickle
import pandas as pd 
import numpy as np

class EM():
    def __init__(self):
        # seed the random for later use
        np.random.seed(2)
        # read in the test file
        self.df = pd.read_csv(r'C:\Users\Zhen Wu\Desktop\CS2750\hw8\data\training_data.csv').values
        # initialize Q
        self.Q_val = -1e10
        # initialize the parameters
        self.pi = np.random.random(4)
        # the sum of the probabilities should be 1
        self.pi = self.pi/sum(self.pi)
        # for each attribute, the list is organized in this manner:
        # list = [[all possible values of the attribute given class=1], [...given class=2], [...given class=3], [...given class=4]]
        # for attribute x1,x5 (because they both have 5 possible values)
        np.random.seed(2)
        item_1 = np.random.random([4,5])
        self.x1_list = [m/sum(m) for m in item_1]
        np.random.seed(2)
        item_1 = np.random.random([4,5])
        self.x5_list = [m/sum(m) for m in item_1]
        # for attribute x2,x3 (because they both have 3 possible values)
        np.random.seed(2)
        item_1 = np.random.random([4,3])
        self.x2_list = [m/sum(m) for m in item_1]
        np.random.seed(2)
        item_1 = np.random.random([4,3])
        self.x3_list = [m/sum(m) for m in item_1]
        # for attribute x4,x6 (because they both have 4 possible values)
        np.random.seed(2)
        item_1 = np.random.random([4,4])
        self.x4_list = [m/sum(m) for m in item_1]
        np.random.seed(2)
        item_1 = np.random.random([4,4])
        self.x6_list = [m/sum(m) for m in item_1]

    def expectation_term_1(self,data_entry,j):
        mult_class_1 = []
        mult_class_2 = []
        mult_class_3 = []
        mult_class_4 = []

        for i in range(len(data_entry)):
            # if the value is missing
            if(data_entry[i]==0):
                mult_class_1.append(1)
                mult_class_2.append(1)
                mult_class_3.append(1)
                mult_class_4.append(1)
            # if the value is not missing, apply the conditional probabilities in the parameters
            else:
                list_name = "x"+str(i+1)+"_list"
                ind = data_entry[i]-1
                attr_list = getattr(self,list_name)
                
                mult_class_1.append(attr_list[0][ind])
                mult_class_2.append(attr_list[1][ind])
                mult_class_3.append(attr_list[2][ind])
                mult_class_4.append(attr_list[3][ind])

        # calculate p(c=j|D,theta)
        # the denominator considers all 4 classes
        nominator = 0
        denominator = 0
        for i in range(4):
            component = self.pi[i]*np.prod(eval("mult_class_"+str(i+1)))
            denominator += component
            # pick one of them to be the nominator by the class label
            if(i+1==j):
                nominator = component
        
        return nominator/denominator

    def expectation_term_2(self,data_entry,i,j,k):
        # if the value is not k and not missing, do not bother
        if(data_entry[i-1]!=0 and data_entry[i-1]!=k):
            return 0
        # if the value is valid (discuss missing or not later because these 2 cases have a common factor)
        else:
            first_multiplier = self.expectation_term_1(data_entry,j)
            list_name = "x"+str(i)+"_list"
            # ind = k-1
            # print(ind)
            attr_list = getattr(self,list_name)

            # if the value is missing, we need to compute the second multiplier
            if(data_entry[i-1]==0):
                second_multiplier = attr_list[j-1][k-1]
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
        pi_nominator = 0
        for n in range(4):
            pi_component = self.compute_N_terms_1(n+1)
            pi_denominator += pi_component
            if(n+1==j):
                pi_nominator = pi_component
        pi_para = pi_nominator/pi_denominator

        return pi_para

    def compute_para_theta(self,i,j,k):
        # compute theta
        list_name = "x"+str(i)+"_list"
        attr_list = getattr(self,list_name)

        theta_denominator = 0
        theta_nominator = 0
        for n in range(1,len(attr_list[0])+1):
            theta_component = self.compute_N_terms_2(i,j,n)
            theta_denominator += theta_component
            if(n==k):
                theta_nominator = theta_component
        theta_para = theta_nominator/theta_denominator

        return theta_para
    
    def compute_Q(self):
        # initialize matrices for new parameters
        # if the new Q is larger than the previous one, set the old parameters to these
        pi_update = np.zeros(4)
        x1_update = np.zeros((4,5))
        x2_update = np.zeros((4,3))
        x3_update = np.zeros((4,3))
        x4_update = np.zeros((4,4))
        x5_update = np.zeros((4,5))
        x6_update = np.zeros((4,4))

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
                list_name = "x"+str(i)+"_list"
                attr_list = getattr(self,list_name)
                length = len(attr_list[0])
                # for each possible value of the attribute
                for k in range(1,length+1):
                    N_ijk = self.compute_N_terms_2(i,j,k)
                    theta_new = self.compute_para_theta(i,j,k)
                    log_theta = np.log(theta_new)
                    second_part += N_ijk*log_theta

                    # update the parameters
                    update_list_name = "x"+str(i)+"_update"
                    eval(update_list_name)[j-1][k-1] = theta_new
        
        # now compute Q
        Q_new = first_part + second_part

        if(Q_new > self.Q_val and Q_new-self.Q_val > 0.001):
            setattr(self,'pi',pi_update)
            setattr(self,'x1_list',x1_update)
            setattr(self,'x2_list',x2_update)
            setattr(self,'x3_list',x3_update)
            setattr(self,'x4_list',x4_update)
            setattr(self,'x5_list',x5_update)
            setattr(self,'x6_list',x6_update)

        return Q_new

# apply and save the model
a = EM()
flag = 0
while(flag==0):
    Q_old = a.Q_val
    Q_new = a.compute_Q()
    print(Q_new)
    # if only very small difference, stop the iterations
    if(Q_new-Q_old) <= 0.001:
        flag = 1
    if(Q_new > Q_old):
        a.Q_val = Q_new
    print(a.pi)

pickle.dump(a,open(r'C:\Users\Zhen Wu\Desktop\CS2750\hw8\data\em','wb'))


            

        


    

