import pickle
import pandas as pd
import numpy as np
import random
from sklearn.metrics import multilabel_confusion_matrix
from scipy.stats import ttest_ind_from_stats

# load pickle part: credits to https://python.tutorialink.com/attributeerror-when-reading-a-pickle-file/

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "NB_EM_learning"
        return super().find_class(module, name)
    
print('load the NB model:')

with open(r'C:\Users\Zhen Wu\Desktop\CS2750\hw8\data\em', 'rb') as f:
    unpickler = MyCustomUnpickler(f)
    obj = unpickler.load()

# load the parameters
pi = obj.pi
theta = obj.theta

# cannot pickle methods, sad

def compute_class(data_entry):
    # most part of this part credits to TA Mesut
    # find the most likely class based on attributes 1-5
    cgivend = np.ones(4) 
    for n in range(len(data_entry)): 
        if data_entry[n] != 0: 
            cgivend *= theta[n,:,data_entry[n]-1] 
    cgivend *= pi 
    cgivend /= np.sum(cgivend)
    return np.argmax(cgivend) + 1

# load data and split attributes
data = pd.read_csv(r'C:\Users\Zhen Wu\Desktop\CS2750\hw8\data\testing_data.csv').values
train = data[:,0:5]
label = data[:,5]


# get predictions
pred = []
for n in range(len(train)):
    customer_class = compute_class(train[n])
    most_likely_k = np.argmax(theta[5][customer_class-1]) + 1
    pred.append(most_likely_k)

# randomized predictor
pred2 = []
for n in range(len(train)):
    random.seed(2)
    pred2.append(random.randrange(1,5,1))


# compute metrics
conf_matrix = multilabel_confusion_matrix(label, pred)
print('confusion matrix for test data:\n',conf_matrix)
mis_err = []
for c in conf_matrix:
    tn, fp, fn, tp = np.ravel(c)
    mis_err_test = (fp+fn)/(tn+fp+fn+tp)
    print('misclassification error\n',mis_err_test)
    mis_err.append(mis_err_test)

print('mean misclassification error\n',np.mean(mis_err))

# check if the two results are significantly different
std1 = np.std(pred)
std2 = np.std(pred2)
mean1 = np.mean(pred)
mean2 = np.mean(pred2)
size = len(label)

tstat, pvalue = ttest_ind_from_stats(mean1, std1, size, mean2, std2, size)
print('p value is: ',pvalue)



