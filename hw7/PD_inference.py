import pickle
import pandas as pd
import numpy as np

# load pickle part: credits to https://python.tutorialink.com/attributeerror-when-reading-a-pickle-file/

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "PD_learning"
        return super().find_class(module, name)
    
print('load the bbn model:')

with open(r'C:\Users\Zhen Wu\Desktop\CS2750\hw7\HW-7-Data\bbn', 'rb') as f:
    unpickler = MyCustomUnpickler(f)
    obj = unpickler.load()

# get the ml estimate matrix
# the matrix is organized in the following manner:
'''                 pneumonia_false             pneumonia_true
fever_false               0.4                       0.1
fever_true                0.6                       0.9
paleness_false            0.5                       0.3
paleness_true             0.5                       0.7
cough_false               0.9                       0.1
cough_true                0.1                       0.9
HighWBC_false             0.5                       0.2
HighWBC_true              0.5                       0.8
'''
matrix = obj.ml

# method for computing the probability
def compute_probability(symptom_list):
    # the probabilities multiplier for each of the attributes when pneumonia is true (the nominator of the formula)
    multiplier_p_t = []
    # the probabilities multiplier for each of the attributes when pneumonia is false
    multiplier_p_f = []
    
    for i in range(len(symptom_list)):
        # if the value for this attribute is unknown (can be t/f), after summing up it's 1
        if symptom_list[i]==-1:
            multiplier_p_t.append(1)
            multiplier_p_f.append(1)
        # if the value is either t/f, get the ml estimates
        else:
            multiplier_p_t.append(matrix[i*2+symptom_list[i]][1])
            multiplier_p_f.append(matrix[i*2+symptom_list[i]][0])

    nominator = 0.02*np.prod(multiplier_p_t)
    denominator = 0.02*np.prod(multiplier_p_t) + 0.98*np.prod(multiplier_p_f)
    return nominator/denominator



# read example.csv
data = pd.read_csv(r'C:\Users\Zhen Wu\Desktop\CS2750\hw7\HW-7-Data\Example.csv').values
for i in range(len(data)):
    symptom_list = data[i,:]
    print(compute_probability(symptom_list))
