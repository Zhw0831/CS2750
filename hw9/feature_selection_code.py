import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# read in the data
data = pd.read_csv('FeatureSelectionData.csv')
# split features and labels
features = data.iloc[:,:-1]
labels = data.iloc[:,-1].to_numpy()

# compute fisher score
def Fisher_score(x, y):
    # the lists to store the positive(1)/negative(0) data instances
    x_1 = []
    x_0 = []

    for i in range(len(x)):
        if(y[i]==1):
            x_1.append(x[i])
        elif(y[i]==0):
            x_0.append(x[i])

    # compute mean and std for both lists
    mean_1 = np.mean(x_1)
    mean_0 = np.mean(x_0)
    std_1 = np.std(x_1)
    std_0 = np.std(x_0)

    # compute fisher score
    score = ((mean_1-mean_0)**2)/(std_1**2 + std_0**2)
    return score

# compute AUROC
def AUROC_score(x,y):
    score = roc_auc_score(y,x)
    return score

# compute fisher score for each of the dimensions
fisher_score_list = []
auroc_score_list = []

for i in range(len(features.columns)):
    x = features.iloc[:,i].to_numpy()

    f_score = Fisher_score(x,labels)
    fisher_score_list.append(f_score)

    a_score = AUROC_score(x,labels)
    auroc_score_list.append(a_score)


# sort the list
# find the index so we know which score is which after sorting
f_ind = np.argsort(fisher_score_list)
fisher_score_list.sort(reverse=True)

a_ind = np.argsort(auroc_score_list)
auroc_score_list.sort(reverse=True)

top_20_fisher_ind = f_ind[::-1][:20]
top_20_fisher_score = fisher_score_list[0:20]

top_20_auroc_ind = a_ind[::-1][:20]
top_20_auroc_score = auroc_score_list[0:20]

print("fisher score top 20")
for i in range(20):
    rank = i + 1
    dim_label = top_20_fisher_ind[i] + 1
    dim_score = top_20_fisher_score[i]
    print(f"Top {rank}: label {dim_label}, score {dim_score}")


print("auroc score top 20")
for i in range(20):
    rank = i + 1
    dim_label = top_20_auroc_ind[i] + 1
    dim_score = top_20_auroc_score[i]
    print(f"Top {rank}: label {dim_label}, score {dim_score}")

