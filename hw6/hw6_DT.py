import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler  

### ML Classification models
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_recall_curve


dataframe1 = pd.read_csv('pima_train.csv')
array1 = dataframe1.values
X_train = array1[:,0:8]
Y_train= array1[:,8]

dataframe2 = pd.read_csv('pima_test.csv')
array2 = dataframe2.values
X_test = array2[:,0:8]
Y_test= array2[:,8]
targets=['class 0', 'class 1']


# normalize the X components of the data
scaler = StandardScaler()
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


clf = DecisionTreeClassifier(random_state=2)
#clf = DecisionTreeClassifier(random_state=2,max_depth=4)
#clf = DecisionTreeClassifier(random_state=2,min_samples_leaf=5)


print('Decision Tree Classifier')

# train the DT model
clf.fit(X_train, Y_train)

### Train datat
print("***** Train data stats *****")
prediction_LR = clf.predict(X_train)
# Test Accuracy
print("Train score: {:.2f}".format(clf.score(X_train, Y_train)))
#Test Confusion matrix
conf_matrix = confusion_matrix(Y_train, prediction_LR)
print(conf_matrix)

# make class prediction on test data 
print("***** Test data stats *****")

# make class prediction for the LogReg model
prediction_LR = clf.predict(X_test)

# Test Accuracy
print("Test score: {:.2f}".format(clf.score(X_test, Y_test)))

#Test Confusion matrix
conf_matrix = confusion_matrix(Y_test, prediction_LR)
print(conf_matrix)
# extract components of the confusion matrix
tn, fp, fn, tp = confusion_matrix(Y_test, prediction_LR).ravel()

# predict the probability for class 1 (not just class label)
probs_LR=clf.predict_proba(X_test)


# calculate AUROC
Auroc_score=roc_auc_score(Y_test, probs_LR[:,1])
print("AUROC score: {:.2f}".format(Auroc_score))


# Draw the ROC curve
plt.figure(1)
# ROC curve components
fpr, tpr, thresholdsROC = roc_curve(Y_test, probs_LR[:,1])
#plot
plt.plot(fpr,tpr)
plt.title("ROC curve")
plt.xlabel("1-SPEC")
plt.ylabel("SENS")
plt.show

# Draw the PR curve
plt.figure(2)
# Components of the Precision recall curve
precision, recall, thresholdsPR = precision_recall_curve(Y_test, probs_LR[:,1])
# plot
plt.plot(recall,precision)
plt.title("PR curve")
plt.xlabel("SENS (Recall)")
plt.ylabel("PPV (Precision)")
plt.show

# calculate the number of nodes and plot the tree
n_nodes = clf.tree_.node_count
print("number of nodes:", n_nodes)
tree.plot_tree(clf)
plt.show()