import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_recall_curve

from sklearn.ensemble import RandomForestClassifier

dataframe1 = pd.read_csv('pima_train.csv')
array1 = dataframe1.values
X_train = array1[:,0:8]
Y_train= array1[:,8]

dataframe2 = pd.read_csv('pima_test.csv')
array2 = dataframe2.values
X_test = array2[:,0:8]
Y_test= array2[:,8]
targets=['class 0', 'class 1']

# option 1 model
# clf = RandomForestClassifier(min_samples_leaf=5, n_estimators=30, random_state=0).fit(X_train, Y_train)
# option 2 model
# clf = RandomForestClassifier(max_depth=None, n_estimators=30, random_state=0).fit(X_train, Y_train)
# option 3 model
clf = RandomForestClassifier(max_depth=1, n_estimators=30, random_state=0).fit(X_train, Y_train)

print("***** Train data stats *****")
prediction_DT = clf.predict(X_train)
# Test Accuracy
print("Train score: {:.2f}".format(clf.score(X_train, Y_train)))
#Test Confusion matrix
conf_matrix = confusion_matrix(Y_train, prediction_DT)
print(conf_matrix)

# make class prediction on test data 
print("***** Test data stats *****")
prediction_DT = clf.predict(X_test)
# Test Accuracy
print("Test score: {:.2f}".format(clf.score(X_test, Y_test)))
#Test Confusion matrix
conf_matrix = confusion_matrix(Y_test, prediction_DT)
print(conf_matrix)
# extract components of the confusion matrix
tn, fp, fn, tp = confusion_matrix(Y_test, prediction_DT).ravel()

# predict probability for each class
probs_DT=clf.predict_proba(X_test)    # check how to get the projection here !!!!
# calculate AUROC
Auroc_score=roc_auc_score(Y_test, probs_DT[:,1])
print("AUROC score: {:.2f}".format(Auroc_score))

# Draw the ROC curve
plt.figure(1)
# ROC curve components
fpr, tpr, thresholds = roc_curve(Y_test, probs_DT[:,1])
#plot
plt.plot(fpr,tpr)
# Draw ROC curve
plt.title("ROC curve")
plt.xlabel("1-SPEC")
plt.ylabel("SENS")
plt.show()


# Draw the PR curve
plt.figure(2)
# Components of the Precision recall curve
precision, recall, thresholdsPR = precision_recall_curve(Y_test, probs_DT[:,1])
# plot
plt.plot(recall,precision)
plt.title("PR curve")
plt.xlabel("SENS (Recall)")
plt.ylabel("PPV (Precision)")
plt.show()
