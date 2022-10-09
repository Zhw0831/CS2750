import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler  

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# load the dataset and split into train and test
dataframe1 = pd.read_csv('pima_train.csv')
array1 = dataframe1.values
X_train = array1[:,0:8]
Y_train= array1[:,8]

dataframe2 = pd.read_csv('pima_test.csv')
array2 = dataframe2.values
X_test = array2[:,0:8]
Y_test= array2[:,8]
targets=['class 0', 'class 1']

# normalize the data
scaler = StandardScaler()
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

# apply the svm 
clf = MLPClassifier(random_state=2,hidden_layer_sizes=(4,3),activation='logistic')
# fit the model
clf.fit(X_train, Y_train)
# get the predictions
Y_pred = clf.predict(X_test)


# evaluate the metrics

# Test Accuracy
print("Test score: {:.2f}".format(clf.score(X=X_test, y=Y_test)))

#Test Confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
print(conf_matrix)
# extract components of the confusion matrix
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()


# calculate the misclassification error, SENS (recall),SPEC, PPV (precision), NPV on the test data
mis_err_test = (fp+fn)/(tn+fp+fn+tp)
sens_test = tp/(tp+fn)
spec_test = tn/(tn+fp)
ppv_test = tp/(tp+fp)
npv_test = tn/(tn+fn)

print('misclassification error for test data:\n',mis_err_test)
print('SENS for test data:\n',sens_test)
print('SPEC for test data:\n',spec_test)
print('PPV for test data:\n',ppv_test)
print('NPV for test data:\n',npv_test)


# predict the probability for class 1 (not just class label)
probs_mlp = clf.predict_proba(X_test)

# calculate AUROC
Auroc_score=roc_auc_score(Y_test, probs_mlp[:,1])
print("AUROC score: {:.2f}".format(Auroc_score))


# Draw the ROC curve
plt.figure(1)
# ROC curve components
fpr, tpr, thresholdsROC = roc_curve(Y_test, probs_mlp[:,1])
#plot
plt.plot(fpr,tpr)
plt.title("ROC curve")
plt.xlabel("1-SPEC")
plt.ylabel("SENS")
plt.show()

auprc = average_precision_score(Y_test, probs_mlp[:,1])
print("AUPRC score: {:.2f}".format(auprc))

# Draw the PR curve
plt.figure(2)
# Components of the Precision recall curve
precision, recall, thresholdsPR = precision_recall_curve(Y_test, probs_mlp[:,1])
# plot
plt.plot(recall,precision)
plt.title("PR curve")
plt.xlabel("SENS (Recall)")
plt.ylabel("PPV (Precision)")
plt.show()




# get the predictions
Y_pred = clf.predict(X_train)


# evaluate the metrics

# Train Accuracy
print("Train score: {:.2f}".format(clf.score(X=X_train, y=Y_train)))

#Train Confusion matrix
conf_matrix = confusion_matrix(Y_train, Y_pred)
print(conf_matrix)
# extract components of the confusion matrix
tn, fp, fn, tp = confusion_matrix(Y_train, Y_pred).ravel()


# calculate the misclassification error, SENS (recall),SPEC, PPV (precision), NPV on the train data
mis_err_test = (fp+fn)/(tn+fp+fn+tp)
sens_test = tp/(tp+fn)
spec_test = tn/(tn+fp)
ppv_test = tp/(tp+fp)
npv_test = tn/(tn+fn)

print('misclassification error for train data:\n',mis_err_test)
print('SENS for train data:\n',sens_test)
print('SPEC for train data:\n',spec_test)
print('PPV for train data:\n',ppv_test)
print('NPV for train data:\n',npv_test)


# predict the probability for class 1 (not just class label)
probs_mlp = clf.predict_proba(X_train)

# calculate AUROC
Auroc_score=roc_auc_score(Y_train, probs_mlp[:,1])
print("AUROC score: {:.2f}".format(Auroc_score))


# Draw the ROC curve
plt.figure(1)
# ROC curve components
fpr, tpr, thresholdsROC = roc_curve(Y_train, probs_mlp[:,1])
#plot
plt.plot(fpr,tpr)
plt.title("ROC curve")
plt.xlabel("1-SPEC")
plt.ylabel("SENS")
plt.show()

auprc = average_precision_score(Y_train, probs_mlp[:,1])
print("AUPRC score: {:.2f}".format(auprc))

# Draw the PR curve
plt.figure(2)
# Components of the Precision recall curve
precision, recall, thresholdsPR = precision_recall_curve(Y_train, probs_mlp[:,1])
# plot
plt.plot(recall,precision)
plt.title("PR curve")
plt.xlabel("SENS (Recall)")
plt.ylabel("PPV (Precision)")
plt.show()



