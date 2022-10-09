import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

data = pd.read_csv('clustering_data.csv').values
# fit the agglomerative model
model = AgglomerativeClustering(n_clusters=3, linkage='complete').fit(data)

# print out the size of each group
print('cluster size of label 0: ',sum(model.labels_==0))
print('cluster size of label 1: ',sum(model.labels_==1))
print('cluster size of label 2: ',sum(model.labels_==2))
# print('cluster size of label 3: ',sum(model.labels_==3))

plt.figure()
# plot the data points
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))
plt.show()