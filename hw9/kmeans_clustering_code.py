import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('clustering_data.csv').values
# fit the k-means model
# kmeans = KMeans(n_clusters=3).fit(data)
kmeans = KMeans(n_clusters=4, init='random').fit(data)
# print out the center coordinates
print(kmeans.cluster_centers_)
# print out the size of each group
print('cluster size of label 0: ',sum(kmeans.labels_==0))
print('cluster size of label 1: ',sum(kmeans.labels_==1))
print('cluster size of label 2: ',sum(kmeans.labels_==2))
print('cluster size of label 3: ',sum(kmeans.labels_==3))
# print out the sum of squared center-point distances
print('sum of squared center-point distances: ',kmeans.inertia_)

plt.figure()
# plot the data points
plt.scatter(data[:,0], data[:,1], c=kmeans.labels_.astype(float))
# plot the cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
plt.show()