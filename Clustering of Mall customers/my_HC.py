import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing mall dataset with pandas
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#Using the dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#fitting hierarchical clustering to mall dataset
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

#Visualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100, c='Red',label='Clusster1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100, c='blue',label='Clusster2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100, c='green',label='Clusster3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100, c='cyan',label='Clusster4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100, c='magenta',label='Clusster5')
plt.title('Clusster of clients')
plt.xlabel('Annual income(k$)')
plt.ylabel('Spending score(1-100)')
plt.legend()
plt.show()