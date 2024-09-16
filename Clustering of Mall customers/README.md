
# Clustering Algorithms Project

This project implements several clustering algorithms on the **Mall_Customers** dataset to segment customers based on their annual income and spending score. Each clustering algorithm is implemented in a separate Python script, and this README provides details about the project's structure and how to use it.

## Project Structure

The repository contains the following files:

- `mykmeans.py` - K-Means clustering implementation
- `my_HC.py` - Hierarchical Clustering (HC) implementation

### Dataset

The dataset used is `Mall_Customers.csv`, which contains information about customers from a mall. The features used in clustering are:

- **Annual Income** (in $1000s)
- **Spending Score** (on a scale of 1-100)

### Installation and Requirements

To run the scripts, you'll need to install the following Python libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Usage

Each script follows the same structure:

1. **Importing Libraries**: Libraries like NumPy, Pandas, Matplotlib, and Scikit-learn are imported.
2. **Loading the Dataset**: The dataset `Mall_Customers.csv` is loaded into the script.
3. **Clustering the Data**:
   - For K-Means, the elbow method is used to determine the optimal number of clusters.
   - For Hierarchical Clustering, dendrograms are used to visualize the clustering process and select the optimal number of clusters.
4. **Applying the Clustering Algorithm**:
   - For K-Means, the dataset is divided into clusters, and centroids are identified.
   - For Hierarchical Clustering, an agglomerative approach is used to form clusters.
5. **Visualization**: Clusters are visualized using scatter plots, with different colors representing different customer segments.

### Example: K-Means Clustering

The `mykmeans.py` file contains the implementation of K-Means clustering.

**Steps**:

1. **Using the Elbow Method**: The within-cluster sum of squares (WCSS) is calculated for different numbers of clusters, and the elbow method is used to determine the optimal number of clusters.

```python
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

2. **Clustering**: The optimal number of clusters (in this case, 5) is used to fit the K-Means algorithm to the dataset and predict the clusters.

```python
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
```

3. **Visualization**: The results are visualized with a scatter plot, where different clusters are represented with distinct colors, and the cluster centroids are highlighted.

```python
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

### Visualizations

Each clustering method includes visualizations of the clusters formed. For example, K-Means clusters customers based on their annual income and spending score, with each cluster represented in a different color and the centroids highlighted in yellow.

### License

This project is open-source and available under the MIT License.
