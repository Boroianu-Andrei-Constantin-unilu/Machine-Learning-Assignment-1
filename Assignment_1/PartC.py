import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import *
from sklearn.metrics import *

# Reading the file is necessary for data analysis
iris_data = pd.read_csv('iris.csv')

# Task C.I.2
# Introduce features of the dataset
iris_features = iris_data.drop(columns=['species'])

# Apply proper k-means clustering, with k = 3 (species in dataset)
kmeans = KMeans(n_clusters=3, random_state=42)
iris_data['cluster'] = kmeans.fit_predict(iris_features)

# Pair the plot with the corrected data, excluding the species column
sns.pairplot(iris_data, hue='cluster', palette='Set1', diag_kind='kde')
plt.suptitle('K-Means Clustering on Iris Dataset', y=1.02)
plt.show()

# Task C.I.3
# Compute the silhouette score
silhouette_avg = silhouette_score(iris_features, iris_data['cluster'])
print(silhouette_avg)

# Task C.I.4
# Map the cluster numbers to the species for comparison
species_to_number = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
iris_data['species_num'] = iris_data['species'].map(species_to_number)

# Confusion matrix comparing the k-means clustering to the actual species
conf_matrix = confusion_matrix(iris_data['species_num'], iris_data['cluster'])

# Task C.I.5
# Reading the file is necessary for data analysis
unknown_species_data = pd.read_csv('unknown_species.csv')

# Clean the data, by removing non-numerical columns
unknown_species_cleaned_data = unknown_species_data.drop(columns=['id', 'species'])

# Apply the k-means clustering model, in order to predict and display the clusters for the unknown species
predicted_clusters = kmeans.predict(unknown_species_cleaned_data)
print(conf_matrix, predicted_clusters)

# Task C.I.6
# Implementing k-means from scratch
def initialitze_centroids(X, k):
    # Randomly initialize k centroids from the data points
    np.random.seed(42)
    random_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[random_indices]
    return centroids

def assign_clusters(X, centroids):
    # Assign each data point to the nearest centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    # Compute the new centroids, as the means of the data points in each cluster
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans_scratch(X, k, max_iters=100, tolerance=1e-4):
    # Run the k-means algorithm from scratch
    # Step A: Initialize centroids
    centroids = initialitze_centroids(X, k)
    
    for i in range(max_iters):
        # Step B: Assign clusters based on closest centroids
        labels = assign_clusters(X, centroids)
        
        # Step C: Update centroids, based on the clusters
        new_centroids = update_centroids(X, labels, k)
        
        # Step D: Check for convergence (if centroids undergo no significant change)
        if np.linalg.norm(new_centroids - centroids) < tolerance:
            break
        
        centroids = new_centroids
        
    return labels, centroids

# Run the k-means algorithm from scratch
k = 3 # Number of species in the dataset
X_iris = iris_features.to_numpy()
scratch_labels, scratch_centroids = kmeans_scratch(X_iris, k)
print(scratch_labels[:10], scratch_centroids)

# Task C.II.2
# Reading the file is necessary for data analysis
ulu_data = pd.read_csv('ulu.csv')
ulu_data2 = pd.read_csv('ulu.csv')
ulu_data3 = pd.read_csv('ulu.csv')

# Apply DBSCAN clustering to the data
dbscan = DBSCAN(eps=1.5, min_samples=5) # Standard hyperparameters
ulu_data['dbscan_cluster'] = dbscan.fit_predict(ulu_data)

# Check the number of DBSCAN clusters identified
n_clusters_dbscan = len(set(ulu_data['dbscan_cluster'])) - (1 if -1 in ulu_data['dbscan_cluster'] else 0)
print(n_clusters_dbscan)

# Task C.II.3