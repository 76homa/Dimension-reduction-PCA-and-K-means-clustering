from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# Sample data (replace this with your dataset)
file_path = "C:/Users/homa.behmardi/Downloads/eachbandbehaviour.xlsx"
data = pd.read_excel(file_path, sheet_name="Sheet1")

# Assuming 'SITE' is the categorical column in your DataFrame
# Perform one-hot encoding
encoded_columns = pd.get_dummies(data['sector'], prefix='sector')
encoded_data = pd.concat([data[['Payload', 'Throughput']], encoded_columns], axis=1)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=2)  # Set the number of components you want to keep
pca_features = pca.fit_transform(encoded_data[['Payload', 'Throughput']])

# Determine the optimal number of clusters (k) using the elbow method
inertia_values = []
possible_k_values = range(1, 11)  # Trying k values from 1 to 10

for k in possible_k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pca_features)
    inertia_values.append(kmeans.inertia_)

# Choose the best k based on the elbow curve (you need to visually inspect it)
best_k = 10  # Change this value based on your observation

# Initialize KMeans object with the best k
kmeans = KMeans(n_clusters=best_k)

# Fit the model to the PCA-transformed data
kmeans.fit(pca_features)

# Get cluster assignments and cluster centers
cluster_labels = kmeans.labels_

# Add the cluster labels to the original DataFrame
data['Cluster'] = cluster_labels

# Visualize the results (for 2D data)
x = pca_features[:, 0]
y = pca_features[:, 1]
cluster_labels = data['Cluster']

# Create a scatter plot
plt.scatter(x, y, c=cluster_labels, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='black', s=100)  # Cluster centers
plt.xlabel('PCA Payload')
plt.ylabel('PCA Throughput')
plt.title('K-Means Clustering after PCA')
plt.show()

# Assuming 'data' is your DataFrame with the clustering results
cluster_1_data = data[data['Cluster'] == 1]
