import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# Load the data and perform label encoding
df = pd.read_csv('Table.csv')
columns = ['Social Behavior', 'Diet', 'Habitat']

label_encoder = LabelEncoder()
for column in columns:
    df[column] = label_encoder.fit_transform(df[column])

df_numeric = df.drop(columns=['Animal'])

# Perform Standard Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Perform PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(data=df_pca, columns=['PC1', 'PC2'])  # Define column names for PCA components
print(df_pca)
# Plot PCA
plt.scatter(df_pca['PC1'], df_pca['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Plot')
plt.show()
# Proceed with clustering
cluster_range = range(2, 10)
wcss_values = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_pca[['PC1', 'PC2']])
    wcss_values.append(kmeans.inertia_)

plt.plot(cluster_range, wcss_values)
plt.xlabel('K')
plt.ylabel('wcss')
plt.title('Elbow Line')
plt.xticks(cluster_range)
plt.show()
# Proceed with clustering using fixed number of clusters (optimal_k = 3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(df_pca[['PC1', 'PC2']])
# Assuming df_pca is your DataFrame containing PCA components PC1 and PC2



cluster_labels = kmeans.labels_

# Assign cluster labels and map them to cluster names
df['Cluster'] = cluster_labels

cluster_labels_mapping = {
    0: 'Carnivore Animals',
    1: 'Social Herbivore Animals',
    2: 'Endangered species',
    # Add more mappings as needed
}
df['Cluster Name'] = df['Cluster'].map(cluster_labels_mapping)

# Print the cluster assignments
print(df[['Animal', 'Cluster Name']])

# Plot clusters with cluster names
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=cluster_labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', marker='x')
plt.title('Clusters with Cluster Names')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
