import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from dataloader.dataloader import DataLoader
from models.models_UL import visualize_clusters, find_best_k
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def test_pca_components():
    # Load your data
    print('\nLoading data...')
    x, y = DataLoader().load()
    
    # Pre-process your data if necessary
    print('\nPre-processing data...')
    x_scaled = DataLoader.pre_process_data(x)

    # Define a range of n_components to test
    n_components_range = [2, 3, 5, 10]  # Example range, adjust as needed
    
    # Define a range of n_clusters to test
    n_clusters_range = [2, 3, 4, 5]  # Example range, adjust as needed

    # Test each n_components value
    for n_components in n_components_range:
        print(f"\nTesting PCA with n_components = {n_components}...")
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x_scaled)
        
        # Test each n_clusters value
        for n_clusters in n_clusters_range:
            print(f"\nTesting K-means with n_clusters = {n_clusters}...")
            
            # Perform K-means clustering with the PCA-transformed data
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(x_pca)
            
            # Evaluate clustering performance (example: silhouette score)
            silhouette_avg = silhouette_score(x_pca, clusters)
            print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")
            
            # Visualize clusters if desired
            visualize_clusters(x_pca, clusters, y, f'Clusters for PCA with {n_components} components and k={n_clusters} clusters')

        
        # Optionally, plot explained variance ratio
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var_ratio = np.cumsum(explained_var_ratio)
        plt.figure()
        plt.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    test_pca_components()
