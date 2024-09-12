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
    
    # Initialize empty lists to store silhouette scores
    silhouette_scores = []

    # Test each n_components value
    for n_components in n_components_range:
        print(f"\nTesting PCA with n_components = {n_components}...")
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x_scaled)
        
        # Initialize empty list to store silhouette scores for current n_components
        silhouette_scores_components = []
        
        # Test each n_clusters value
        for n_clusters in n_clusters_range:
            print(f"\nTesting K-means with n_clusters = {n_clusters}...")
            
            # Perform K-means clustering with the PCA-transformed data
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(x_pca)
            
            # Evaluate clustering performance (silhouette score)
            silhouette_avg = silhouette_score(x_pca, clusters)
            silhouette_scores_components.append(silhouette_avg)
            print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")
        
        # Append silhouette scores for current n_components to the main list
        silhouette_scores.append(silhouette_scores_components)

    # Convert silhouette_scores to numpy array for plotting
    silhouette_scores = np.array(silhouette_scores)

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(n_components_range)))
    for i, n_components in enumerate(n_components_range):
        plt.plot(n_clusters_range, silhouette_scores[i], marker='o', linestyle='-', color=colors[i], label=f'n_components = {n_components}')
    
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters for PCA with Different Components')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    test_pca_components()
