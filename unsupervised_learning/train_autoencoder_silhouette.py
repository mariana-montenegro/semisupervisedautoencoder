import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from models.autoencoder import build_autoencoder
from models.models_UL import find_best_autoencoder_kmeans_config, visualize_clusters_tsne
from dataloader.dataloader import DataLoader
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def run_train_model():
    print('\nLoading data...')
    x, y = DataLoader().load()
    
    print('\nPre-processing data...')
    x_scaled = DataLoader.pre_process_data(x)

    latent_layer_sizes = [128, 64, 32, 16, 8, 4]
    k_values = [2, 3, 4, 5, 6, 7, 8, 9]

    user_choice = input("Do you want to find the best configuration for clustering? (yes/no): ").lower()

    if user_choice == 'yes':
        best_config, latent_representations = find_best_autoencoder_kmeans_config(x_scaled, latent_layer_sizes, k_values)
        print("Best Configuration:", best_config)
    else:
        # Use default configuration
        best_config = {'latent_dim': 4, 'k': 2}  

        latent_dim = best_config['latent_dim']
        input_dim = x_scaled.shape[1]
        autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        autoencoder.fit(x_scaled, x_scaled, epochs=100, batch_size=8, verbose=1)
        latent_representations = encoder.predict(x_scaled)

    # Perform K-means clustering on the latent representations
    kmeans = KMeans(n_clusters=best_config['k'])
    clusters = kmeans.fit_predict(latent_representations)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(latent_representations, clusters)
    print(f"Silhouette Score for {best_config['k']} clusters: {silhouette_avg}")

    # Visualize clusters using t-SNE visualization
    visualize_clusters_tsne(latent_representations, y, best_config['k'],
                            f'Clusters for Latent Dimension {best_config["latent_dim"]} and k={best_config["k"]}')

    # Plot silhouette scores
    plot_silhouette_scores(latent_representations, clusters)

def plot_silhouette_scores(X, clusters):
    silhouette_scores = silhouette_samples(X, clusters)
    silhouette_avg = silhouette_score(X, clusters)

    fig, ax = plt.subplots(figsize=(8, 6))

    y_lower = 10
    for i in range(max(clusters) + 1):
        cluster_silhouette_values = silhouette_scores[clusters == i]
        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.tab10(float(i) / max(clusters))
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")

    plt.title(f"Silhouette Plot for Clustering Results (Average Score: {silhouette_avg:.2f})")
    plt.show()

def run():    
    run_train_model()

if __name__ == '__main__':
    run()
