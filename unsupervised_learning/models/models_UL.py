#INTERNAL
from matplotlib.colors import ListedColormap
from models.autoencoder import build_autoencoder
#EXTERNAL   
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def find_best_autoencoder_kmeans_config(training_data, latent_layer_sizes, k_values, epochs=100, batch_size=8, num_runs=5):
    best_mean_score = float('-inf')
    best_config = None
    best_latent_representations = None

    for latent_dim in latent_layer_sizes:
        input_dim = training_data.shape[1]
        autoencoder, encoder = build_autoencoder(input_dim, latent_dim)

        print(f"Training autoencoder with latent dimension {latent_dim}...")

        # Store silhouette scores and latent representations for each run
        run_scores = []
        run_latent_representations = []

        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}...")

            autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
            autoencoder.fit(training_data, training_data, epochs=epochs, batch_size=batch_size, verbose=1)
            latent_representations = encoder.predict(training_data)

            run_latent_representations.append(latent_representations)

            run_silhouette_scores = []

            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42)
                clusters = kmeans.fit_predict(latent_representations)
                score = silhouette_score(latent_representations, clusters)
                print(f"Latent Size: {latent_dim}, k: {k}, Silhouette Score: {score}")
                run_silhouette_scores.append(score)

            run_scores.append(run_silhouette_scores)

        # Calculate mean silhouette score for the current configuration
        mean_scores = np.mean(run_scores, axis=0)

        # Update best_config and best_latent_representations if a configuration with a higher mean score is found
        for k_idx, k in enumerate(k_values):
            if mean_scores[k_idx] > best_mean_score:
                best_mean_score = mean_scores[k_idx]
                best_config = {'latent_dim': latent_dim, 'k': k, 'mean_score': best_mean_score}
                best_latent_representations = run_latent_representations[np.argmax(mean_scores)]

    return best_config, best_latent_representations

def visualize_clusters_tsne(latent_representations, true_labels, k, title):
    
    if latent_representations.shape[1] == 1:
        # If only 1 latent unit, use it directly for visualization
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_representations, np.zeros_like(latent_representations), c=true_labels, cmap='viridis', marker='o', s=50, alpha=0.7)
        plt.title(title)
        plt.xlabel('Latent Unit 1')
        plt.ylabel('Dummy Y-axis')
        plt.colorbar(label='True Labels')
        plt.show()
        return
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_representations)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(latent_representations)
    label_mapping = {1: 'Schizophrenia', 2: 'Bipolar Disorder', 3: 'Psychotic Affective Type', 4:'Other Primary Psychotic Disorder', 5: 'Substance Induced Psychotic Disorder', 6:'Secondary Psychotic Syndrome', 7: 'Schizoaffective disorder'}
    descriptive_true_labels = [label_mapping[label] for label in true_labels]
    #plt.figure(figsize=(8,6))
    custom_cmap = create_custom_cmap(7)

    # Plot points for KMeans clusters
    for cluster_label in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_label)
        cluster_colors=['#831F3D','#56C5D0']
        plt.scatter(latent_2d[cluster_indices, 0], latent_2d[cluster_indices, 1],color=cluster_colors[cluster_label], alpha=1, s=200, label=f'Cluster {cluster_label+1}')

    # Plot points for true labels with descriptive labels using a loop
    for label in np.unique(descriptive_true_labels):
        indices = np.where(np.array(descriptive_true_labels) == label)
        label_numeric = np.where(np.unique(descriptive_true_labels) == label)[0][0]
        label_count = len(indices[0])
        plt.scatter(latent_2d[indices, 0], latent_2d[indices, 1], color=custom_cmap(label_numeric), marker='x', s=50, label=f'{label} ({label_count} samples)')

    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.subplots_adjust(right=0.65)  # Adjust the layout to prevent overlapping

    plt.show()

def create_custom_cmap(num_classes):
    code_colors = ['#ff1791', '#ff8700', '#ffd300', '#deff0a', '#a1ff0a', '#0aff99', '#0aefff', '#147df5', '#580aff', '#be0aff']
    assert num_classes <= len(code_colors), "Not enough code colors defined for the specified number of classes."
    cmap = ListedColormap(code_colors[:num_classes])
    return cmap

def visualize_clusters(pca_2d, clusters, true_labels, title):
    label_mapping = {1: 'Schizophrenia', 2: 'Bipolar Disorder', 3: 'Psychotic Affective Type', 4:'Other Primary Psychotic Disorder', 5: 'Substance Induced Psychotic Disorder', 6:'Secondary Psychotic Syndrome', 7: 'Schizoaffective disorder'}
    descriptive_true_labels = [label_mapping[label] for label in true_labels]
    
    # Create a custom colormap for true labels
    custom_cmap = create_custom_cmap(7)  # Assuming you have 7 unique true labels
    
    # Plot points for KMeans clusters
    for cluster_label in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_label)
        cluster_colors=['#831F3D','#56C5D0','#BD7000','#008A00', '#FA6800']
        plt.scatter(pca_2d[cluster_indices, 0], pca_2d[cluster_indices, 1], color=cluster_colors[cluster_label], alpha=1, s=200, label=f'Cluster {cluster_label+1}')

    # Plot points for true labels with descriptive labels using a loop
    for label in np.unique(descriptive_true_labels):
        indices = np.where(np.array(descriptive_true_labels) == label)
        label_numeric = np.where(np.unique(descriptive_true_labels) == label)[0][0]
        label_count = len(indices[0])
        plt.scatter(pca_2d[indices, 0], pca_2d[indices, 1], color=custom_cmap(label_numeric), marker='x', s=50, label=f'{label} ({label_count} samples)')

    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Legend', loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.subplots_adjust(right=0.65)  # Adjust the layout to prevent overlapping

    plt.show()

def find_best_k(X, max_k):
    best_k = 2
    best_score = -1
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        
        if silhouette_avg > best_score:
            best_k = k
            best_score = silhouette_avg
    
    return best_k, best_score