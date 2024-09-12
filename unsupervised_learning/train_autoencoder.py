#INTERNAL
from models.autoencoder import build_autoencoder
from models.models_UL import find_best_autoencoder_kmeans_config, visualize_clusters_tsne
from dataloader.dataloader import DataLoader
#EXTERNAL
import warnings
warnings.filterwarnings('ignore')


# UNSUPERVISED LEARNING WITH AUTOENCODER
def run_train_model():

    print('\nLoading data...')
    x, y= DataLoader().load()
    
    print('\nPre-processing data...')
    x_scaled = DataLoader.pre_process_data(x)

    latent_layer_sizes = [128, 64, 32, 16, 8, 4]
    k_values = [2,3,4,5,6,7,8,9]

    user_choice = input("Do you want to find the best configuration for clustering? (yes/no): ").lower()

    if user_choice == 'yes':
        best_config, latent_representations = find_best_autoencoder_kmeans_config(x_scaled, latent_layer_sizes, k_values)
        print("Best Configuration:", best_config)
    else:
        # Use default configuration
        best_config = {'latent_dim': 4, 'k': 2}  

        latent_dim = best_config['latent_dim']
        input_dim = x_scaled.shape[1]
        print('inout dim:', input_dim)
        autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        autoencoder.fit(x_scaled, x_scaled, epochs=100, batch_size=8, verbose=1)
        latent_representations = encoder.predict(x_scaled)

    visualize_clusters_tsne(latent_representations, y, best_config['k'],
                        f'Clusters for Latent Dimension {best_config["latent_dim"]} and k={best_config["k"]}')


def run():    
    run_train_model()

if __name__ == '__main__':
    run()

