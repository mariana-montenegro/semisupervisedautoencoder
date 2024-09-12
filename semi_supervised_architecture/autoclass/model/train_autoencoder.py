#Internal
from model.autoencoder import Autoencoder

# External
import tensorflow as tf
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import time
import numpy as np
import os
import pandas as pd
import seaborn as sns


# Set random seeds for reproducibility
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class TrainAutoencoder:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-04)   
        self.counter_c = 0  # Counter for 'C' step plots
        
    def combine_loss(self, x_true, x_pred_autoencoder, y_true, y_pred_classifier):
        """ 
        Computes the combine error loss
        Args
        ----------
            x_true (tensor): True values
            x_pred_autoencoder (tensor): Model predictions from the autoencoder
            y_true (tensor): True values
            x_pred_classifier (tensor): Model predictions from the autoencoder + dense classification layer
        Returns
        -------
            Combine loss between the output and the input
        """
        loss_reconstruction = self.mse_loss(x_true, x_pred_autoencoder) #between the data x
        loss_classification = self.bce_loss(y_true, y_pred_classifier) #between the labels y

        factor_classification = 0.5
        loss_combined = loss_reconstruction*(1-factor_classification) + loss_classification*factor_classification
        
        return loss_combined
        
    def train_step(self, x_target, y_target):
        """ 
        Performs the learning step of the training process
        Args
        ----------
            source_input

        Returns
        -------
            loss
        """
        with tf.GradientTape() as tape:
            x_pred_autoencoder, _ , y_pred_classifier = self.autoencoder(x_target)
            loss = self.combine_loss(x_target, x_pred_autoencoder,y_target, y_pred_classifier)

        variables = self.autoencoder.trainable_variables 
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss 
    

    def create_colors(self, kk_classes, ku_classes):
        yellows = sns.color_palette('Greens_d', len(kk_classes))
        purples = sns.color_palette('rocket', len(ku_classes))
        
        kk_color_mapping = {kk: yellows[i] for i, kk in enumerate(kk_classes)}
        ku_color_mapping = {ku: purples[i] for i, ku in enumerate(ku_classes)}

        color_mapping = {**kk_color_mapping, **ku_color_mapping}

        return color_mapping

    def plot_latent_space(self, data, true_labels, pid, step=0, show=False):
        # Directory for saving plots
        path = "./LatentSpacePlots"
        if not os.path.exists(path):
            os.mkdir(path)

        sns.set_theme()
        sns.set_context("talk") #dict, or one of {paper, notebook, talk, poster}

        kk_classes = self.FLAGS.KK_classes_name
        ku_classes = self.FLAGS.KU_classes_name

        # Get the latent vectors and true labels
        self.autoencoder.trainable = False  # Set model to inference mode
        latent_vectors = []
        binary_labels = []
        for batch, (x_batch, y_batch) in enumerate(data):
            _, latent_vector, _ = self.autoencoder(x_batch, training=False)
            latent_vectors.extend(latent_vector.numpy())
            binarized_y_batch = np.max(y_batch, axis=1)
            binary_labels.extend(binarized_y_batch)
            
        latent_vectors = np.array(latent_vectors)

        true_label_mapping = {1: 'Schizophrenia', 2: 'Bipolar Disorder', 3: 'Psychotic Affective Type', 4:'Other Primary Psychotic Disorder', 5: 'Substance Induced Psychotic Disorder', 6:'Secondary Psychotic Syndrome', 7: 'Schizoaffective disorder'}
        descriptive_true_labels = [true_label_mapping[label] for label in true_labels]
        custom_colors = self.create_colors(kk_classes, ku_classes)

        # Determine a suitable perplexity for dummy data use
        n_samples = latent_vectors.shape[0]
        perplexity = min(30, n_samples - 1)  # Perplexity must be less than n_samples

        if perplexity < 2:  # Ensure perplexity is at least 2 for t-SNE to function properly
            perplexity = 2

        # Perform t-SNE transformation
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        latent_vectors_tsne = tsne.fit_transform(latent_vectors)

        # Create a DataFrame for plotting
        df = pd.DataFrame(latent_vectors_tsne, columns=['Latent Dimension 1', 'Latent Dimension 2'])
        df['Label'] = descriptive_true_labels

        # Create the scatter plot and legend handles simultaneously
        handles = []
        labels = []
        for label in np.unique(descriptive_true_labels):
            indices = np.where(np.array(descriptive_true_labels) == label)
            label_count = len(indices[0])
            label_df = df.loc[indices]
            ax = sns.scatterplot(data=label_df, x='Latent Dimension 1', y='Latent Dimension 2', color=custom_colors[label], label=f'{label} ({label_count} samples)', s=50)
            ax.legend_.remove()  # Remove legend for each iteration
            handles, labels = ax.get_legend_handles_labels()
        
        one_sample_classes = [class_label for class_label in np.unique(descriptive_true_labels) if descriptive_true_labels.count(class_label) == 1]
        if len(one_sample_classes) == 0:
            # Exclude classes with only one sample from the order
            order = [i for i in [4, 0, 2, 1, 5, 3] if labels[i].split('(')[0] not in one_sample_classes]
        else:
            order = [4, 0, 2, 1, 6, 5, 3]

        ax.legend([handles[i] for i in order], [labels[i] for i in order], title='Legend', loc='upper left', bbox_to_anchor=(1.02, 1))

        #ax.legend(handles, labels, title='Legend', loc='upper left', bbox_to_anchor=(1.02, 1)) #dummy data

        # Annotate points with ID names
        for x, y, pid_name, true_labels in zip(latent_vectors_tsne[:, 0], latent_vectors_tsne[:, 1], pid, true_labels):
            plt.annotate(pid_name, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

        if isinstance(step, int):
            ax.set_title(f"Latent Space Visualization Epoch {int(step):03d}")
        elif step == 'A':
            ax.set_title("Latent Space Visualization for All Data")
        elif step == 'C':
            ax.set_title("Latent Space Visualization for Classification")

        ax.set_xlabel('Latent Dimension 1', labelpad=10)
        ax.set_ylabel('Latent Dimension 2', labelpad=10)

        # Save the plot
        if isinstance(step, int):
            plt.savefig(f"{path}/Step_{step:03d}.png", bbox_inches="tight")
        elif step == 'A':
            plt.savefig(f"{path}/Step_AllData.png", bbox_inches="tight")
        elif step == 'C':
            self.counter_c += 1  # Increment the counter for 'C' step plots
            plt.savefig(f"{path}/Step_Classification_{self.counter_c:03d}.png", bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()  # Close the plot to free memory
            self.autoencoder.trainable = True  # Set model back to training mode

    def train_autoencoder(self, data, true_labels, pid): #data_train
        """ Train and save the autoencoder
        Args: 
            data_autoencoder (list): List with pre-processed target set
            FLAGS (argparse): Implementation parameters
        Returns
        -------
            Autoencoder model saved
        """ 
        self.autoencoder = Autoencoder(self.FLAGS) 
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.bce_loss = tf.keras.losses.BinaryCrossentropy()

        #### LEAVE ONE PATIENT OUT CROSS VALIDATION #####

        # Training loop
        last_loss = {'epoch':0,'value':1000}
        for epoch in range(self.FLAGS.n_epochs):
            print(f'Epoch {epoch+1}/{self.FLAGS.n_epochs}')
            start = time.time()
            loss_epoch = []
            for batch, (x_train, y_train) in enumerate(data):
                loss_batch = self.train_step(x_train, y_train)
                loss_epoch.append(loss_batch)
                if batch == len(data)-1:
                    print(f'{batch+1}/{len(data)} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f}')
            if (last_loss['value'] - np.mean(loss_epoch)) >= self.FLAGS.min_delta:
                last_loss['value'] = np.mean(loss_epoch)
                last_loss['epoch'] = epoch+1 
                self.autoencoder.save_weights('m.h5')                            
            if ((epoch+1) - last_loss['epoch']) >= self.FLAGS.patience:
                break 
            self.plot_latent_space(data, true_labels, pid, step=epoch+1, show=False)
            
        #print("Plotting autoencoder model architecture...")
        #plot_model(self.autoencoder, to_file='autoencoder_model.png', show_shapes=True, show_layer_names=True)
        #print('\nPlotting final latent space...')
        #self.plot_latent_space(data, true_labels, pid, step=self.FLAGS.n_epochs, show=True)

    def create_gif(self):
        """ Create a GIF from saved images """
        images = []
        path = "./LatentSpacePlots"
        for filename in sorted(os.listdir(path)):
            if filename.endswith(".png"):
                images.append(Image.open(os.path.join(path, filename)))
        images[0].save('latent_space.gif', save_all=True, append_images=images[1:], loop=0, duration=400)

    def apply_model_and_plot_latent(self, data, true_labels, pid):
        print('\nPlotting latent space for all data...')
        self.plot_latent_space(data, true_labels, pid, step='A', show=False)



