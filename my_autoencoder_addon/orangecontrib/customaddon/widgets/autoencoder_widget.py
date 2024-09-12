from Orange.widgets import widget, gui, settings
from Orange.widgets.widget import Input
from PyQt5.QtWidgets import QLabel
from orangecontrib.customaddon.flags import argparser
from orangecontrib.customaddon.autoencoder import Autoencoder
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time

class OWAutoencoder(widget.OWWidget):
    name = "Autoencoder"
    description = "Train the autoencoder"
    icon = "widget_icons/Autoencoder.svg"
    want_main_area = False

    class Inputs:
        data_train = Input("Train Data", object)
        true_labels = Input("True Labels", list)

    def __init__(self):
        super().__init__()
        self.info_label = QLabel("Training autoencoder...")
        self.controlArea.layout().addWidget(self.info_label)
        self.data_train = None
        self.true_labels = None
        self.FLAGS = argparser()

    @Inputs.data_train
    def set_data(self, data_train):
        self.data_train = data_train
        self.check_data()

    @Inputs.true_labels
    def set_labels(self, true_labels):
        self.true_labels = true_labels
        self.check_data()

    def check_data(self):
        if self.data_train is not None and self.true_labels is not None:
            self.info_label.setText("Received training data and true labels.")
            self.train_autoencoder()

    def train_autoencoder(self):
        autoencoder = Autoencoder(self.FLAGS)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-04)
        mse_loss = tf.keras.losses.MeanSquaredError()
        bce_loss = tf.keras.losses.BinaryCrossentropy()
        last_loss = {'epoch': 0, 'value': 1000}
        for epoch in range(self.FLAGS.n_epochs):
            print(f'Epoch {epoch + 1}/{self.FLAGS.n_epochs}')
            start = time.time()
            loss_epoch = []
            for batch, (x_train, y_train) in enumerate(self.data_train):
                with tf.GradientTape() as tape:
                    x_pred_autoencoder, _, y_pred_classifier = autoencoder(x_train)
                    loss = mse_loss(x_train, x_pred_autoencoder) * (1 - self.FLAGS.autoencoder_weight) + bce_loss(y_train, y_pred_classifier) * self.FLAGS.classifier_weight
                gradients = tape.gradient(loss, autoencoder.trainable_variables)
                optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
                loss_epoch.append(loss)
                if batch == len(self.data_train) - 1:
                    print(f'{batch + 1}/{len(self.data_train)} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f}')
            if (last_loss['value'] - np.mean(loss_epoch)) >= self.FLAGS.min_delta:
                last_loss['value'] = np.mean(loss_epoch)
                last_loss['epoch'] = epoch + 1
                autoencoder.save_weights('autoencoder_weights.h5')
            if (epoch + 1 - last_loss['epoch']) >= self.FLAGS.patience:
                break

        self.plot_latent_space(autoencoder, self.data_train, self.true_labels, epoch + 1)

    def plot_latent_space(self, autoencoder, data, true_labels, step=0, show=True):
        sns.set_theme()
        sns.set_context("talk")
        latent_vectors = []
        for batch, (x_batch, y_batch) in enumerate(data):
            _, latent_vector, _ = autoencoder(x_batch, training=False)
            latent_vectors.extend(latent_vector.numpy())
        latent_vectors = np.array(latent_vectors)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        latent_vectors_tsne = tsne.fit_transform(latent_vectors)
        df = pd.DataFrame(latent_vectors_tsne, columns=['Latent Dimension 1', 'Latent Dimension 2'])
        true_label_mapping = {1: 'Schizophrenia', 2: 'Bipolar Disorder', 3: 'PAT', 4: 'Other Primary Psychotic Disorder', 5: 'Substance Induced Psychotic Disorder', 6: 'Secondary Psychotic Syndrome', 7: 'Schizoaffective disorder'}
        descriptive_true_labels = [true_label_mapping[label] for label in true_labels]
        df['Label'] = descriptive_true_labels
        custom_colors = sns.color_palette('hsv', len(np.unique(descriptive_true_labels)))
        ax = sns.scatterplot(data=df, x='Latent Dimension 1', y='Latent Dimension 2', hue='Label', palette=custom_colors)
        if show:
            plt.show()
