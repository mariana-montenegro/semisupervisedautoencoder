import time
from Orange.widgets import widget, gui, settings
from Orange.widgets.widget import Input
from PyQt5.QtWidgets import QLabel
from orangecontrib.customaddon.flags import argparser
from orangecontrib.customaddon.autoencoder import Autoencoder
from orangecontrib.customaddon.binary_classifier import Binary_classifier
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import binarize
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

class OWClassifier(widget.OWWidget):
    name = "Binary Classifier"
    description = "Train and evaluate the binary classifier"
    icon = "widget_icons/Classifier.svg"
    want_main_area = False

    class Inputs:
        data_train = Input("Train Data", object)
        data_test = Input("Test Data", object)
        true_labels = Input("True Labels", list)

    def __init__(self):
        super().__init__()
        self.info_label = QLabel("Training binary classifier...")
        self.controlArea.layout().addWidget(self.info_label)
        self.data_train = None
        self.data_test = None
        self.true_labels = None
        self.FLAGS = argparser()

    @Inputs.data_train
    def set_train_data(self, data_train):
        self.data_train = data_train
        self.check_data()

    @Inputs.data_test
    def set_test_data(self, data_test):
        self.data_test = data_test
        self.check_data()

    @Inputs.true_labels
    def set_labels(self, true_labels):
        self.true_labels = true_labels
        self.check_data()

    def check_data(self):
        if self.data_train is not None and self.true_labels is not None and self.data_test is not None:
            self.info_label.setText("Received training and test data, and true labels.")
            self.train_classifier()

    def train_classifier(self):
        autoencoder = Autoencoder(self.FLAGS)
        classifier = Binary_classifier(self.FLAGS)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-04)
        bce_loss = tf.keras.losses.BinaryCrossentropy()

        autoencoder.load_weights('autoencoder_weights.h5')

        def loss_func(y_true, y_predicted):
            return bce_loss(y_true, y_predicted)

        @tf.function
        def train_step(latent, y_train):
            with tf.GradientTape() as tape:
                pred_classifier = classifier(latent)
                loss_classification = loss_func(y_train, pred_classifier)
            gradients = tape.gradient(loss_classification, classifier.trainable_variables)
            optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))
            return loss_classification, pred_classifier

        last_loss = {'epoch': 0, 'value': 1000}
        for epoch in range(self.FLAGS.n_epochs):
            print(f'\n\nEpoch {epoch + 1}/{self.FLAGS.n_epochs}')
            start = time.time()
            loss_epoch = []
            for batch, (x_train, y_train) in enumerate(self.data_train):
                _, latent, _ = autoencoder(x_train, training=False)
                loss_batch, predictions_batch = train_step(latent, y_train)
                loss_epoch.append(loss_batch)
                if batch == len(self.data_train) - 1:
                    print(f'{batch + 1}/{len(self.data_train)} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f}')
            if (last_loss['value'] - np.mean(loss_epoch)) >= self.FLAGS.min_delta:
                last_loss['value'] = np.mean(loss_epoch)
                last_loss['epoch'] = epoch + 1
            if (epoch + 1 - last_loss['epoch']) >= self.FLAGS.patience:
                break

        self.evaluate_classifier(autoencoder, classifier)

    def evaluate_classifier(self, autoencoder, classifier):
        print('\nPredicting on test set...')
        predictions = []
        true_labels = []
        for num, (x_test, y_test) in enumerate(self.data_test):
            _, latent_vector, _ = autoencoder(x_test, training=False)
            output_classifier = classifier(latent_vector, training=False)
            predicted_labels = binarize(output_classifier, threshold=0.5)
            predictions.append(predicted_labels)
            true_labels.append(y_test.numpy())

        predictions = np.concatenate(predictions)
        true_labels = np.concatenate(true_labels)

        true_labels_binarize = np.max(true_labels, axis=1)
        predictions_binarize = np.max(predictions, axis=1)

        accuracy = np.mean(true_labels_binarize == predictions_binarize)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        self.plot_results(predictions_binarize, true_labels_binarize)

    def plot_results(self, predictions, true_labels):
        sns.set_theme()
        sns.set_context("talk")
        df = pd.DataFrame({'Predicted': predictions, 'True': true_labels})
        ax = sns.scatterplot(data=df, x='Predicted', y='True')
        plt.show()
