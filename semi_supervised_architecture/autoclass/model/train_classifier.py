# Internal
from model.autoencoder import Autoencoder
from model.binary_classifier import Binary_classifier
from model.train_autoencoder import TrainAutoencoder

# External
import pandas as pd
import tensorflow as tf
import time
import numpy as np
from model.performance_metrics import calculate_metrics
from sklearn.preprocessing import binarize

# MLP
class TrainClassifier:
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS
        
    def loss_func(self, y_true, y_predicted): #y_train
        """ Computes the mean-squared error loss
        Args
        ----------
            y_true (tensor): True values
            y_pred (tensor): Model predictions
        Returns
        -------
            Binary cross-entropy 
        """
        loss_MLP = self.bce_loss(y_true, y_predicted)
        return loss_MLP
        
    def train_step(self, latent, y_train):
        """ Train the Classifier with the parameters from the Autoencoder pre trained
        Args: 
            data_test (list): List with pre-processed test set with known and noise data
            FLAGS (argparse): Implementation parameters
        Returns
        -------
            Loss
        """  
        with tf.GradientTape() as tape:
            pred_classifier = self.classifier(latent)            
            loss_classification = self.loss_func(y_train, pred_classifier)

        variables = self.classifier.trainable_variables
        gradients = tape.gradient(loss_classification, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss_classification, pred_classifier

    def train_classifier(self, data):
        """ Train the Classifier using the latent layer from the autoencoder as input
        Args: 
            data (tf.data.Dataset): Dataset for training the classifier
        Returns
        -------
            MLP model trained
        """  
        # Initialize model
        self.autoencoder = Autoencoder(self.FLAGS)
        self.classifier = Binary_classifier(self.FLAGS)
        sequence_in = tf.zeros([1,2302])
        _, latent, _ = self.autoencoder(sequence_in)

        self.autoencoder.load_weights('m.h5') #load the weights from a different class     
        self.bce_loss = tf.keras.losses.BinaryCrossentropy() #from_logits=True
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        # training loop
        last_loss = {'epoch': 0, 'value': 1000}
        for epoch in range(30):#self.FLAGS.n_epochs):
            print(f'\n\nEpoch {epoch+1}/{self.FLAGS.n_epochs}')
            start = time.time()
            loss_epoch = []
            output_classifier_epoch = []
            for batch, (x_train, y_train) in enumerate(data):
                # Use the latent representation from the autoencoder as input to the classifier
                _, latent, _ = self.autoencoder(x_train,training=False)
                loss_batch, predictions_batch = self.train_step(latent, y_train)
                loss_epoch.append(loss_batch)
                output_classifier_epoch.append(predictions_batch)
                if batch == len(data) - 1:
                    print(f'{batch+1}/{len(data)} - {round(time.time() - start)}s - loss: {np.mean(loss_epoch):.4f}')
            if (last_loss['value'] - np.mean(loss_epoch)) >= self.FLAGS.min_delta:
                last_loss['value'] = np.mean(loss_epoch)
                last_loss['epoch'] = epoch + 1 
            if ((epoch + 1) - last_loss['epoch']) >= self.FLAGS.patience:
                break
            #self.classifier.save_weights('m_classifier.h5')

    def apply_classifier_and_plot(self, data, true_labels, pid, loop_num, accumulated_misclassified_samples):
        predictions = []
        misclassified_samples_all = []  # List to store misclassified samples from this loop

        for batch, (x_data, y_data) in enumerate(data):
            _, latent_vector, _ = self.autoencoder(x_data, training=False)
            output_classifier = self.classifier(latent_vector, training=False)
            predicted_labels = binarize(output_classifier, threshold=0.5)
            predictions.append(predicted_labels)  # Flatten predictions

            misclassified_samples_loop = []  # List to store misclassified samples for this loop

            for i, pid_name in enumerate(pid[batch * self.FLAGS.batchsize: (batch + 1) * self.FLAGS.batchsize]):
                if not np.array_equal(predicted_labels[i], y_data[i].numpy()):
                    misclassified_samples_loop.append((predicted_labels[i], pid_name))

            if misclassified_samples_loop:  # Append misclassified samples only if there are any
                misclassified_samples_all.extend([(sample, loop_num) for sample in misclassified_samples_loop])
        
        predictions = np.concatenate(predictions)
        x_data_only = data.map(lambda x, y: x)  # Extract only the input data x
        dataset = tf.data.Dataset.zip((x_data_only, tf.data.Dataset.from_tensor_slices(predictions).batch(self.FLAGS.batchsize, drop_remainder=False)))

        TA = TrainAutoencoder(self.FLAGS)  # Initialize the object since the method is defined with the self attribute
        TA.autoencoder = self.autoencoder
        TA.plot_latent_space(dataset, true_labels, pid, step='C', show=False)

        # Combine the misclassified samples from this loop with accumulated misclassified samples
        accumulated_misclassified_samples.extend(misclassified_samples_all)

        # Create a DataFrame to store all misclassified samples
        misclassified_df = pd.DataFrame(columns=['Sample', 'Predicted Class', 'True Label', 'Training Loop'])

        # Populate the DataFrame with misclassified samples from all loops
        for idx, (sample, loop_num) in enumerate(accumulated_misclassified_samples):
            predicted_label_str = "KK class" if sample[0][0] == 1 else "KU class"
            misclassified_df = misclassified_df.append({'Sample': idx + 1,
                                                        'Predicted Class': predicted_label_str,
                                                        'True Label': sample[1],
                                                        'Training Loop': loop_num}, ignore_index=True)

        # Print misclassified samples as a table
        print("\nMisclassified Samples:")
        print(misclassified_df.to_string(index=False))

        return accumulated_misclassified_samples  # Return the updated accumulated misclassified samples list

    def evaluate(self, dataset_test):
        print('\nPredicting on test set...') 
        predictions = []
        true_labels = []
        pred_classifier= []

        for num, (x_test,y_test) in enumerate(dataset_test):
            _, latent_vector, _ = self.autoencoder(x_test,training=False)
            output_classifier = self.classifier(latent_vector,training=False)
            predicted_labels = binarize(output_classifier, threshold=0.5)

            pred_classifier.append(output_classifier.numpy())
            predictions.append(predicted_labels)
            true_labels.append(y_test.numpy())

        pred_classifier = np.concatenate(pred_classifier)
        predictions = np.concatenate(predictions)
        true_labels = np.concatenate(true_labels)

        print("Classifier Output:", pred_classifier)
        print("Predictions:", predictions)
        print("True Labels:", true_labels)

        true_labels_binarize = np.max (true_labels, axis=1)
        predictions_binarize = np.max (predictions, axis=1)

        print("Predictions binarize:", predictions_binarize)
        print("True Labels binarize:", true_labels_binarize)

        f1_score, accuracy = calculate_metrics(true_labels_binarize, predictions_binarize)

        test_loss = np.mean([self.loss_func(true_label, pred) for true_label, pred in zip(true_labels, pred_classifier)])
        print("Loss: ", test_loss)

        return f1_score, accuracy, test_loss, predictions_binarize, true_labels_binarize