#Internal
import numpy as np
from dataloader.dataloader import DataLoader
from model.train_autoencoder import TrainAutoencoder
from model.train_classifier import TrainClassifier
from model.performance_metrics import calculate_cm
from model.argument_parser import argparser

# External
import warnings
warnings.filterwarnings('ignore')


def run_train_model(FLAGS):

    print('\nLoading data...')
    y_target, x, y, y_true, pid = DataLoader().load()
    
    # Initialize lists to store evaluation results
    test_accuracies = []
    test_f1= []
    test_losses = []
    accumulated_misclassified_samples = [] 

    # Lists to collect true and predicted labels for LOO-CV
    y_trues = []
    y_preds = []
    
    # Iterate over each sample for LOOCV
    for k in range(len(x)):
        print('--------------------------------------')
        print(f'\nLeave-one-out cross-validation: Left out sample {k+1}/{len(x)}')
        print('\n')
        
        # Exclude the i-th sample for testing
        x_train_loo = x.drop(k)
        y_train_loo = y.drop(k)
        y_train_true_loo = y_true.drop(k)
        pid_train_loo = pid.drop(k)

        x_test_loo = x.iloc[[k]]  # Accessing a single row, needs double square brackets
        y_test_loo = y.iloc[[k]]  # Accessing a single row, needs double square brackets

        #data = all data, data_train = n-k, data_test = k (one patient sample)
        print('\nPre-processing data...')
        data, data_train, data_test = DataLoader.pre_process_data(y_target, x, y, x_train_loo, y_train_loo, x_test_loo, y_test_loo, FLAGS)

        #Train Autoencoder and save parameters
        print('\nTraining the Autoencoder...')
        train_AE = TrainAutoencoder(FLAGS)
        train_AE.train_autoencoder(data_train, y_train_true_loo, pid_train_loo) 
        train_AE.apply_model_and_plot_latent(data, y_true, pid) 
        
        print('\nCreating GIF...')
        train_AE.create_gif()
        
        #Train MLP and save parameters
        train_AE_BC = TrainClassifier(FLAGS)
        train_AE_BC.train_classifier(data_train) 

        # Evaluate on test set
        print('\nEvaluating...')
        f1_score, accuracy, loss, y_pred_test, y_true_test = train_AE_BC.evaluate(data_test)
        test_accuracies.append(accuracy)
        test_f1.append(f1_score)
        test_losses.append(loss)

        # Append the true and predicted binarized labels for the current test sample
        y_trues.append(y_true_test)
        y_preds.append(y_pred_test)

        # Flatten the predicted labels
        y_pred_flat = np.concatenate(y_preds).astype(int)

        # Convert the true labels and predicted labels to numpy arrays
        y_true_np = np.array(y_trues)
        y_pred_np = np.array(y_pred_flat)
        
        # Apply classifier and plot
        accumulated_misclassified_samples = train_AE_BC.apply_classifier_and_plot(data, y_true, pid, k+1, accumulated_misclassified_samples)

    # Calculate mean accuracy and loss
    mean_f1 = np.mean(test_f1)
    mean_accuracy =np.mean(test_accuracies)
    mean_loss = np.mean(test_losses)
    print(f'\nMean F1-score: {mean_f1}, Mean Accuracy: {mean_accuracy}, Mean Loss: {mean_loss}')

    print('\nPerformance metrics for the entire dataset:')
    print('\n')
    calculate_cm(y_true_np, y_pred_np)

def run():    
    """Loads the parameters, builds the model, trains and evaluates it"""
    FLAGS = argparser()
    if FLAGS.option == 'train':
    	run_train_model(FLAGS)

if __name__ == '__main__':
    run()