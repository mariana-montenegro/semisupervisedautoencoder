import argparse
import os

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default='train', help='train or validation')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--KK_classes', type=int, default=1, help='Number of KK classes')
    parser.add_argument('--KK_classes_name', type=str, nargs='+', default=['Schizophrenia'], help='KK classes names')
    parser.add_argument('--KU_classes_name', type=str, nargs='+', default=['Bipolar Disorder', 'PAT', 'Other Primary Psychotic Disorder', 'Substance Induced Psychotic Disorder', 'Secondary Psychotic Syndrome', 'Schizoaffective disorder'], help='KU classes names')
    parser.add_argument('--test_rate', type=float, default=0.1, help='Rate of test set')
    parser.add_argument('--threshold_min', type=float, default=6, help='Minimum number of functional groups where masking considers FGs')
    parser.add_argument('--patience', type=int, default=15, help='Epochs to keep training without improvement')
    parser.add_argument('--optimizer_fn', type=str, nargs='+', action='append', help='Optimizer Function Parameters - optimizer, learning rate, beta1, beta2 and epsilon')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum delta')
    parser.add_argument('--n_layers', type=int, nargs='+', help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--activation_func', type=str, nargs='+', help='Feed-forward activation function')
    parser.add_argument('--ff_dim', type=int, nargs='+', help='Feed-forward Dimension')
    parser.add_argument('--batchsize', type=int, default=8, help='Batch size')
    parser.add_argument('--max_strlen', type=int, default=150, help='Maximum string length')
    parser.add_argument('--checkpoint_path', type=str, default='autoclass/models', help='Directory for checkpoint weights')
    parser.add_argument('--autoencoder_weight', type=float, default=0.5, help='Weight for autoencoder loss')
    parser.add_argument('--classifier_weight', type=float, default=0.5, help='Weight for classifier loss')
    
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS

def logging(msg, FLAGS):
    fpath = os.path.join(FLAGS.log_dir, "log.txt")
    with open(fpath, "a") as fw:
        fw.write("%s\n" % msg)
    print("------------------------//------------------------")
    print(msg)
    print("------------------------//------------------------")
