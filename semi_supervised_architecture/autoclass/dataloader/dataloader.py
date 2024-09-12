from sklearn.preprocessing import StandardScaler, OneHotEncoder,RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np

class DataLoader:
    @staticmethod
    def load():
        data = pd.read_excel('autoclass\data\merged_data.xlsx') 
        data = data.drop(['ID','Old','Label Old','Type of Episode','FEP','Label'], axis=1)
        target_labels = ['SCZ'] #'SCZ', 'PAT', 'BD', 'Schizoaffective disorder', 'Substance Induced Psychotic Disorder'
        label_mapping = {label: idx + 1 for idx, label in enumerate(target_labels)} #labels index + 1
        
        # Exclude samples labeled as 'CTR'
        data = data[data['new'] != 'CTR']

        # Target Autoencoder (100% KK classes)
        x_target = data[data['new'].isin(target_labels)].drop(['new','Label New','Sample'], axis=1).reset_index(drop=True)
        y_target = data[data['new'].isin(target_labels)]['new'].reset_index(drop=True).replace(label_mapping)
        y_target_true = data[data['new'].isin(target_labels)]['Label New'].reset_index(drop=True)
        pid_target = data[data['new'].isin(target_labels)]['Sample'].reset_index(drop=True)

        # Non-Target (KU classes)
        x_non_target = data[~data['new'].isin(target_labels)].drop(['new','Label New', 'Sample'], axis=1).reset_index(drop=True)
        y_non_target = pd.Series([0] * len(x_non_target)) #label 0
        y_non_target_true = data[~data['new'].isin(target_labels)]['Label New'].reset_index(drop=True)
        pid_non_target = data[~data['new'].isin(target_labels)]['Sample'].reset_index(drop=True)

        # Merge Target and Non-Target with new labels
        x = pd.concat([x_target, x_non_target], axis=0).reset_index(drop=True)
        y = pd.concat([y_target, y_non_target], axis=0).reset_index(drop=True)
        y_true = pd.concat([y_target_true, y_non_target_true], axis=0).reset_index(drop=True)
        pid = pd.concat([pid_target, pid_non_target], axis=0).reset_index(drop=True)

        return y_target, x, y, y_true, pid 
    
    @staticmethod
    def pre_process_data(y_target, x, y, x_train, y_train, x_test, y_test, FLAGS):
        scaler = StandardScaler().fit(x_train)
        # scaler = RobustScaler().fit(x_train)
        x_std = scaler.transform(x)
        x_train_std = scaler.transform(x_train)
        x_test_std = scaler.transform(x_test)
        

        """ Train Dataset 
        input: n-k samples, categorical labels (0,1,2,3...)
        output: one hot encoded labels: [1 0],[0 1],[0 0]
        """
        x_train_std_dataset = tf.data.Dataset.from_tensor_slices(x_train_std).batch(FLAGS.batchsize, drop_remainder=False)
        y_train_one_hot = OneHotEncoder(handle_unknown='ignore').fit(y_target.values.reshape(-1, 1))
        # transform the target dataset with KU classes
        y_train_one_hot = y_train_one_hot.transform(y_train.values.reshape(-1, 1)).toarray()
        print('Train size: ', y_train_one_hot.shape)
        #print('Train target: ', y_train_one_hot)
        y_train_one_hot = tf.convert_to_tensor(y_train_one_hot, dtype=tf.float32)
        y_train_dataset = tf.data.Dataset.from_tensor_slices(y_train_one_hot).batch(FLAGS.batchsize, drop_remainder=False)
        data_train = tf.data.Dataset.zip((x_train_std_dataset, y_train_dataset))

        """ Test Dataset 
        input: k sample, categorical labels (0,1,2,3...)
        output: one hot encoded labels: [1 0],[0 1],[0 0]
        """
        x_test_std_dataset = tf.data.Dataset.from_tensor_slices(x_test_std).batch(FLAGS.batchsize, drop_remainder=False)
        # fit to y_test_KK so the labels from KU will be considerer as unknown by the OneHotEncoder, and encoded as [0 0]
        y_test_encoder = OneHotEncoder(handle_unknown='ignore').fit(y_target.values.reshape(-1, 1))
        # transform the target dataset with KU classes
        y_test_one_hot = y_test_encoder.transform(y_test.values.reshape(-1, 1)).toarray()
        print('Test size: ', y_test_one_hot.shape)
        y_test_one_hot = tf.convert_to_tensor(y_test_one_hot, dtype=tf.float32)
        y_test_dataset = tf.data.Dataset.from_tensor_slices(y_test_one_hot).batch(FLAGS.batchsize, drop_remainder=False)
        data_test = tf.data.Dataset.zip((x_test_std_dataset, y_test_dataset))

        """ Intire Dataset 
        input: 100% KK + KU classes, categorical labels (0,1,2,3...)
        output: one hot encoded labels: [1 0],[0 1],[0 0]
        """
        x_std_dataset = tf.data.Dataset.from_tensor_slices(x_std).batch(FLAGS.batchsize, drop_remainder=False)
        # fit to y_test_KK so the labels from KU will be considerer as unknown by the OneHotEncoder, and encoded as [0 0]
        y_encoder = OneHotEncoder(handle_unknown='ignore').fit(y_target.values.reshape(-1, 1))
        # transform the target dataset with KU classes
        y_one_hot = y_encoder.transform(y.values.reshape(-1, 1)).toarray()
        print('All data size: ', y_one_hot.shape)
        y_one_hot = tf.convert_to_tensor(y_one_hot, dtype=tf.float32)
        y_dataset = tf.data.Dataset.from_tensor_slices(y_one_hot).batch(FLAGS.batchsize, drop_remainder=False)
        data = tf.data.Dataset.zip((x_std_dataset, y_dataset))

        return data, data_train, data_test
    