# External
import tensorflow as tf

# MLP classifier
class Binary_classifier(tf.keras.Model):
    def __init__(self, FLAGS):
        super(Binary_classifier, self).__init__()
        self.FLAGS = FLAGS

        self.units_dense_1 = 512 # dimension of the latent layer from the autoencoder 
        self.units_dense_2 = 128
        self.units_dense_3 = 64
        self.units_dense_4 = self.FLAGS.KK_classes
        
        self.l_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.dense_1 = tf.keras.layers.Dense(self.units_dense_1)#,activation='relu')
        
        self.dense_2 = tf.keras.layers.Dense(self.units_dense_2)#,activation='relu')
        self.dense_3 = tf.keras.layers.Dense(self.units_dense_3)#,activation='relu')
        self.dense_4 = tf.keras.layers.Dense(self.units_dense_4,activation='sigmoid')
        
        self.dp = tf.keras.layers.Dropout(0.2)

    def call(self, latent_vector, training=True):
        
        
        output_1 = self.dense_1(latent_vector)
        output_2 = self.l_relu(output_1)
        # output_2 = self.dp(output_1,training=training)
        output_3 = self.dense_2(output_2)
        output_4 = self.l_relu(output_3)
        # output_4 = self.dp(output_3,training=training)
        output_5 = self.dense_3(output_4)
        output_6 = self.l_relu(output_5)
        pred_out = self.dense_4(output_6)
        
        return pred_out
    

