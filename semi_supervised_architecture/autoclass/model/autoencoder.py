# External
import tensorflow as tf

class Autoencoder(tf.keras.Model):

    def __init__(self, FLAGS):
        super().__init__()

        # declares the parameters
        self.FLAGS = FLAGS 
        self.units_dense_1 = 2048
        self.units_dense_2 = 1024
        self.units_dense_3 = 512 
        self.units_dense_4 = 256 
        self.units_dense_9 = 128 #latent layer = smaller representation of the data
        self.units_dense_10 = 256
        self.units_dense_5 = 512 
        self.units_dense_6 = 1024
        self.units_dense_7 = 2048
        self.units_dense_8 = 2302 # input dimension

        self.units_dense_classifier = self.FLAGS.KK_classes 
        
        self.dp = tf.keras.layers.Dropout(0.2)

        self.dense_1 = tf.keras.layers.Dense(self.units_dense_1)#,activation='relu')
        self.dense_2 = tf.keras.layers.Dense(self.units_dense_2)#,activation='relu')
        self.dense_3 = tf.keras.layers.Dense(self.units_dense_3)#,activation='relu')
        self.dense_4 = tf.keras.layers.Dense(self.units_dense_4)#,activation='relu')
        self.dense_9 = tf.keras.layers.Dense(self.units_dense_9)#,activation='relu') #latent layer = smaller representation of the data
        self.dense_10 = tf.keras.layers.Dense(self.units_dense_10)#,activation='relu')
        self.dense_5 = tf.keras.layers.Dense(self.units_dense_5)#,activation='relu')
        self.dense_6 = tf.keras.layers.Dense(self.units_dense_6)#,activation='relu')
        self.dense_7 = tf.keras.layers.Dense(self.units_dense_7)#,activation='relu')
        self.dense_8 = tf.keras.layers.Dense(self.units_dense_8,activation='linear')
        self.l_relu = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.dense_classifier = tf.keras.layers.Dense(self.units_dense_classifier,activation='sigmoid')

    def call(self, x, training=True): 
        output_1 = self.dp(x,training=training)
        output_2 = self.dense_1(output_1)
        output_3 = self.l_relu(output_2)
        output_4 = self.dp(output_3,training=training)
        output_5 = self.dense_2(output_4)
        output_6 = self.l_relu(output_5)
        output_7 = self.dp(output_6,training=training)
        output_8 = self.dense_3(output_7)
        output_9 = self.l_relu(output_8)
        output_11 = self.dp(output_9,training=training)
        output_12 = self.dense_4(output_11)
        output_13 = self.l_relu(output_12)
        output_latent = self.dense_9(output_13) #latent layer = smaller representation of the data
        output_14 = self.l_relu(output_latent)
        output_15 = self.dense_10(output_14)
        output_16 = self.l_relu(output_15)
        output_17 = self.dp(output_16,training=training)
        output_18 = self.dense_5(output_17)
        output_19 = self.l_relu(output_18)
        output_20 = self.dp(output_19,training=training)
        output_21 = self.dense_6(output_20)
        output_22 = self.l_relu(output_21)
        output_23 = self.dp(output_22,training=training)
        output_24 = self.dense_7(output_23)
        output_25 = self.l_relu(output_24)
        output_final = self.dense_8(output_25)

        output_classifier = self.dense_classifier(output_latent)
        
        return output_final, output_latent , output_classifier
    
    def get_config(self):
        config = super(Autoencoder, self).get_config()
        # Add any necessary configuration parameters for your autoencoder
        return config
    
    