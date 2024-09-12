from keras.models import Model
from keras.layers import Input, Dense

def build_autoencoder(input_shape, latent_dim):

    input_layer = Input(shape=(input_shape,))
    encoder = Dense(512, activation='relu')(input_layer)
    encoder = Dense(256, activation='relu')(encoder)
    latent = Dense(latent_dim, activation='relu')(encoder)
    decoder = Dense(256, activation='relu')(latent)
    decoder = Dense(512, activation='relu')(decoder)
    output_layer = Dense(input_shape, activation='sigmoid')(decoder)

    autoencoder_model = Model(inputs=input_layer, outputs=output_layer)
    latent_output = Model(inputs=input_layer, outputs=latent)

    return autoencoder_model, latent_output

