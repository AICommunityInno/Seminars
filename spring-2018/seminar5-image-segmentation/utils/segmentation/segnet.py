from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
import json

kernel = 3

def build_model(img_w, img_h, n_channels=3):
    encoding_layers = [
        Conv2D(64, (kernel, kernel), padding='same', input_shape=(img_h, img_w, n_channels)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    ]

    autoencoder = models.Sequential()
    autoencoder.encoding_layers = encoding_layers

    for l in autoencoder.encoding_layers:
        autoencoder.add(l)

    decoding_layers = [
        UpSampling2D(),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Conv2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(1, (1, 1), padding='valid'),
        BatchNormalization(),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)

    # autoencoder.add(Reshape((1, img_h * img_w)))
    # autoencoder.add(Permute((2, 1)))
    
    autoencoder.add(Activation('sigmoid'))
    
    return autoencoder
