from __future__ import print_function

''' 
panotti_models.py
Author: Scott Hawley

Where we'll put various NN models.

dumbCNN:  This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
   and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU


''' 
dumbCNN:  This is kind of a mixture of a dumbed-down versoin of Keun Woo Choi's 
    compact CNN model  (https://github.com/keunwoochoi/music-auto_tagging-keras)
    and the Keras MNIST classifier example
            (https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py)

    Uses same kernel, filters and pool size for everything
'''
def dumbCNN(X, Y, nb_classes, nb_layers=4):
    nb_filters = 32  # number of convolutional filters = "feature maps"
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling

    channels = X.shape[1]   # channels = 1 for mono, 2 for stereo

    print(" dumbCNN: X.shape = ",X.shape,", channels = ",channels)
    input_shape = (channels, X.shape[2], X.shape[3])  
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid', input_shape=input_shape))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(Activation('relu'))   # why not ELU? no reason. accident?

    for layer in range(nb_layers-1):   # add more layers than just the first
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(BatchNormalization(axis=1, mode=2))
        model.add(ELU(alpha=1.0))  
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))    # why not ELU? no reason. accident?
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model
    
