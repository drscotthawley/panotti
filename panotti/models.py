from __future__ import print_function

'''
panotti_models.py
Author: Scott Hawley

Where we'll put various NN models.

MyCNN:  This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
   and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from os.path import isfile

from distutils.version import LooseVersion


'''
MyCNN:  This is kind of a mixture of a dumbed-down versoin of Keun Woo Choi's
    compact CNN model  (https://github.com/keunwoochoi/music-auto_tagging-keras)
    and the Keras MNIST classifier example
            (https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py)

    Uses same kernel, filters and pool size for everything
'''
def MyCNN(X, nb_classes, nb_layers=4):
    nb_filters = 32  # number of convolutional filters = "feature maps"
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    cl_dropout = 0.5    # conv. layer dropout
    dl_dropout = 0.6    # dense layer dropout

    channels = X.shape[1]   # channels = 1 for mono, 2 for stereo

    print(" MyCNN: X.shape = ",X.shape,", channels = ",channels)
    input_shape = (channels, X.shape[2], X.shape[3])
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid', input_shape=input_shape))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(Activation('relu'))

    for layer in range(nb_layers-1):   # add more layers than just the first
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        #model.add(BatchNormalization(axis=1, mode=2)) # ELU authors reccommend no BatchNorm
        model.add(ELU(alpha=1.0))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(cl_dropout))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(dl_dropout))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def MyCNN_Keras2(X, nb_classes, nb_layers=4):
    from keras import backend as K
    K.set_image_data_format('channels_first')

    nb_filters = 32  # number of convolutional filters = "feature maps"
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    cl_dropout = 0.5    # conv. layer dropout
    dl_dropout = 0.8    # dense layer dropout

    channels = X.shape[1]   # channels = 1 for mono, 2 for stereo

    print(" MyCNN_Keras2: X.shape = ",X.shape,", channels = ",channels)
    input_shape = (channels, X.shape[2], X.shape[3])
    model = Sequential()
    #model.add(Conv2D(nb_filters, kernel_size, border_mode='valid', input_shape=input_shape))
    model.add(Conv2D(nb_filters, kernel_size, border_mode='valid', input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    for layer in range(nb_layers-1):   # add more layers than just the first
        model.add(Conv2D(nb_filters, kernel_size))
        #model.add(BatchNormalization(axis=1, mode=2)) # ELU authors reccommend no BatchNorm
        model.add(ELU(alpha=1.0))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(cl_dropout))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(dl_dropout))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model




def make_model(X, class_names, nb_layers=4, try_checkpoint=True,
    no_cp_fatal=False, weights_file='weights.hdf5'):

    model = None
    from_scratch = True
    # Initialize weights using checkpoint if it exists.
    if (try_checkpoint):
        print("Looking for previous weights...")
        if ( isfile(weights_file) ):
            print ('Weights file detected. Loading from ',weights_file)
            model = load_model(weights_file)
            from_scratch = False
        else:
            if (no_cp_fatal):
                raise Exception("No weights file detected; can't do anything.  Aborting.")
            else:
                print('No weights file detected, so starting from scratch.')

    if from_scratch:
        if (LooseVersion(keras.__version__) < LooseVersion("2")):
            print("Making Keras 1 version of model")
            model = MyCNN(X, nb_classes=len(class_names), nb_layers=nb_layers)
        else:
            print("Making Keras 2 version of model")
            model = MyCNN_Keras2(X, nb_classes=len(class_names), nb_layers=nb_layers)

    model.summary()

    return model
