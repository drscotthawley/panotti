from __future__ import print_function

'''
panotti_models.py
Author: Scott Hawley

Where we'll put various NN models.

MyCNN:  This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
   and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD, Adam

from os.path import isfile
from panotti.multi_gpu import *
from tensorflow.python.client import device_lib
from panotti.multi_gpu import make_parallel, get_available_gpus


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
    model.add(Conv2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))        # Leave this relu & BN here.  ELU is not good here (my experience)

    for layer in range(nb_layers-1):   # add more layers than just the first
        model.add(Conv2D(nb_filters, kernel_size))
        #model.add(BatchNormalization(axis=1))  # ELU authors reccommend no BatchNorm. I confirm
        model.add(ELU(alpha=1.0))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(cl_dropout))

    model.add(Flatten())
    model.add(Dense(128))            # 128 is 'arbitrary' for now
    #model.add(Activation('relu'))   # relu (no BN) works fine here, however...
    model.add(ELU(alpha=1.0))        # ELU does a little better than relu here
    model.add(Dropout(dl_dropout))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model



def make_model(X, class_names, nb_layers=4, try_checkpoint=True,
    weights_file='weights.hdf5', quiet=False, missing_weights_fatal=False):
    ''' In the following, the reason we hang on to & return serial_model,
         is because Keras can't save parallel models, but according to fchollet
         the serial & parallel versions will always share the same weights
         (Strange but true!)
    '''

    gpu_count = get_available_gpus()

    if gpu_count <= 1:
        serial_model = MyCNN_Keras2(X, nb_classes=len(class_names), nb_layers=nb_layers)
    else:
        with tf.device("/cpu:0"):
            serial_model = MyCNN_Keras2(X, nb_classes=len(class_names), nb_layers=nb_layers)

    # Initialize weights using checkpoint if it exists.
    if (try_checkpoint):
        print("Looking for previous weights...")
        if ( isfile(weights_file) ):
            print ('Weights file detected. Loading from ',weights_file)
            loaded_model = load_model(weights_file)   # strip any previous parallel part, to be added back in later
            serial_model.set_weights( loaded_model.get_weights() )   # assign weights based on checkpoint
        else:
            if (missing_weights_fatal):
                print("Need weights file to continue.  Aborting")
                assert(not missing_weights_fatal)
            else:
                print('No weights file detected, so starting from scratch.')

    if (gpu_count >= 2):
        print(" Parallel run on",gpu_count,"GPUs")
        model = make_parallel(serial_model, gpu_count=gpu_count)
    else:
        model = serial_model

    opt = 'adadelta' # Adam(lr = 0.00001)  #
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    serial_model.compile(loss=loss, optimizer=opt, metrics=metrics)

    if (not quiet):
        print("Summary of serial model (duplicated across",gpu_count,"GPUs):")
        serial_model.summary()  # print out the model layers

    return model, serial_model
