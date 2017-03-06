#! /usr/bin/env python

''' 
Classify sounds using database
Author: Scott H. Hawley

This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
   and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

Trained using Fraunhofer IDMT's database of monophonic guitar effects, 
   clips were 2 seconds long, sampled at 44100 Hz
'''
from __future__ import print_function
import numpy as np
import librosa
from panotti.models import *
from panotti.datautils import *
from keras.callbacks import ModelCheckpoint
import os
from os.path import isfile
from timeit import default_timer as timer


def train_network():
    np.random.seed(1)

    # get the data
    X_train, Y_train, paths_train, class_names, sr = build_dataset(path="Preproc/Train/")
    X_test, Y_test, paths_test, class_names_test, sr = build_dataset(path="Preproc/Test/")
    assert( class_names == class_names_test)


    # make the model
    model = dumbCNN(X_train,Y_train, nb_classes=len(class_names), nb_layers=4)
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
    model.summary()


    # Initialize weights using checkpoint if it exists. (Checkpointing requires h5py)
    load_checkpoint = True
    checkpoint_filepath = 'weights.hdf5'
    if (load_checkpoint):
        print("Looking for previous weights...")
        if ( isfile(checkpoint_filepath) ):
            print ('Checkpoint file detected. Loading weights.')
            model.load_weights(checkpoint_filepath)
        else:
            print ('No checkpoint file detected.  Starting from scratch.')
    else:
        print('Starting from scratch (no checkpoint)')
    checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True)


    # train and score the model
    batch_size = 128
    nb_epoch = 100
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test), callbacks=[checkpointer])
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    train_network()
