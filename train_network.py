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
from keras.callbacks import ModelCheckpoint #,EarlyStopping
import os
from os.path import isfile
from timeit import default_timer as timer


def train_network(weights_file="weights.hdf5", classpath="Preproc/Train/"):
    np.random.seed(1)

    # Get the data
    X_train, Y_train, paths_train, class_names = build_dataset(path=classpath)

    # Instantiate the model
    model = make_model(X_train, class_names, no_cp_fatal=False, weights_file=weights_file)

    # Train the model, meter with auto-split of 25% of training data
    #   (So, given original Train/Test split of 80/20%, we end up with 
    #    Train/Val/Test split of 60/20/20, as Andrew Ng recommends in his ML course )
    batch_size = 20
    epochs = 50
    checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)
    #earlystopping = EarlyStopping(patience=12)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_split=0.25, callbacks=[checkpointer])

    # Score the model against Test dataset
    X_test, Y_test, paths_test, class_names_test  = build_dataset(path=classpath+"../Test/")
    assert( class_names == class_names_test )
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="trains network using training dataset")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'), 
        help='weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-c', '--classpath', #type=argparse.string, 
        help='Train dataset directory with list of classes', default="Preproc/Train/")
    args = parser.parse_args()
    train_network(weights_file=args.weights, classpath=args.classpath)

