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
#from keras.callbacks import ModelCheckpoint #,EarlyStopping
import os
from os.path import isfile
from timeit import default_timer as timer
from panotti.multi_gpu import MultiGPUModelCheckpoint


def train_network(weights_file="weights.hdf5", classpath="Preproc/Train/", epochs=50, batch_size=20):
    np.random.seed(1)

    # Get the data
    X, Y, paths_train, class_names = build_dataset(path=classpath, batch_size=batch_size)

    # Can't trust Keras' validation_split if you're using multi-gpu. Have to do it yourself
    validation_split = 0.25
    split_ind = int(X.shape[0] * (1.0 - validation_split))
    X_train, Y_train = X[:split_ind,:,:], Y[:split_ind,:]
    X_val, Y_val = X[split_ind:,:,:], Y[split_ind:,:]

    # Instantiate the model
    model, serial_model = make_model(X_train, class_names, weights_file=weights_file)

    # Train the model, meter with auto-split of 25% of training data
    #   (So, given original Train/Test split of 80/20%, we end up with
    #    Train/Val/Test split of 60/20/20, as Andrew Ng recommends in his ML course )
    checkpointer = MultiGPUModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True, serial_model=serial_model)
    #earlystopping = EarlyStopping(patience=12)

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_val, Y_val), callbacks=[checkpointer]) #validation_split=0.25))

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
    parser.add_argument('--epochs', default=20, type=int, help="Number of iterations to train for")
    parser.add_argument('--batch_size', default=40, type=int, help="Number of clips to send to GPU at once")

    args = parser.parse_args()
    train_network(weights_file=args.weights, classpath=args.classpath, epochs=args.epochs, batch_size=args.batch_size)
