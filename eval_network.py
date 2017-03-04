from __future__ import print_function

''' 
Classify sounds using database - evaluation code
Author: Scott H. Hawley

This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
   and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

Trained using Fraunhofer IDMT's database of monophonic guitar effects, 
   clips were 2 seconds long, sampled at 44100 Hz
'''

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras import backend
from keras.utils import np_utils
import os
from os.path import isfile

from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer
from sklearn.metrics import roc_auc_score, roc_curve, auc

mono=True


def get_class_names(path="Preproc/"):  # class names are subdirectory names in Preproc/ directory
    class_names = os.listdir(path)
    return class_names

def get_total_files(path="Preproc/",train_percentage=0.8): 
    sum_total = 0
    sum_train = 0
    sum_test = 0
    subdirs = os.listdir(path)
    for subdir in subdirs:
        files = os.listdir(path+subdir)
        n_files = len(files)
        sum_total += n_files
        n_train = int(train_percentage*n_files)
        n_test = n_files - n_train
        sum_train += n_train
        sum_test += n_test
    return sum_total, sum_train, sum_test

def get_sample_dimensions(path='Preproc/'):
    classname = os.listdir(path)[0]
    files = os.listdir(path+classname)
    infilename = files[0]
    audio_path = path + classname + '/' + infilename
    melgram = np.load(audio_path)
    print("   get_sample_dimensions: melgram.shape = ",melgram.shape)
    return melgram.shape
 

def encode_class(class_name, class_names):  # makes a "one-hot" vector for each class name called
    try:
        idx = class_names.index(class_name)
        vec = np.zeros(len(class_names))
        vec[idx] = 1
        return vec
    except ValueError:
        return None

def decode_class(vec, class_names):  # generates a number from the one-hot vector
    return int(np.argmax(vec))

def shuffle_XY_paths(X,Y,paths):   # generates a randomized order, keeping X&Y(&paths) together
    assert (X.shape[0] == Y.shape[0] )
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    newpaths = paths
    for i in range(len(idx)):
        newX[i] = X[idx[i],:,:]
        newY[i] = Y[idx[i],:]
        newpaths[i] = paths[idx[i]]
    return newX, newY, newpaths


def build_datasets(train_percentage=0.8, preproc=False):
    '''
    So we make the training & testing datasets here, and we do it separately.
    Why not just make one big dataset, shuffle, and then split into train & test?
    because we want to make sure statistics in training & testing are as similar as possible
    '''
    if (preproc):
        path = "Preproc/"
    else:
        path = "Samples/"

    class_names = get_class_names(path=path)
    print("class_names = ",class_names)

    total_files, total_train, total_test = get_total_files(path=path, train_percentage=train_percentage)
    print("total files = ",total_files)

    nb_classes = len(class_names)
    mel_dims = get_sample_dimensions(path=path)
    # pre-allocate memory for speed (old method used np.concatenate, slow)
    X_train = np.zeros((total_train, mel_dims[1], mel_dims[2], mel_dims[3]))   
    Y_train = np.zeros((total_train, nb_classes))  
    X_test = np.zeros((total_test, mel_dims[1], mel_dims[2], mel_dims[3]))  
    Y_test = np.zeros((total_test, nb_classes))  
    paths_train = []
    paths_test = []

    train_count = 0
    test_count = 0
    for idx, classname in enumerate(class_names):
        this_Y = np.array(encode_class(classname,class_names) )
        this_Y = this_Y[np.newaxis,:]
        class_files = os.listdir(path+classname)
        n_files = len(class_files)
        n_load =  n_files
        n_train = int(train_percentage * n_load)
        printevery = 100
        print("")
        for idx2, infilename in enumerate(class_files[0:n_load]):          
            audio_path = path + classname + '/' + infilename
            if (0 == idx2 % printevery):
                print('\r Loading class: {:14s} ({:2d} of {:2d} classes)'.format(classname,idx+1,nb_classes),
                       ", file ",idx2+1," of ",n_load,": ",audio_path,sep="")
            #start = timer()
            if (preproc):
              melgram = np.load(audio_path)
              sr = 44100
            else:
              aud, sr = librosa.load(audio_path, mono=mono,sr=None)
              melgram = librosa.logamplitude(librosa.feature.melspectrogram(aud, sr=sr, n_mels=96),ref_power=1.0)[np.newaxis,np.newaxis,:,:]
            #end = timer()
            #print("time = ",end - start) 
            if (idx2 < n_train):
                # concatenate is SLOW for big datasets; use pre-allocated instead
                #X_train = np.concatenate((X_train, melgram), axis=0)  
                #Y_train = np.concatenate((Y_train, this_Y), axis=0)
                X_train[train_count,:,:] = melgram
                Y_train[train_count,:] = this_Y
                paths_train.append(audio_path)     # list-appending is still fast. (??)
                train_count += 1
            else:
                X_test[test_count,:,:] = melgram
                Y_test[test_count,:] = this_Y
                #X_test = np.concatenate((X_test, melgram), axis=0)
                #Y_test = np.concatenate((Y_test, this_Y), axis=0)
                paths_test.append(audio_path)
                test_count += 1
        print("")

    print("Shuffling order of data...")
    X_train, Y_train, paths_train = shuffle_XY_paths(X_train, Y_train, paths_train)
    X_test, Y_test, paths_test = shuffle_XY_paths(X_test, Y_test, paths_test)

    return X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names, sr



def build_model(X,Y,nb_classes):
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size
    nb_layers = 4
    input_shape = (1, X.shape[2], X.shape[3])

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid', input_shape=input_shape))
    model.add(BatchNormalization(axis=1, mode=2))
    model.add(Activation('relu'))

    for layer in range(nb_layers-1):
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(BatchNormalization(axis=1, mode=2))
        model.add(ELU(alpha=1.0))  
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model
    

if __name__ == '__main__':
    np.random.seed(1)

    # get the data
    X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names, sr = build_datasets(preproc=True)

    # make the model
    model = build_model(X_train,Y_train, nb_classes=len(class_names))
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
    model.summary()

    # Initialize weights using checkpoint if it exists. (Checkpointing requires h5py)
    checkpoint_filepath = 'weights.hdf5'
    if (True):
        print("Looking for previous weights...")
        if ( isfile(checkpoint_filepath) ):
            print ('Checkpoint file detected. Loading weights.')
            model.load_weights(checkpoint_filepath)
        else:
            print ('No checkpoint file detected. You gotta train_network first.')
            exit(1) 
    else:
        print('Starting from scratch (no checkpoint)')


    print("class names = ",class_names)

    batch_size = 128
    num_pred = X_test.shape[0]

    # evaluate the model
    print("Running model.evaluate...")
    scores = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])

    
    print("Running predict_proba...")
    y_scores = model.predict_proba(X_test[0:num_pred,:,:,:],batch_size=batch_size)
    auc_score = roc_auc_score(Y_test, y_scores)
    print("AUC = ",auc_score)

    n_classes = len(class_names)

    print(" Counting mistakes ")
    mistakes = np.zeros(n_classes)
    for i in range(Y_test.shape[0]):
        pred = decode_class(y_scores[i],class_names)
        true = decode_class(Y_test[i],class_names)
        if (pred != true):
            mistakes[true] += 1
    mistakes_sum = int(np.sum(mistakes))
    print("    Found",mistakes_sum,"mistakes out of",Y_test.shape[0],"attempts")
    print("      Mistakes by class: ",mistakes)

    print("Generating ROC curves...")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], 
             lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
