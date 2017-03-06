#! /usr/bin/env python

''' 
Classify sounds using database - evaluation code
Author: Scott H. Hawley

This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
   and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

Trained using Fraunhofer IDMT's database of monophonic guitar effects, 
   clips were 2 seconds long, sampled at 44100 Hz
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from os.path import isfile
from panotti.models import * 
from panotti.datautils import *
from sklearn.metrics import roc_auc_score, roc_curve, auc
from timeit import default_timer as timer


def create(n, constructor=list):   # creates an list of empty lists
    for _ in xrange(n):
        yield constructor()

def count_mistakes(y_scores,Y_test,paths_test,class_names):
    n_classes = len(class_names)
    mistake_count = np.zeros(n_classes)
    mistake_log = list(create(n_classes))
    for i in range(Y_test.shape[0]):
        pred = decode_class(y_scores[i],class_names)
        truth = decode_class(Y_test[i],class_names)
        if (pred != truth):
            mistake_count[truth] += 1
            mistake_log[truth].append( paths_test[i]+": should be "+class_names[truth]+
                " but came out as "+class_names[pred])

    mistakes_sum = int(np.sum(mistake_count))
    print("    Found",mistakes_sum,"total mistakes out of",Y_test.shape[0],"attempts")
    print("      Mistakes by class: ")
    for i in range(n_classes):
        print("          class \'",class_names[i],"\': ",int(mistake_count[i]), sep="")
        for j in range(len(mistake_log[i])):
            print("                  ",mistake_log[i][j])
    return


def eval_network():
    np.random.seed(1)

    # get the data
    X_test, Y_test, paths_test, class_names, sr = build_dataset(
        path="Preproc/Test/",shuffle=False,load_frac=1)

    # make the model
    model = dumbCNN(X_test,Y_test, nb_classes=len(class_names),nb_layers=4)
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
    count_mistakes(y_scores,Y_test,paths_test,class_names)

    print("Generating ROC curves...")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2                      # line width
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw, label='class '+class_names[i]+": AUC="+str(roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return


if __name__ == '__main__':
    eval_network()
