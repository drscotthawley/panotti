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
import os
from os.path import isfile
from panotti.models import * 
from panotti.datautils import *
from sklearn.metrics import roc_auc_score, roc_curve, auc
from timeit import default_timer as timer


if __name__ == '__main__':
    np.random.seed(1)

    # get the data
    X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names, sr = build_datasets(preproc=True)

    # make the model
    model = dumbCNN(X_train,Y_train, nb_classes=len(class_names))
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
