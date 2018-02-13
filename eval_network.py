#! /usr/bin/env python3

'''
Classify sounds using database - evaluation code.
Generates a score based on contents of Preproc/Test/

Author: Scott H. Hawley

This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
   and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

Trained using Fraunhofer IDMT's database of monophonic guitar effects,
   clips were 2 seconds long, sampled at 44100 Hz
'''
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')

from keras.models import  load_model
import matplotlib.pyplot as plt
import librosa
import os
from panotti.models import *
from panotti.datautils import *
from sklearn.metrics import roc_auc_score, roc_curve, auc
from timeit import default_timer as timer


def create(n, constructor=list):   # creates an list of empty lists
    for _ in range(n):
        yield constructor()

def count_mistakes(y_scores,Y_test,paths_test,class_names):
    n_classes = len(class_names)
    mistake_count = np.zeros(n_classes)
    mistake_log = list(create(n_classes))
    max_string_len = 0
    for i in range(Y_test.shape[0]):
        pred = decode_class(y_scores[i],class_names)
        truth = decode_class(Y_test[i],class_names)
        if (pred != truth):
            mistake_count[truth] += 1
            max_string_len = max( len(paths_test[i]), max_string_len )
            mistake_log[truth].append( paths_test[i].ljust(max_string_len)+": should be "+class_names[truth]+
                " but came out as "+class_names[pred])

    mistakes_sum = int(np.sum(mistake_count))
    print("    Found",mistakes_sum,"total mistakes out of",Y_test.shape[0],"attempts")
    print("      Mistakes by class: ")

    for i in range(n_classes):
        print("          class \'",class_names[i],"\': ",int(mistake_count[i]), sep="")
        for j in range(len(mistake_log[i])):
            print("                  ",mistake_log[i][j])
    return


def eval_network(weights_file="weights.hdf5", classpath="Preproc/Test/", batch_size=40):
    np.random.seed(1)

    # get the data
    X_test, Y_test, paths_test, class_names = build_dataset(path=classpath, batch_size=batch_size)
    print("class names = ",class_names)
    n_classes = len(class_names)

    # Load the model
    model, serial_model = setup_model(X_test, class_names, weights_file=weights_file, missing_weights_fatal=True)
    model.summary()

    num_pred = X_test.shape[0]

    print("Running predict...")
    y_scores = model.predict(X_test[0:num_pred,:,:,:],batch_size=batch_size)


    print("Counting mistakes ")
    count_mistakes(y_scores,Y_test,paths_test,class_names)

    print("Measuring ROC...")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    auc_score = roc_auc_score(Y_test, y_scores)
    print("Global AUC = ",auc_score)

    print("\nDrawing ROC curves...")
    fig = plt.figure()
    lw = 2                      # line width
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw, label=class_names[i]+": AUC="+'{0:.4f}'.format(roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.draw()
    #plt.show(block=False)
    roc_filename = "roc_curves.png"
    print("Saving curves to file",roc_filename)
    plt.savefig(roc_filename)
    plt.close(fig)
    print("")

    # evaluate the model
    print("Running model.evaluate...")
    scores = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    print("All model scores:")
    print(model.metrics_names)
    print(scores)

    print("\nFinished.")
    #plt.show()
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="evaluates network on testing dataset")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-c', '--classpath', #type=argparse.string,
        help='test dataset directory with list of classes', default="Preproc/Test/")
    parser.add_argument('--batch_size', default=40, type=int, help="Number of clips to send to GPU at once")

    args = parser.parse_args()
    eval_network(weights_file=args.weights, classpath=args.classpath, batch_size=args.batch_size)
