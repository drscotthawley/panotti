#! /usr/bin/env python
''' 
Given one audio clip, output what the network thinks
'''
from __future__ import print_function
import numpy as np
import librosa
import os
from os.path import isfile
from panotti.models import *  
from panotti.datautils import *


def predict_one(signal, sr, class_names, model, weights_file="weights.hdf5"):
    X = make_layered_melgram(signal,sr)
    return model.predict_proba(X,batch_size=1)[0]


def main(args):
    np.random.seed(1)
    # Load the model
    weights_file=args.weights
    model = load_model(weights_file)
    if model is None:
        print("No weights file found.  Aborting")
        exit(1)
    model.summary()

    class_names = get_class_names(args.classpath)
    nb_classes = len(class_names)

    file_count = 0
    for infile in args.file:
        if os.path.isfile(infile):
            file_count += 1
            print("Operating on file",infile,"...")

            signal, sr = librosa.load(infile, mono=False, sr=44100)   # librosa naturally makes mono from stereo btw
            y_proba = predict_one(signal, sr, class_names, model, weights_file=args.weights)

            print("    ",infile,": ",end="")
            for i in range(nb_classes):
                print( class_names[i],": ",y_proba[i],", ",end="",sep="")
            print("--> ANSWER:", class_names[ np.argmax(y_proba)])
        else:
            print(" *** File",infile,"does not exist.  Skipping.")
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="predicts which class file(s) belong(s) to")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'), 
        help='weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-c', '--classpath', #type=argparse.string, 
        help='directory with list of classes', default="Preproc/Test/")

    parser.add_argument('file', help="file(s) to classify", nargs='+')   
    args = parser.parse_args()

    main(args)




