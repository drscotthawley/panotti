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


def predict_class(signal, sr, class_names, model=None):
    X = make_melgram(signal,sr)
    if model is None:
        model = load_model(X, class_names, no_cp_fatal=True)
    return model.predict_proba(X,batch_size=1)[0], model



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="predicts which class file(s) belong(s) to")
#   parser.add_argument("n_elev", help="number of discrete poitions of elevation",type=int)
    parser.add_argument('file', help="file(s) to classify", nargs='+')   
    args = parser.parse_args()
    model = None

    np.random.seed(1)
    class_names = get_class_names(path="Preproc/Test/")
    nb_classes = len(class_names)

    file_count = 0
    for infile in args.file:
        if os.path.isfile(infile):
            file_count += 1
            print("Operating on file",infile,"...")

            signal, sr = librosa.load(infile, mono=False, sr=44100)   # librosa naturally makes mono from stereo btw
            y_proba, model = predict_class(signal, sr, class_names, model)

            print("    ",infile,": ",end="")
            for i in range(nb_classes):
                print( class_names[i],": ",y_proba[i],", ",end="",sep="")
            print("--> ANSWER:", class_names[ np.argmax(y_proba)])
        else:
            print(" *** File",infile,"does not exist.  Skipping.")
 
