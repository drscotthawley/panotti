#! /usr/bin/env python3
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # less TF messages, thanks

def get_canonical_shape(signal):
    if len(signal.shape) == 1:
        return (1, signal.shape[0])
    else:
        return signal.shape


def predict_one(signal, sr, model):# class_names, model)#, weights_file="weights.hdf5"):
    X = make_layered_melgram(signal,sr)
    #print("signal.shape, melgram_shape, sr = ",signal.shape, X.shape, sr)
    return model.predict(X,batch_size=1,verbose=False)[0]


def main(args):
    np.random.seed(1)
    weights_file=args.weights
    dur = args.dur
    resample = args.resample
    mono = args.mono

    # Load the model
    model = load_model(weights_file)
    if model is None:
        print("No weights file found.  Aborting")
        exit(1)
    #model.summary()

    class_names = get_class_names(args.classpath)
    nb_classes = len(class_names)
    print(nb_classes," classes to choose from")

    file_count = 0
    json_file = open("data.json", "w")
    json_file.write('{\n"items":[')

    idnum = 0
    numfiles = len(args.file)
    print("Reading",numfiles,"files")
    for infile in args.file:
        if os.path.isfile(infile):
            file_count += 1
            print("File",infile,":",end="")

            signal, sr = librosa.load(infile, mono=mono, sr=resample)

            # resize / cut / pad signal to make expect length of clip (used in training)
            padded_signal = signal
            if (mono) and (dur is not None) and (resample is not None):
                max_shape = [1, int(dur * resample)]
                shape = get_canonical_shape(signal)
                signal = np.reshape(signal, shape)
                padded_signal = np.zeros(max_shape)
                use_shape = max_shape[:]
                use_shape[0] = min(shape[0], max_shape[0])
                use_shape[1] = min(shape[1], max_shape[1])
                padded_signal[:use_shape[0], :use_shape[1]] = signal[:use_shape[0], :use_shape[1]]

            #print("padded_signal.shape = ",padded_signal.shape)

            y_proba = predict_one(padded_signal, sr, model) # class_names, model, weights_file=args.weights)

            for i in range(nb_classes):
                print( class_names[i],": ",y_proba[i],", ",end="",sep="")
            answer = class_names[ np.argmax(y_proba)]
            print("--> ANSWER:", class_names[ np.argmax(y_proba)])
            outstr = '\n  {\n   "id": "'+str(idnum)+'",\n      "name":"'+infile+'",\n      "tags":[\n   "'+answer+'"]\n  }'
            if (idnum < numfiles-1):
                outstr += ','
            json_file.write(outstr)
        else:
            pass #print(" *** File",infile,"does not exist.  Skipping.")
        idnum += 1

    json_file.write("]\n}\n")
    json_file.close()

    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="predicts which class file(s) belong(s) to")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-c', '--classpath', #type=argparse.string,
        help='directory with list of classes', default="Preproc/Test/")
    parser.add_argument("-m", "--mono", help="convert input audio to mono",action="store_true")
    parser.add_argument("-r", "--resample", type=int, default=44100, help="convert input audio to mono")
    parser.add_argument('-d', "--dur",  type=float, default=None,   help='Max duration (in seconds) of each clip')

    parser.add_argument('file', help="file(s) to classify", nargs='+')
    args = parser.parse_args()

    main(args)
