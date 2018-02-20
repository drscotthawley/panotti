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


def predict_one(signal, sr, model, expected_melgram_shape):# class_names, model)#, weights_file="weights.hdf5"):
    X = make_layered_melgram(signal,sr)
    print("signal.shape, melgram_shape, sr = ",signal.shape, X.shape, sr)

    if (X.shape[1:] != expected_melgram_shape):   # resize if necessary, pad with zeros
        Xnew = np.zeros([1]+list(expected_melgram_shape))
        min1 = min(  Xnew.shape[1], X.shape[1]  )
        min2 = min(  Xnew.shape[2], X.shape[2]  )
        min3 = min(  Xnew.shape[3], X.shape[3]  )
        Xnew[0,:min1,:min2,:min3] = X[0,:min1,:min2,:min3]  # truncate
        X = Xnew
    return model.predict(X,batch_size=1,verbose=False)[0]


def main(args):
    np.random.seed(1)
    weights_file=args.weights
    dur = args.dur
    resample = args.resample
    mono = args.mono

    # Load the model
    model, class_names = load_model_ext(weights_file)
    if model is None:
        print("No weights file found.  Aborting")
        exit(1)

    #model.summary()

    #TODO: Keras load_models is spewing warnings about not having been compiled. we can ignore those,
    #   how to turn them off?  Answer: can invoke with python -W ignore ...

    #class_names = get_class_names(args.classpath) # now encoding names in model weights file
    nb_classes = len(class_names)
    print(nb_classes," classes to choose from: ",class_names)
    expected_melgram_shape = model.layers[0].input_shape[1:]
    print("Expected_melgram_shape = ",expected_melgram_shape)
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

            signal, sr = load_audio(infile, mono=mono, sr=resample)

            y_proba = predict_one(signal, sr, model, expected_melgram_shape) # class_names, model, weights_file=args.weights)

            for i in range(nb_classes):
                print( class_names[i],": ",y_proba[i],", ",end="",sep="")
            answer = class_names[ np.argmax(y_proba)]
            print("--> ANSWER:", class_names[ np.argmax(y_proba)])
            outstr = '\n  {\n   "id": "'+str(idnum)+'",\n      "name":"'+infile+'",\n      "tags":[\n   "'+answer+'"]\n  }'
            if (idnum < numfiles-1):
                outstr += ','
            json_file.write(outstr)
            json_file.flush()     # keep json file up to date
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
    #parser.add_argument('-c', '--classpath', #type=argparse.string, help='directory with list of classes', default="Preproc/Test/")
    parser.add_argument("-m", "--mono", help="convert input audio to mono",action="store_true")
    parser.add_argument("-r", "--resample", type=int, default=44100, help="convert input audio to mono")
    parser.add_argument('-d', "--dur",  type=float, default=None,   help='Max duration (in seconds) of each clip')

    parser.add_argument('file', help="file(s) to classify", nargs='+')
    args = parser.parse_args()

    main(args)
