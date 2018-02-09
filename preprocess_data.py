#! /usr/bin/env python

'''
Preprocess audio
'''
from __future__ import print_function
import numpy as np
from panotti.datautils import *
import librosa
import librosa.display
from audioread import NoBackendError
import os
from multiprocessing import Pool
import platform
if platform.system() == 'Windows':
    import pickle

global_args = []    # after several hours of reading StackExchange on passing args w/ multiprocessing, I'm giving up and using globals

def get_canonical_shape(signal):
    if len(signal.shape) == 1:
        return (1, signal.shape[0])
    else:
        return signal.shape


def find_max_shape(path, mono=False, sr=None, dur=None, clean=False):
    if (mono) and (sr is not None) and (dur is not None):   # special case for speedy testing
        return [1, int(sr*dur)]
    shapes = []
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirname, filename)
            try:
                signal, sr = librosa.load(filepath, mono=mono, sr=sr)
            except NoBackendError as e:
                print("Could not open audio file {}".format(filepath))
                raise e
            if (clean):
                return get_canonical_shape(signal)
            shapes.append(get_canonical_shape(signal))

    return (max(s[0] for s in shapes), max(s[1] for s in shapes))


def convert_one_file(file_index):
    global global_args

    if not global_args:
        if platform.system() == 'Windows':
            f = open('data.pickle', 'rb')
            global_args = pickle.load(f)

    (printevery, class_index, class_files, nb_classes, classname, n_load, dirname, resample, mono, already_split, n_train, outpath, subdir, max_shape, clean) = global_args
    infilename = class_files[file_index]
    audio_path = dirname + '/' + infilename
    if (0 == file_index % printevery) or (file_index+1 == len(class_files)):
        print("\r Processing class ",class_index+1,"/",nb_classes,": \'",classname,
            "\', File ",file_index+1,"/", n_load,": ",audio_path,"                                 ",
            sep="",end="\r")

    sr = None
    if (resample is not None):
        sr = resample

    try:
        signal, sr = librosa.load(audio_path, mono=mono, sr=sr)
    except NoBackendError as e:
        print("\n*** ERROR: Could not open audio file {}".format(path),"\n")
        raise e

    shape = get_canonical_shape(signal)
    signal = np.reshape(signal,shape)
    padded_signal = np.zeros(max_shape)
    use_shape = list(max_shape[:])
    use_shape[0] = min( shape[0], max_shape[0] )
    use_shape[1] = min( shape[1], max_shape[1] )
    padded_signal[:use_shape[0], :use_shape[1]] = signal[:use_shape[0], :use_shape[1]]

    layers = make_layered_melgram(padded_signal, sr)

    if not already_split:
        if (file_index >= n_train):
            outsub = "Test/"
        else:
            outsub = "Train/"
    else:
        outsub = subdir

    outfile = outpath + outsub + classname + '/' + infilename+'.npy'
    np.save(outfile,layers)



def preprocess_dataset(inpath="Samples/", outpath="Preproc/", train_percentage=0.8, resample=None, already_split=False, sequential=False, mono=False, dur=None, clean=False):
    global global_args

    if (resample is not None):
        print(" Will be resampling at",resample,"Hz")

    if (True == already_split):
        print(" Data is already split into Train & Test")
        class_names = get_class_names(path=inpath+"Train/")   # get the names of the subdirectories
        sampleset_subdirs = ["Train/","Test/"]
    else:
        print(" Will be imposing 80-20 (Train-Test) split")
        class_names = get_class_names(path=inpath)   # get the names of the subdirectories
        sampleset_subdirs = ["./"]

    if (True == sequential):
        print(" Sequential ordering")
    else:
        print(" Shuffling ordering")

    print(" Finding max shape...")
    max_shape = find_max_shape(inpath, mono=mono, sr=resample, dur=dur, clean=clean)
    print(''' Padding all files with silence to fit shape:
              Channels : {}
              Samples  : {}
          '''.format(max_shape[0], max_shape[1]))

    nb_classes = len(class_names)
    print(" class_names = ",class_names)

    train_outpath = outpath+"Train/"
    test_outpath = outpath+"Test/"
    if not os.path.exists(outpath):
        os.mkdir( outpath );   # make a new directory for preproc'd files
        os.mkdir( train_outpath );
        os.mkdir( test_outpath );

    cpu_count = os.cpu_count()
    print("",cpu_count,"CPUs detected: Parallel execution across",cpu_count,"CPUs")

    for subdir in sampleset_subdirs: #non-class subdirs of Samples (in case already split)

        for class_index, classname in enumerate(class_names):   # go through the classes
            print("")           # at the start of each new class, newline

            # make new Preproc/ subdirectories for class
            if not os.path.exists(train_outpath+classname):
                os.mkdir( train_outpath+classname );
                os.mkdir( test_outpath+classname );
            dirname = inpath+subdir+classname
            #print("dirname = ",dirname)
            class_files = os.listdir(dirname)   # all filenames for this class
            class_files.sort()
            #print("class_files = ",class_files)
            if (not sequential): # shuffle directory listing (e.g. to avoid alphabetic order)
                np.random.shuffle(class_files)   # shuffle directory listing (e.g. to avoid alphabetic order)

            n_files = len(class_files)
            n_load = n_files            # sometimes we may multiple by a small # for debugging
            n_train = int( n_load * train_percentage)
            #print(", ",n_files," files in this class",sep="")

            printevery = 20

            global_args = (printevery, class_index, class_files, nb_classes, classname, n_load, dirname, resample, mono, already_split, n_train, outpath, subdir, max_shape, clean)

            parallel = True
            file_indices = tuple( range(len(class_files)) )

            if (not parallel):
                for file_index in file_indices:    # loop over all files
                    task=0
                    convert_one_file(file_index)
            else:
                if platform.system() == 'Windows':
                    f = open('data.pickle', 'wb')
                    pickle.dump(global_args, f)
                pool = Pool(cpu_count)
                pool.map(convert_one_file, file_indices)

    print("")    # at the very end, newline
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="preprocess_data: convert sames to python-friendly data format for faster loading")
    parser.add_argument("-a", "--already", help="data is already split into Test & Train (default is to add 80-20 split",action="store_true")
    parser.add_argument("-s", "--sequential", help="don't randomly shuffle data for train/test split",action="store_true")
    parser.add_argument("-m", "--mono", help="convert input audio to mono",action="store_true")
    parser.add_argument("-r", "--resample", type=int, default=44100, help="convert input audio to mono")
    parser.add_argument('-d', "--dur",  type=float, default=None,   help='Max duration (in seconds) of each clip')
    parser.add_argument("--clean", help="Assume 'clean data'; Do not check to find max shape (faster)", action='store_true')

    args = parser.parse_args()
    preprocess_dataset(resample=args.resample, already_split=args.already, sequential=args.sequential, mono=args.mono, dur=args.dur, clean=args.clean)
