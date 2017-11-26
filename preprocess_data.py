#! /usr/bin/env python

''' 
Preprocess audio
'''
from __future__ import print_function
import numpy as np
from panotti.datautils import *
import librosa
import librosa.display
import os


def preprocess_dataset(inpath="Samples/", outpath="Preproc/", train_percentage=0.8, resample=None, already_split=False, sequential=False):

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

    nb_classes = len(class_names)
    print("\nclass_names = ",class_names)
        
    train_outpath = outpath+"Train/"
    test_outpath = outpath+"Test/"
    if not os.path.exists(outpath):
        os.mkdir( outpath );   # make a new directory for preproc'd files
        os.mkdir( train_outpath );  
        os.mkdir( test_outpath );   

    for subdir in sampleset_subdirs: #non-class subdirs of Samples (in case already split)
        for idx, classname in enumerate(class_names):   # go through the classes
            print("")
    
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
    
            for idx2, infilename in enumerate(class_files):    # go through all files for this class
                audio_path = dirname + '/' + infilename
                #print(" audio_path = ",audio_path)
                if (0 == idx2 % printevery) or (idx2+1 == len(class_files)):
                    print("\r Processing class ",idx+1,"/",nb_classes,": \'",classname,
                        "\', File ",idx2+1,"/", n_load,": ",audio_path,"                  ", 
                        sep="",end="")
                
                sr = None
                if (resample is not None):
                    sr = resample
                signal, sr = librosa.load(audio_path, mono=False, sr=sr)    # read audio file
    
                layers = make_layered_melgram(signal, sr)

                if not already_split:
                    if (idx2 >= n_train):
                        outsub = "Test/"
                    else:
                        outsub = "Train/"
                else:
                    outsub = subdir
    
                outfile = outpath + outsub + classname + '/' + infilename+'.npy'
                np.save(outfile,layers)
            
    print("")
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="preprocess_data.py: convert audio files to Python-friendly format for faster loading")
    parser.add_argument("-a", "--already", help="data is already split into Test & Train (default is to add 80-20 split",action="store_true")
    parser.add_argument("-s", "--sequential", help="don't randomly shuffle data for train/test split",action="store_true")
    args = parser.parse_args()
    preprocess_dataset(resample=44100, already_split=args.already, sequential=args.sequential)

