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


def preprocess_dataset(inpath="Samples/", outpath="Preproc/", train_percentage=0.8, resample=None):

    if (resample is not None):
        print(" NOTICE: Resampling at",resample,"Hz")
        
    trainpath = outpath+"Train/"
    testpath = outpath+"Test/"
    if not os.path.exists(outpath):
        os.mkdir( outpath );   # make a new directory for preproc'd files
        os.mkdir( trainpath );  
        os.mkdir( testpath );   


    class_names = get_class_names(path=inpath)   # get the names of the subdirectories
    nb_classes = len(class_names)
    print("\nclass_names = ",class_names)
    for idx, classname in enumerate(class_names):   # go through the subdirs
        print("")

        # make new subdirectories for class
        if not os.path.exists(trainpath+classname):
            os.mkdir( trainpath+classname );   
            os.mkdir( testpath+classname );   


        class_files = os.listdir(inpath+classname)   # all filenames for this class
        np.random.shuffle(class_files)   # shuffle directory listing (e.g. to avoid alphabetic order)

        n_files = len(class_files)
        n_load = n_files            # sometimes we may multiple by a small # for debugging
        n_train = int( n_load * train_percentage)
        #print(", ",n_files," files in this class",sep="")

        printevery = 10

        for idx2, infilename in enumerate(class_files):    # go through all files for this class
            audio_path = inpath + classname + '/' + infilename

            if (0 == idx2 % printevery):
                print("\r Processing class ",idx+1,"/",nb_classes,": \'",classname,
                    "\', File ",idx2+1,"/", n_load,": ",audio_path,"                  ", 
                    sep="",end="")
            
            sr = None
            if (resample is not None):
                sr = resample
            aud, sr = librosa.load(audio_path, mono=False, sr=sr)    # read audio file

            # make mono file into "1-channel multichannel"
            if (aud.ndim == 1):
                aud = np.reshape( aud, (1,aud.shape[0]))

            # get mel-spectrogram for each channel, and layer them into multi-dim array
            for channel in range(aud.shape[0]):
                melgram = make_melgram(aud[channel])
                #melgram = librosa.logamplitude(librosa.feature.melspectrogram(aud[channel], 
                #    sr=sr, n_mels=96),ref_power=1.0)[np.newaxis,np.newaxis,:,:]

                if (0 == channel):
                    layers = melgram
                else:
                    layers = np.append(layers,melgram,axis=1)  # we keep axis=0 free for keras batches

            if (idx2 >= n_train):
                outsub = "Test/"
            else:
                outsub = "Train/"

            outfile = outpath + outsub + classname + '/' + infilename+'.npy'
            np.save(outfile,layers)
            
    print("")
    return

if __name__ == '__main__':
    preprocess_dataset(resample=44100)
