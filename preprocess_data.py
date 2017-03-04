from __future__ import print_function

''' 
Preprocess audio
'''
import numpy as np
import librosa
import librosa.display
import os

def get_class_names(path="Samples/"):  # class names are subdirectory names in Samples/ directory
    class_names = os.listdir(path)
    return class_names

def preprocess_dataset(inpath="Samples/", outpath="Preproc/"):

    if not os.path.exists(outpath):
        os.mkdir( outpath, 0755 );   # make a new directory for preproc'd files

    class_names = get_class_names(path=inpath)   # get the names of the subdirectories
    nb_classes = len(class_names)
    print("class_names = ",class_names)
    for idx, classname in enumerate(class_names):   # go through the subdirs

        if not os.path.exists(outpath+classname):
            os.mkdir( outpath+classname, 0755 );   # make a new subdirectory for preproc class

        class_files = os.listdir(inpath+classname)
        n_files = len(class_files)
        n_load = n_files
        print(' class name = {:14s} - {:3d}'.format(classname,idx),
            ", ",n_files," files in this class",sep="")

        printevery = 20
        for idx2, infilename in enumerate(class_files):
            audio_path = inpath + classname + '/' + infilename
            if (0 == idx2 % printevery):
                print('\r Loading class: {:14s} ({:2d} of {:2d} classes)'.format(classname,idx+1,nb_classes),
                       ", file ",idx2+1," of ",n_load,": ",audio_path,sep="")
            #start = timer()
            aud, sr = librosa.load(audio_path, sr=None)
            melgram = librosa.logamplitude(librosa.feature.melspectrogram(aud, sr=sr, n_mels=96),ref_power=1.0)[np.newaxis,np.newaxis,:,:]
            outfile = outpath + classname + '/' + infilename+'.npy'
            np.save(outfile,melgram)

if __name__ == '__main__':
    preprocess_dataset()


