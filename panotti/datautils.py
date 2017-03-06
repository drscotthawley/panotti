''' 
p_datautils.py:  Just some routines that we use for moving data around
'''
from __future__ import print_function
import numpy as np
import librosa
import os
from os.path import isfile


# class names are subdirectory names in Preproc/ directory
def get_class_names(path="Preproc/Train/", sort=True):
    if (sort):     
        class_names = sorted(os.listdir(path))     # sorted alphabetically for consistency with "ls" command
    else:
        class_names = os.listdir(path)             # not in same order as "ls", because Python
    return class_names


def get_total_files(class_names, path="Preproc/Train/"): 
    sum_total = 0
    for subdir in class_names:
        files = os.listdir(path+subdir)
        n_files = len(files)
        sum_total += n_files
    return sum_total


def get_sample_dimensions(class_names, path='Preproc/Train/'):
    classname = class_names[0]
    audio_path = path + classname + '/'
    infilename = os.listdir(audio_path)[0]
    melgram = np.load(audio_path+infilename)
    print("   get_sample_dimensions: melgram.shape = ",melgram.shape)
    return melgram.shape
 

def encode_class(class_name, class_names):  # makes a "one-hot" vector for each class name called
    try:
        idx = class_names.index(class_name)
        vec = np.zeros(len(class_names))
        vec[idx] = 1
        return vec
    except ValueError:
        return None


def decode_class(vec, class_names):  # generates a number from the one-hot vector
    return int(np.argmax(vec))


def shuffle_XY_paths(X,Y,paths):   # generates a randomized order, keeping X&Y(&paths) together
    assert (X.shape[0] == Y.shape[0] )
    print("shuffle_XY_paths: Y.shape[0], len(paths) = ",Y.shape[0], len(paths))
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    newpaths = paths
    for i in range(len(idx)):
        newX[i] = X[idx[i],:,:]
        newY[i] = Y[idx[i],:]
        newpaths[i] = paths[idx[i]]
    return newX, newY, newpaths


# can be used for test dataset as well
def build_dataset(path="Preproc/Train/", load_frac=1.0):

    class_names = get_class_names(path=path)
    print("class_names = ",class_names)
    nb_classes = len(class_names)

    total_files = get_total_files(class_names, path=path)
    total_load = int(total_files * load_frac)
    print("total files = ",total_files,", going to load total_load = ",total_load)

    # pre-allocate memory for speed (old method used np.concatenate, slow)
    mel_dims = get_sample_dimensions(class_names,path=path)  # get dims of sample data file
    print(" melgram dimensions: ",mel_dims)
    X = np.zeros((total_load, mel_dims[1], mel_dims[2], mel_dims[3]))   
    Y = np.zeros((total_load, nb_classes))  
    paths = []

    load_count = 0
    for idx, classname in enumerate(class_names):
        print("")
        this_Y = np.array(encode_class(classname,class_names) )
        this_Y = this_Y[np.newaxis,:]
        class_files = os.listdir(path+classname)
        n_files = len(class_files)
        n_load =  int(n_files * load_frac)
        printevery = 100

        file_list = class_files[0:n_load]
        for idx2, infilename in enumerate(file_list):          
            audio_path = path + classname + '/' + infilename
            if (0 == idx2 % printevery):
                print("\r Loading class ",idx+1,"/",nb_classes,": \'",classname,
                    "\', File ",idx2+1,"/", n_load,": ",audio_path,"                  ", 
                    sep="",end="")

            melgram = np.load(audio_path)
            
            X[count,:,:] = melgram
            Y[count,:] = this_Y
            paths.append(audio_path)     
            load_count += 1

    print("")
    if ( count != total_load ):  # check to make sure we loaded everything we thought we would
        raise Exception("Loaded "+str(load_count)+" files but was expecting "+str(total_load) )

    sr = 44100    # uh...probably shouldn't be hard-coding this. ??

    X, Y, paths = shuffle_XY_paths(X,Y,paths)  # mix up classes, & files within classes

    return X, Y, paths, class_names, sr
