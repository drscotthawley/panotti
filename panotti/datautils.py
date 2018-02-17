'''
datautils.py:  Just some routines that we use for moving data around
'''
from __future__ import print_function
import numpy as np
import librosa
import os
from os.path import isfile
from scipy.misc import imread, imsave

def listdir_nohidden(path):        # ignore hidden files
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

# class names are subdirectory names in Preproc/ directory
def get_class_names(path="Preproc/Train/", sort=True):
    if (sort):
        class_names = sorted(listdir_nohidden(path))     # sorted alphabetically for consistency with "ls" command
    else:
        class_names = listdir_nohidden(path)             # not in same order as "ls", because Python
    return class_names


def get_total_files(class_names, path="Preproc/Train/"):
    sum_total = 0
    for subdir in class_names:
        files = os.listdir(path+subdir)
        n_files = len(files)
        sum_total += n_files
    return sum_total


def load_melgram(file_path, dtype=np.float32):
    #auto-detect load method based on filename extension
    name, extension = os.path.splitext(file_path)
    if ('.npy' == extension):
        melgram = np.load(audio_path)
    elif ('.png' == extension) or ('.jpeg' == extension):
        arr = imread(file_path)
        melgram = np.reshape(arr, (1,1,arr.shape[0],arr.shape[1]))  # convert 2-d image
        melgram = np.flip(melgram, 0)     # we save images 'rightside up' but librosa internally presents them 'upside down'
    else:
        print("load_melgram: Error: unrecognized file extension '",extension,"' for file ",file_path,sep="")
    return melgram


def get_sample_dimensions(class_names, path='Preproc/Train/'):
    classname = class_names[0]
    audio_path = path + classname + '/'
    infilename = os.listdir(audio_path)[0]
    melgram = load_melgram(audio_path+infilename)
    print("   get_sample_dimensions: "+infilename+": melgram.shape = ",melgram.shape)
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
    #print("shuffle_XY_paths: Y.shape[0], len(paths) = ",Y.shape[0], len(paths))
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    newpaths = paths[:]
    for i in range(len(idx)):
        newX[i] = X[idx[i],:,:]
        newY[i] = Y[idx[i],:]
        newpaths[i] = paths[idx[i]]
    return newX, newY, newpaths

def make_melgram(mono_sig, sr):
    #melgram = librosa.logamplitude(librosa.feature.melspectrogram(mono_sig,  # latest librosa deprecated logamplitude in favor of amplitude_to_db
    #    sr=sr, n_mels=96),ref_power=1.0)[np.newaxis,np.newaxis,:,:]

    melgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(mono_sig,
        sr=sr, n_mels=96))[np.newaxis,np.newaxis,:,:]    # @keunwoochoi uses 96 mels

    '''
    # librosa docs also include a perceptual CQT example:
    CQT = librosa.cqt(mono_sig, sr=sr, fmin=librosa.note_to_hz('A1'))
    freqs = librosa.cqt_frequencies(CQT.shape[0], fmin=librosa.note_to_hz('A1'))
    perceptual_CQT = librosa.perceptual_weighting(CQT**2, freqs, ref=np.max)
    melgram = perceptual_CQT[np.newaxis,np.newaxis,:,:]
    '''
    return melgram


# turn multichannel audio as multiple melgram layers
def make_layered_melgram(signal, sr):
    if (signal.ndim == 1):      # given the way the preprocessing code is  now, this may not get called
        signal = np.reshape( signal, (1,signal.shape[0]))

    # get mel-spectrogram for each channel, and layer them into multi-dim array
    for channel in range(signal.shape[0]):
        melgram = make_melgram(signal[channel],sr)

        if (0 == channel):
            layers = melgram
        else:
            layers = np.append(layers,melgram,axis=1)  # we keep axis=0 free for keras batches
    return layers


def nearest_multiple( a, b ):   # returns number smaller than a, which is the nearest multiple of b
    return  int(a/b) * b


# can be used for test dataset as well
def build_dataset(path="Preproc/Train/", load_frac=1.0, batch_size=None):

    class_names = get_class_names(path=path)
    print("class_names = ",class_names)
    nb_classes = len(class_names)

    total_files = get_total_files(class_names, path=path)
    total_load = int(total_files * load_frac)
    if (batch_size is not None):                # keras gets particular: dataset size must be mult. of batch_size
        total_load = nearest_multiple( total_load, batch_size)
    print("       total files = ",total_files,", going to load total_load = ",total_load)

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
        n_load =  int(n_files * load_frac)   # n_load is how many files of THIS CLASS are expected to be loaded
        printevery = 100

        file_list = class_files[0:n_load]
        for idx2, infilename in enumerate(file_list):
            audio_path = path + classname + '/' + infilename
            if (0 == idx2 % printevery) or (idx2+1 == len(class_files)):
                print("\r Loading class ",idx+1,"/",nb_classes,": \'",classname,
                    "\', File ",idx2+1,"/", n_load,": ",audio_path,"                  ",
                    sep="",end="")

            #auto-detect load method based on filename extension
            melgram = load_melgram(audio_path)
            if (melgram.shape != mel_dims):
                print("\n\n    ERROR: mel_dims = ",mel_dims,", melgram.shape = ",melgram.shape)
            X[load_count,:,:] = melgram
            Y[load_count,:] = this_Y
            paths.append(audio_path)
            load_count += 1
            if (load_count >= total_load):   # Abort loading files after last even multiple of batch size
                break

    print("")
    if ( load_count != total_load ):  # check to make sure we loaded everything we thought we would
        raise Exception("Loaded "+str(load_count)+" files but was expecting "+str(total_load) )

    X, Y, paths = shuffle_XY_paths(X,Y,paths)  # mix up classes, & files within classes

    return X, Y, paths, class_names
