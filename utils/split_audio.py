#! /usr/bin/env python
'''
split_audio.py
Author: Scott Hawley

Splits an audio file into clips of length "clip_dur" where is clip_dur the duration in seconds
Generated files get a "_sN" appended to their filename (before the extension) where N just 
counts the clip number

Works on mono, stereo,...arbitrary numbers of channels

If the sound file duration is not an integer multiple of clip_dur, then it will
generate the "last piece" padded with zeros on the end (so all clips have the same duration)
'''
from __future__ import print_function

import numpy as np
import librosa
import os

def fix_last_element(clip_list, intended_length, axis):
    #intended_length = clip_list[0].shape[axis]
    last_length = clip_list[-1].shape[axis]
    num_zeros = intended_length - last_length
    if (num_zeros > 0):
        ndims = clip_list[-1].ndim
        pad_list = []
        for i in range(ndims):
            if (axis == i):
                pad_list.append((0,num_zeros))
            else:
                pad_list.append((0,0))
        clip_list[-1] = np.pad( clip_list[-1], pad_list, mode='constant')

    clips = np.asarray( clip_list )
    return clips



def split_audio(file_list, clip_dur=2, remove_orig=True):
    for infile in file_list:
        if os.path.isfile(infile):
            print("     Splitting file: ",infile," into ",clip_dur,"-second clips... ",end="",sep="")
            signal, sr = librosa.load(infile, sr=None, mono=False)   # don't assume sr or mono
            if (1 == signal.ndim):
                print("       this is a mono file.  signal.shape = ",signal.shape)
            else:
                print("       this is a multi-channel file: signal.shape = ",signal.shape)
            axis = signal.ndim - 1
            signal_length = signal.shape[axis]
            stride = clip_dur * sr                             # length of clip in samples
            print("stride= ",stride,", signal_length = ", signal_length )                 
            indices = np.arange(stride, signal_length,stride)   # where to split
            clip_list = np.split( signal, indices, axis=axis)       # do the splitting
            intended_length = stride
            clips = fix_last_element(clip_list, intended_length, axis) # what to do with last clip

            sections = int( np.ceil( signal.shape[axis] / stride) ) # just to check 
            if( sections != clips.shape[0]):                        # just in case
                print("              **** Warning: sections = "+str(sections)+", but clips.shape[0] = "+str(clips.shape[0]) )
            ndigits = len(str(sections))   # find out # digits needed to print section #s
            for i in range(sections):
                clip = clips[i]
                filename_no_ext = os.path.splitext(infile)[0]
                ext = os.path.splitext(infile)[1]
                outfile = filename_no_ext+"_s"+'{num:{fill}{width}}'.format(num=i+1, fill='0', width=ndigits)+ext
                print("        Saving file",outfile)
                librosa.output.write_wav(outfile,clip,sr)

            if remove_orig:
                os.remove(infile)
        else:
            print("     *** File",infile,"does not exist.  Skipping.")
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Splits file(s) into multiple clips")
    parser.add_argument("-r", "--remove", help="remove original (long) file(s) after splitting",
                    action="store_true")
    parser.add_argument("clip_dur", help="duraction of each clip in seconds",type=int)
    parser.add_argument('file', help="file(s) to split", nargs='+')   
    args = parser.parse_args()
    split_audio(args.file, clip_dur=args.clip_dur, remove_orig=args.remove)
