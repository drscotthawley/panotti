#! /usr/bin/env python
'''
Author: Scott H. Hawley

Based on paper,
A SOFTWARE FRAMEWORK FOR MUSICAL DATA AUGMENTATION
Brian McFee, Eric J. Humphrey, and Juan P. Bello
https://bmcfee.github.io/papers/ismir2015_augmentation.pdf

'''
from __future__ import print_function
import numpy as np
import librosa
from random import getrandbits
import sys, getopt, os
from multiprocessing import Pool
from functools import partial


def random_onoff():                # randomly turns on or off
    return bool(getrandbits(1))


# returns a list of augmented audio data, stereo or mono
def augment_audio(y, sr, n_augment = 0, allow_speedandpitch = True, allow_pitch = True,
    allow_speed = True, allow_dyn = True, allow_noise = True, allow_timeshift = True, tab="",quiet=False):

    mods = [y]                  # always returns the original as element zero
    length = y.shape[0]

    for i in range(n_augment):
        if not quiet:
            print(tab+"augment_audio: ",i+1,"of",n_augment)
        y_mod = y
        count_changes = 0

        # change speed and pitch together
        if (allow_speedandpitch) and random_onoff():
            length_change = np.random.uniform(low=0.9,high=1.1)
            speed_fac = 1.0  / length_change
            if not quiet:
                print(tab+"    resample length_change = ",length_change)
            tmp = np.interp(np.arange(0,len(y),speed_fac),np.arange(0,len(y)),y)
            #tmp = resample(y,int(length*lengt_fac))    # signal.resample is too slow
            minlen = min( y.shape[0], tmp.shape[0])     # keep same length as original;
            y_mod *= 0                                    # pad with zeros
            y_mod[0:minlen] = tmp[0:minlen]
            count_changes += 1

        # change pitch (w/o speed)
        if (allow_pitch) and random_onoff():
            bins_per_octave = 24        # pitch increments are quarter-steps
            pitch_pm = 4                                # +/- this many quarter steps
            pitch_change =  pitch_pm * 2*(np.random.uniform()-0.5)
            if not quiet:
                print(tab+"    pitch_change = ",pitch_change)
            y_mod = librosa.effects.pitch_shift(y, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
            count_changes += 1

        # change speed (w/o pitch),
        if (allow_speed) and random_onoff():
            speed_change = np.random.uniform(low=0.9,high=1.1)
            if not quiet:
                print(tab+"    speed_change = ",speed_change)
            tmp = librosa.effects.time_stretch(y_mod, speed_change)
            minlen = min( y.shape[0], tmp.shape[0])        # keep same length as original;
            y_mod *= 0                                    # pad with zeros
            y_mod[0:minlen] = tmp[0:minlen]
            count_changes += 1

        # change dynamic range
        if (allow_dyn) and random_onoff():
            dyn_change = np.random.uniform(low=0.5,high=1.1)  # change amplitude
            if not quiet:
                print(tab+"    dyn_change = ",dyn_change)
            y_mod = y_mod * dyn_change
            count_changes += 1

        # add noise
        if (allow_noise) and random_onoff():
            noise_amp = 0.005*np.random.uniform()*np.amax(y)
            if random_onoff():
                if not quiet:
                    print(tab+"    gaussian noise_amp = ",noise_amp)
                y_mod +=  noise_amp * np.random.normal(size=length)
            else:
                if not quiet:
                    print(tab+"    uniform noise_amp = ",noise_amp)
                y_mod +=  noise_amp * np.random.normal(size=length)
            count_changes += 1

        # shift in time forwards or backwards
        if (allow_timeshift) and random_onoff():
            timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
            if not quiet:
                print(tab+"    timeshift_fac = ",timeshift_fac)
            start = int(length * timeshift_fac)
            if (start > 0):
                y_mod = np.pad(y_mod,(start,0),mode='constant')[0:y_mod.shape[0]]
            else:
                y_mod = np.pad(y_mod,(0,-start),mode='constant')[0:y_mod.shape[0]]
            count_changes += 1

        # last-ditch effort to make sure we made a change (recursive/sloppy, but...works)
        if (0 == count_changes):
            if not quiet:
                print("No changes made to signal, trying again")
            mods.append(  augment_audio(y, sr, n_augment = 1, tab="      ", quiet=quiet)[1] )
        else:
            mods.append(y_mod)

    return mods


def augment_one_file(file_list, n_augment, quiet, file_index):

    infile = file_list[file_index]
    if os.path.isfile(infile):
        print("    Operating on file ",infile,", making ",n_augment," augmentations...",sep="")
        y, sr = librosa.load(infile, sr=None)
        mods = augment_audio(y, sr, n_augment=n_augment, quiet=quiet)
        for i in range(len(mods)-1):
            filename_no_ext = os.path.splitext(infile)[0]
            ext = os.path.splitext(infile)[1]
            outfile = filename_no_ext+"_aug"+str(i+1)+ext
            if not quiet:
                print("      mod = ",i+1,": saving file",outfile,"...")
            librosa.output.write_wav(outfile,mods[i+1],sr)
    else:
        print(" *** File",infile,"does not exist.  Skipping.")
    return

def main(args):
    np.random.seed(1)
    quiet = args.quiet

    if args.test:  # just testing the augment_audio.py on sample data
        y, sr = librosa.load(librosa.util.example_audio_file(),sr=None)
        librosa.output.write_wav("orig.wav",y,sr)
        mods = augment_audio(y, sr, n_augment=args.N, quiet=quiet)
        for i in range(len(mods)-1):
            outfile = "modded"+str(i+1)+".wav"
            librosa.output.write_wav(outfile,mods[i+1],sr)
        sys.exit()

    # read in every file on the list, augment it lots of times, output all those
    file_indices = tuple( range(len(args.file)) )
    cpu_count = os.cpu_count()
    pool = Pool(cpu_count)
    pool.map(partial(augment_one_file, args.file, args.N, args.quiet), file_indices)

    '''
    for infile in args.file:
        if os.path.isfile(infile):
            print("    Operating on file ",infile,", making ",args.N," augmentations...",sep="")
            y, sr = librosa.load(infile, sr=None)
            mods = augment_audio(y, sr, n_augment=args.N, quiet=quiet)
            for i in range(len(mods)-1):
                filename_no_ext = os.path.splitext(infile)[0]
                ext = os.path.splitext(infile)[1]
                outfile = filename_no_ext+"_aug"+str(i+1)+ext
                if not quiet:
                    print("      mod = ",i+1,": saving file",outfile,"...")
                librosa.output.write_wav(outfile,mods[i+1],sr)
        else:
            print(" *** File",infile,"does not exist.  Skipping.")
    '''
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Perform data augmentation')
    parser.add_argument("-q", "--quiet", help="quiet mode; reduce output",
                    action="store_true")
    parser.add_argument("-t", "--test", help="test on sample data (takes precedence over other args)", action="store_true")
    parser.add_argument("N", help="number of augmentations to generate",type=int)
    parser.add_argument('file', help="sound files to augment", nargs='*')
    args = parser.parse_args()
    main(args)
