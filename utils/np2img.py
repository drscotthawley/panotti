#!/usr/bin/env python3

# Converts NumPy format files to image files, using whatever image format is specified
# default = jpeg
# does not remove old extension, i.e. new file is ____.npy.jpeg (no reason other than laziness)
# Runs in parallel

# TODO: if <file> is actually a directory, then it should greate a new <directory>.jpeg
#  AND *recursively* create a new version of original directory with all npy files replaced
#  by image files.
#   Intended usage:  ./np2img Preproc/      (
#                    would generate Preproc.jpeg/Train/*.npy.jpeg, etc)
# ...'course, alternatively we could just let ../preprocess_data.py save as jpegs
#  and enable ../panotti/datautils.py, etc to read them.

import os
#from multiprocessing.pool import ThreadPool as Pool
from multiprocessing.pool import Pool
import numpy as np
from functools import partial
from scipy.misc import imsave

def convert_one_file(file_list, out_format, mono, file_index):
    infile = file_list[file_index]
    if os.path.isfile(infile):
        basename, extension = os.path.splitext(infile)
        if ('.npz' == extension) or ('.npy' == extension):
            outfile = basename+"."+out_format
            print("    Operating on file",infile,", converting to ",outfile)
            if ('.npz' == extension):
                with np.load(infile) as data:
                    arr = data['melgram']
            else:
                    arr = np.load(infile)
            channels = arr.shape[-1]
            if (channels <= 4):
                #arr = np.reshape(arr, (arr.shape[2],arr.shape[3]))
                arr = np.moveaxis(arr, 1, 3).squeeze()      # we use the 'channels_first' in tensorflow, but images have channels_first. squeeze removes unit-size axes
                arr = np.flip(arr, 0)    # flip spectrogram image right-side-up before saving, for easier viewing
                if (2 == channels): # special case: 1=greyscale, 3=RGB, 4=RGBA, ..no 2.  so...?
                    # pad a channel of zeros (for blue) and you'll just be stuck with it forever. so channels will =3
                    b = np.zeros((arr.shape[0], layers.shape[1], 3))  # 3-channel array of zeros
                    b[:,:,:-1] = arr                          # fill the zeros on the 1st 2 channels
                    imsave(outfile, b, format=out_format)
                else:
                    imsave(outfile, arr, format=out_format)
            else:
                print("   Skipping file",infile,": Channels > 4. Not representable in jpeg or png format.")
        else:
            print("    Skipping file",infile,": not numpy format")
    else:
        print("    Skipping file",infile,": file not found")

    return


def main(args):
    # farm out the list of files across multiple cpus
    file_indices = tuple( range(len(args.file)) )
    cpu_count = os.cpu_count()
    pool = Pool(cpu_count)
    pool.map(partial(convert_one_file, args.file, args.format, args.mono), file_indices)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert numpy array file to image format')
    parser.add_argument('--format', help="format of output image (jpeg, png, etc). Default = png", type=str, default='png')
    parser.add_argument("-m", "--mono", help="Use greyscale encoding for mono files (otherwise use RGB)",action="store_true")
    parser.add_argument('file', help=".npy file(s) to convert", nargs='+')
    args = parser.parse_args()
    main(args)
