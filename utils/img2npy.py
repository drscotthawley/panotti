#!/usr/bin/env python3

# Converts Image files format files to numpy, reading whatever image format is given
# does not remove old extension, i.e. new file is ____.jpeg.npy (no reason other than laziness)
# Runs in parallel

# TODO: if <file> is actually a directory, then it should greate a new <directory>.npy
#  AND *recursively* create a new version of original directory with all npy files replaced


from PIL import Image
import os
from multiprocessing import Pool
import numpy as np
from functools import partial


def convert_one_file(file_list,  mono, file_index):

    infile = file_list[file_index]
    out_format = 'npy'
    if os.path.isfile(infile):
        outfile = infile+"."+out_format
        print("    Operating on file",infile,", converting to ",outfile)
        img = Image.open(infile)
        arr = np.asarray(img)
        arr = np.reshape(arr, (1,1,arr.shape[0],arr.shape[1]))
        print('arr.shape = ',arr.shape)
        np.save(outfile,arr)
    return


def main(args):

    file_indices = tuple( range(len(args.file)) )

    cpu_count = os.cpu_count()
    pool = Pool(cpu_count)
    pool.map(partial(convert_one_file, args.file, args.mono), file_indices)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert image file to numpy array format')
    parser.add_argument("-m", "--mono", help="Save greyscale encoding for mono files (otherwise # color channels)",action="store_true")
    parser.add_argument('file', help=".npy file(s) to convert", nargs='+')
    args = parser.parse_args()
    main(args)
