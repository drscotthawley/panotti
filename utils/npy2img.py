#!/usr/bin/env python3

# Converts NumPy format files to image files, using whatever image format is specified
# default = jpeg
# does not remove old extension, i.e. new file is ____.npy.jpeg (no reason other than laziness)
# Runs in parallel

# TODO: if <file> is actually a directory, then it should greate a new <directory>.jpeg
#  AND *recursively* create a new version of original directory with all npy files replaced
#  by image files.
#   Intended usage:  ./npy2img Preproc/      (
#                    would generate Preproc.jpeg/Train/*.npy.jpeg, etc)
# ...'course, alternatively we could just let ../preprocess_data.py save as jpegs
#  and enable ../panotti/datautils.py, etc to read them.

from PIL import Image
import os
from multiprocessing import Pool
import numpy as np
from functools import partial


def convert_one_file(file_list, out_format, mono, file_index):
    infile = file_list[file_index]
    if os.path.isfile(infile):
        outfile = infile+"."+out_format
        print("    Operating on file",infile,", converting to ",outfile)
        arr = np.load(infile)
        arr = np.reshape(arr, (arr.shape[2],arr.shape[3]))
        print('arr.shape = ',arr.shape)
        if (mono):
            im = Image.fromarray(arr).convert('L')
        else:
            im = Image.fromarray(arr).convert('RGB')
        im.save(outfile)
    return


def main(args):

    file_indices = tuple( range(len(args.file)) )

    cpu_count = os.cpu_count()
    pool = Pool(cpu_count)
    pool.map(partial(convert_one_file, args.file, args.format, args.mono), file_indices)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert numpy array file to img format')
    parser.add_argument('--format', help="format of output image (jpeg, png, etc). Default = jpeg", type=str, default='jpeg')
    parser.add_argument("-m", "--mono", help="Use greyscale encoding for mono files (otherwise use RGB)",action="store_true")
    parser.add_argument('file', help=".npy file(s) to convert", nargs='+')
    args = parser.parse_args()
    main(args)
