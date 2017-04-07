#! /usr/bin/env python
'''
concat_audio.py
Author: Scott Hawley

Joins a bunch of audio files into one big file.
Filename of output is whatever the last filename is + "_Long" 

Works on mono, stereo,...arbitrary numbers of channels

'''
from __future__ import print_function

import numpy as np
import librosa
import os


def main(args):
	count = 0
	for infile in args.file:
		count += 1
		if os.path.isfile(infile):
			print("Input file: ",infile,"... ",end="",sep="")
			signal, sr = librosa.load(infile, sr=None, mono=False)   # don't assume sr or mono
			if (1 == signal.ndim):
				print("this is a mono file.  signal.shape = ",signal.shape)
			else:
				print("this is a multi-channel file: signal.shape = ",signal.shape)
			axis = signal.ndim - 1

			if (1 == count):
				long_clip = signal
			else:
				long_clip = np.concatenate((long_clip,signal),axis=axis)

		else:
			print(" *** File",infile,"does not exist.  Skipping.")

	filename_no_ext = os.path.splitext(infile)[0]
	ext = '.wav'  # os.path.splitext(infile)[1]
	outfile = filename_no_ext+"_Long"+ext
	print("Saving file",outfile)
	librosa.output.write_wav(outfile,long_clip,sr)
	return



if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="concat_audio: concatenates multiple files into one")
	parser.add_argument('file', help="file(s) to concat", nargs='+')   
	args = parser.parse_args()
	main(args)

