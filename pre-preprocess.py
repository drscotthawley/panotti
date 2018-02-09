import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *


def process_rawdata():
    files =  tqdm([f for f in os.listdir('raw_data') if f.endswith('.wav')])
    for f in files:
        files.set_description("Processing %s" % f)
        path = (os.path.join('raw_data', f))
        label = path.replace('audio', 'labels').replace('.wav', '.txt')
        label_times = np.array([np.float(s.split('\t')[0]) for s in open(label, 'r').readlines()])
        y, sr  = librosa.load(path, 44100)
        increment = int(44100 * .15)
        for i in range(0, len(y), increment):
            start = i/44100
            end = (i + increment)/44100
            clip = librosa.util.fix_length(y[i:i+increment], increment)
            outpath = 'Samples/no/%s-%d-%d.wav'%(f.replace(' ','-'), i, i+increment)
            if np.any(label_times[(label_times >= start) & (label_times <= end)]):
                outpath = outpath.replace('/no/', '/yes/')
            librosa.output.write_wav(outpath, clip, sr)


if __name__ == '__main__':
    process_rawdata()