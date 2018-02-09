import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

from panotti.models import *
from panotti.datautils import *
from predict_class import predict_one


model = load_model('weights.hdf5')

def test():
    class_names = get_class_names('Preproc/Test/')

    plt.figure(figsize=(12, 8))
    files =  tqdm([f for f in os.listdir('raw_data') if f.endswith('.wav')])
    for f in files:
        files.set_description("Processing %s" % f)
        path = (os.path.join('raw_data', f))
        increment = int(44100 * .15)
        y, sr  = librosa.load(path, 44100)
        outfile = open(os.path.join('raw_data/generated/', f.replace('.wav', '.txt')), 'w')
        for i in range(4410, len(y), increment):
            start = i/44100
            end = (i + increment)/44100
            half = increment/44100
            clip = librosa.util.fix_length(y[i:i+increment], increment)
            outpath = 'Samples/no/%s-%d-%d.wav'%(f.replace(' ','-'), i, i+increment)
            
            # if class_names[np.argmax(predict_one(clip, sr, model))] == 'yes':
            ons = librosa.onset.onset_detect(y=clip, sr=44100, units='samples')/44100.
            for o in ons:
                outfile.write('\t'.join([str(start+o), str(start+o), '\n']))
        
        outfile.close()
              

if __name__ == '__main__':
    test()