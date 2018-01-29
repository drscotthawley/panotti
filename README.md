# Panotti: A Convolutional Neural Network classifier for multichannel audio waveforms

<img src="https://upload.wikimedia.org/wikipedia/commons/a/af/Panoteanen.jpg" alt="Panotti image" height="200">*(Image of large-eared Panotti people, Wikipedia)*

This is a version of the [audio-classifier-keras-cnn](https://github.com/drscotthawley/audio-classifier-keras-cnn) repo (which is a hack of **@keunwoochoi**'s compact_cnn code).  Difference with Panotti is, it has been generalized beyond mono audio, to include stereo or even more "channels."  And it's undergone many refinements.


## Dependencies (see requirements.txt)
* Python 3.5
* numpy
* keras
* tensorflow 
* librosa
* matplotlib
* h5py
* optional: pygame (for exampes/headgames.py), sox (SOund eXchange app; for generating test tones in binaural example)
* optional: sox ("Sound eXchange": command-line utility for examples/binaural. Install via "apt-get install sox")

## Demo
I'm not shipping this with any audio but you can generate some for the 'fake binaural' example (requires sox):

    cd examples
    ./binaural_setup.sh
    cd binaural
    ../../train_network.py

*Coming Soon: Demo via new Panotti Server!  Watch this space.*

## Quick Start
* Make a folder called `Samples/` and inside it create sub-folders with the names of each category you want to train on. Place your audio files in these sub-folders accordingly. 
* run `python preprocess_data.py`
* run `python train_network.py`
* run `python eval_network.py`  - This applies the trained network to the testing dataset and gives you accuracy reports.



## Data Preparation
### Data organization:
Sound files should go into a directory called `Samples/` that is local off wherever the scripts are being run.  Within `Samples`, you should have subdirectories which divide up the various classes.

Example: for the [IDMT-SMT-Audio-Effects database](https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/audio_effects.html), using their monophonic guitar audio clips...

    $ ls -F Samples/
    Chorus/  Distortion/  EQ/  FeedbackDelay/  Flanger/   NoFX/  Overdrive/  Phaser/  Reverb/  SlapbackDelay/
    Tremolo/  Vibrato/
    $
(Within each subdirectory of `Samples`, there are loads of .wav or .mp3 files that correspond to each of those classes.)

*"Is there any sample data that comes with this repo?"*  Not the data itself, but check out the `examples/` directory. ;-)


### Data augmentation & preprocessing:

#### (Optional) Augmentation:

The "augmentation" will [vary the speed, pitch, dynamics, etc.](https://bmcfee.github.io/papers/ismir2015_augmentation.pdf) of the sound files ("data") to try to "bootstrap" some extra data with which to train.  If you want to augment, then you'll run it as

`$ python augment_data.py <N>  Samples/*/*`

where *N* is how many augmented copies of each file you want it to create.  It will place all of these in the Samples/ directory with some kind of "_augX" appended to the filename (where X just counts the number of the augmented data files).
For augmentation it's assumed that all data files have the same length & sample rate.

#### (Required) Preprocessing:
When you preprocess, the data-loading will go *much* faster (e.g., 100 times faster) the next time you try to train the network. So, preprocess.

Preprocessing will pad the files with silence to fit the length to the length longest file and the number of channels to the file with the most channels. It will then generate mel-spectrograms of all data files, and create a "new version" of `Samples/` called `Preproc/`.

It will do an 80-20 split of the dataset, so within `Preproc/` will be the subdirectories `Train/` and `Test/`. These will have the same subdirectory names as `Samples/`, but all the .wav and .mp3 files will have ".npy" on the end now.  Datafiles will be randomly assigned to `Train/` or `Test/`, and there they shall remain.

To do the preprocessing you just run

`$ python preprocess_data.py`


## Training & Evaluating the Network
`$ python train_network.py`
That's all you need.  (I should add command-line arguments to adjust the layer size and number of layers...later.)

It will perform an 80-20 split of training vs. testing data, and give you some validation scores along the way.  

It's set to run for 2000 epochs, feel free to shorten that or just ^C out at some point.  It automatically does checkpointing by saving(/loading) the network weights via a new file `weights.hdf5`, so you can interrupt & resume the training if you need to.

After training, more diagnostics -- ROC curves, AUC -- can be obtained by running

`$ python eval_network.py`

*(Changing the `batch_size` variable between training and evaluation may not be a good idea.  It will probably screw up the Batch Normalization...but maybe you'll get luck.)*



## Results
On the [IDMT Audio Effects Database](https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/audio_effects.html) using the 20,000 monophonic guitar samples across 12 effects classes, this code achieved 99.7% accuracy and an AUC of 0.9999. Specifically, 11 mistakes were made out of about 4000 testing examples; 6 of those were for the 'Phaser' effect, 3 were for EQ, a couple elsewhere, and most of the classes had zero mistakes. (No augmentation was used.)

<a href="url"><img src="http://i.imgur.com/nWHqAWy.png" width="400"></a>

This accuracy is comparable to the [original 2010 study by Stein et al.](http://www.ece.rochester.edu/courses/ECE472/resources/Papers/Stein_2010.pdf), who used a Support Vector Machine.

This was achieved by running for 10 hours on [our workstation with an NVIDIA GTX1080 GPU](https://pcpartpicker.com/b/4xLD4D). 

**Future Work**: So, the Stein et al. data was a great test case, and now we have confidence to apply this method to other datasets.  I just got a new audio dataset I want to try, but it's small and will probably require some augmentation.  And, this github repo and the code itself are still not "cleaned up" for public reading/usage. 

<hr>
-- [@drscotthawley](https://drscotthawley.github.io)

