#! /bin/bash
# Example of binaural source localization.  Generates 30-degree increments in azimuthal plane
# Author: Scott Hawley
#
# Requirements: sox  (SOund eXchange).
#
# NOTES:
#    1. The call to augment & binauralify will process ANY .wav files sitting in the binaural/ directory.
#       Thus if you want to add more test signals, create binaural/ first and add your signals,
#       then run this script
#    2. We split the signals into clips first, THEN augment, rather than augmenting first.  This gives our
#       data better "granularity" and makes it so the Test dataset won't (or is very unlikely to) contain
#       "exact" copies of what's in the Train dataset
#
# TODO: Need checks to see if commands succeed

# root directory of panotti
PANOTTI_HOME=../../

# duration of each generated signal, in seconds
SIGNAL_DUR=40

# sample rate of generated audio signals
RATE=44.1k

# duraton in seconds of each sound clip we get by chopping up signals
CLIP_DUR=2

# number of additional "augmented" versions of each clip to make
N_AUG=4

# number of discrete azimuthal angles to use when generating binaural data
N_AZ=12
let "deg = 360/$N_AZ"

# extension for audio files. "mp3"=less disk space, "wav"=better quality
EXT=wav


# Check if sox exists
command -v sox >/dev/null 2>&1 || { echo >&2 "I require sox but it's not installed. (Try 'sudo apt-get install sox'?) Aborting."; exit 1; }

# Little FYI notice
echo " "
echo "Example of binaural source localization.  Generates $deg-degree increments in azimuthal plane"
echo " "
echo "NOTICE: You are about to generate 14GB of data (12GB of audio and 2GB of spectrograms),"
echo "        and the whole setup process will take at least 10 minutes."
echo "        If you'd like less/more data (& time), abort this script and edit it, and"
echo "        decrease the values of SIGNAL_DUR, RATE, N_AUG, N_AZ and/or EXT"
echo " "
#read -p "Press ^C to abort now, or press enter to continue... "
read -rsn1 -p"Press ^C to abort now, or press any key to continue... ";echo

echo "Creating directory binaural/..."
mkdir -p binaural; cd binaural

echo "Generating signals..."
# e.g. sox -r 44.1k -n gen_white.wav synth 60 whitenoise
echo "      white noise..."; sox -r $RATE -n gen_white.$EXT synth $SIGNAL_DUR whitenoise
echo "      pink noise..."; sox -r $RATE -n gen_pink.$EXT synth $SIGNAL_DUR pinknoise
echo "      brown noise..."; sox -r $RATE -n gen_brown.$EXT synth $SIGNAL_DUR brownnoise
#echo "      tpdf noise..."; sox -r $RATE -n gen_tpdf.$EXT synth $SIGNAL_DUR tpdfnoise
echo "      sine sweep..."; sox -r $RATE -n gen_sinesw.$EXT synth $SIGNAL_DUR sine 20-20000
#echo "      square sweep..."; sox -r $RATE -n gen_squaresw.$EXT synth $SIGNAL_DUR square 50-5000
echo "      fmodded plucks..."; sox -n gen_pluck.$EXT  synth 1 pluck  synth 1 sine fmod 700-100 repeat $SIGNAL_DUR

echo "Splitting into $CLIP_DUR -second clips..."
python $PANOTTI_HOME/utils/split_audio.py -r $CLIP_DUR *.$EXT

echo "Augmenting dataset by a factor of $N_AUG..."
python $PANOTTI_HOME/utils/augment_audio.py -q $N_AUG *.$EXT

echo "Binauralifying into $N_AZ discrete azimuthal locations..."
python $PANOTTI_HOME/utils/binauralify.py -q $N_AZ *.$EXT

echo "Moving clips to Samples/"
mkdir -p Samples
mv class* Samples/

echo "Cleaing directory of all non-essential generated files"
/bin/rm -f gen*.$EXT compact.tar.Z

echo "Pre-procesing data (could take a while)..."
python $PANOTTI_HOME/preprocess_data.py --clean

samples_size=$(du -smh Samples | awk '{print $1}')

cd ..

echo " "
echo "FINISHED. Feel free to delete the Samples/ directory to free up $samples_size.  (Only Preproc/ is used in what follows.)"
echo "Now run the following..."
echo "  cd binaural; $PANOTTI_HOME/train_network.py"
echo " "
