#! /bin/sh
# Example of binaural source localization.  Generates 30-degree increments in azimuthal plane
# Author: Scott Hawley
#
# Requirements: sox  (SOund eXchange).  sudo apt-get install sox

echo "Creating directory binaural/..."
mkdir -p binaural; cd binaural
NOISE_DUR=90
RATE=44.1k
echo "Generating test signals ($NOISE_DUR seconds long each)..."
echo "      white noise..."; sox -r $RATE -n whitenoise.wav synth $NOISE_DUR whitenoise
echo "      pink noise..."; sox -r $RATE -n pinknoise.wav synth $NOISE_DUR pinknoise
echo "      brown noise..."; sox -r $RATE -n brownnoise.wav synth $NOISE_DUR brownnoise
echo "      1k square wave..."; sox -r $RATE -n square1k.wav synth $NOISE_DUR square 1000
echo "Binauralifying..."
python ../../utils/binauralify.py 12 *noise.wav square*.wav
CLIP_DUR=2
echo "Splitting into $CLIP_DUR -second clips..."
python ../../utils/split_audio.py -r $CLIP_DUR cl*/*.wav
echo "Moving clips to Samples/"
mkdir -p Samples
mv cl* Samples/ 
echo "Pre-procesing data (could take a while)..."
python ../../preprocess_data.py
cd ..
echo " "
echo "FINISHED.  Now run this (and let it run for about 10 epochs)..."
echo "   cd binaural; ../../train_network.py"
echo " "
