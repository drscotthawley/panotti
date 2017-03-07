#! /bin/sh
# Example of binaural source localization.  Generates 30-degree increments in azimuthal plane
# Author: Scott Hawley
#
# Requirements: sox  (SOund eXchange).  sudo apt-get install sox

echo "Generating 3-minute test signals..."
echo "      white noise..."; sox -r 44.1k -n whitenoise.wav synth 180 whitenoise
echo "      pink noise..."; sox -r 44.1k -n pinknoise.wav synth 180 pinknoise
echo "      brown noise..."; sox -r 44.1k -n brownnoise.wav synth 180 brownnoise
echo "      1k square wave..."; sox -r 44.1k -n square1k.wav synth 180 square 1000
echo "Binauralifying..."
python ../utils/binauralify.py 12 *noise.wav square*.wav
echo "Splitting into 2-second clips..."
python ../utils/split_audio.py -r 2 cl*/*.wav
echo "Moving clips to Samples/"
mkdir -p Samples
mv cl* Samples/ 
echo "Done."
echo ""
echo "Now run these..."
echo "../preprocess_data.py"
echo "../train_network.py"

