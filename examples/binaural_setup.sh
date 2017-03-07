#! /bin/sh
# Example of binaural source localization.  Generates 30-degree increments in azimuthal plane
# Author: Scott Hawley
#
# Requirements: sox  (SOund eXchange).  sudo apt-get install sox

echo "Generating 5-minute test signals..."
echo "      white noise..."; sox -r 44.1k -n whitenoise.wav synth 300 whitenoise
echo "      pink noise..."; sox -r 44.1k -n pinknoise.wav synth 300 pinknoise
echo "      brown noise..."; sox -r 44.1k -n brownnoise.wav synth 300 brownnoise
echo "      1k square wave..."; sox -r 44.1k -n square1k.wav synth 300 square 1000
echo "Binauralifying..."
python ~/panotti/utils/binauralify.py 12 *noise.wav square*.wav
echo "Splitting into 2-second clips..."
python ~/panotti/utils/split_audio.py -r 2 cl*/*.wav
echo "Moving clips to Samples/"
mkdir -p Samples
mv cl* Samples/ 
echo "Done."
echo ""
echo "Now run these..."
echo "~/panotti/preprocess_data.py"
echo "~/panotti/train_network.py"

