
These are command-line utilties for modifying audio files.

run "python <utilname.py> -h" to get usage info

...except for merge\_mono\_to\_stereo.sh, that's just a shell script. 'cat' that to see usage

Examples:

    python binauralify.py 12 whitenoise.wav

    python split_audio.py 2 e0*/*.wav

(Then maybe `ln -s e0* $HOME/panotti/Samples/` to set up the dataset)

    python concat_audio.py e0*/*_s*.wav

