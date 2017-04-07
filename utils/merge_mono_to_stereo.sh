# This will grab ALL files with .L.wav and .R.wav in the current directory
# and mix them (in pairs) to .STEREO.wav.  It will move the original files to a mono/ directory
mkdir mono
for file1 in ./*.L.wav; do 
  file2=`echo $file1 | sed 's_\(.*\).L.wav_\1.R.wav_'`;
  out=`echo $file1 | sed 's_\(.*\).L.wav_\1.STEREO.wav_'`;
  sox -MS "$file1" "$file2" "$out";
  mv "$file1" mono; mv "$file2" mono;
done
