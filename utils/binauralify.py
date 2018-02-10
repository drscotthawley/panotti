#!/usr/bin/env python
'''
binauralify.py
Send a mono file to a bunch of stereo files of binaural audio, for various source locations/angles

Author: Scott Hawley, modified from Adam Bard's code at http://web.uvic.ca/~adambard/ASP/project/

In the code's current state, it only varies in a horizontal plane ("azimuth"),
but you can easily modify it to go up/down ("elevation") as well.

For my ML runs, I'm trying n_az = 12, i.e. every 30 degrees.

Requirements:
    HRTF data: It will automatically try to download the HRTF measurements from
        http://sound.media.mit.edu/resources/KEMAR/compact.tar.Z
        and install them in the current directory
        If this fails, you can just install them yourself

'''
from __future__ import print_function
import numpy as np
from scipy import *
import librosa
import os, sys
from multiprocessing import Pool
from functools import partial


# TODO: this never checks in case one of the operations fails
def download_if_missing(dirname="compact", filename="compact.tar.Z",
    url="http://sound.media.mit.edu/resources/KEMAR/compact.tar.Z",tar=True):

    if not os.path.isdir(dirname):
        print("Directory \'"+dirname+"/\' not present.  Checking for compressed archive",filename)

        if not os.path.isfile(filename):
            import urllib
            print("   Compressed archive \'"+filename+"\' not present.  Downloading it...")

            if sys.version_info[0] >= 3:    # Python 3 and up
                from urllib.request import urlretrieve
            else:                           # Python 2
                from urllib import urlretrieve
            urlretrieve(url, filename)

        from subprocess import call
        print("   Uncompressing archive...",end="")
        if (tar):
            call(["tar","-zxf",filename])
        else:
            call(["unzip",filename])
        print(" done.")
    return



def readHRTF_file(name):   # from https://github.com/uncopenweb/3DSound
    '''Read the hrtf data from compact MATLAB format files'''
    r = np.fromfile( open(name, 'rb'), np.dtype('>i2'), 256)
    r.shape = (128,2)
    # half the rate to 22050 and scale to 0 -> 1
    r = r.astype(float)
    # should use a better filter here, this is a box lowering the sample rate from 44100 to 22050
    r = (r[0::2,:] + r[1::2,:]) / 65536
    return r



def setangles(elev, azimuth):
    elev = int(elev)
    azimuth = int(azimuth)

    #bring to multiple of ten
    if elev != 0:
        while elev%10 > 0:
            elev = elev + 1

    if elev > 90:
        elev = 90
    if elev < -40:
        elev = -40

    #Set increment of azimuth based on elevation
    if abs(elev) < 30:
        incr = 5
    elif abs(elev) == 30:
        incr = 6
    elif abs(elev) == 40:
        incr = 6.43
        opts = [0, 6, 13, 19, 26, 32, 29, 45, 51, 58, 64, 71, 77, 84, 90, 96, 103, 109, 116, 122, 129, 135, 141, 148, 154, 161, 167, 174, 180]
    elif elev == 50:
        incr = 8
    elif elev == 60:
        incr = 10
    elif elev == 70:
        incr = 15
    elif elev == 80:
        incr = 30
    elif elev == 90:
        incr = 0
        azimuth = 0
    flip = False

    #bring into [-pi,pi]
    while azimuth > 180:
        azimuth = azimuth - 180
    while azimuth < -180:
        azimuth = azimuth + 180

    #check if we need to flip left and right.
    if azimuth < 0:
        azimuth = abs(azimuth)
        flip = True

    if abs(elev) == 40:
        incr = 6.43
        num = incr
        while azimuth > num:
            num = num + incr

        azimuth = str(int(round(num)))
        #special case for non-integer increment

    elif azimuth != 0:
        while azimuth % incr > 0:
            azimuth = azimuth + 1

    if int(azimuth) < 100:
        azimuth = "0" + str(int(azimuth))

    if int(azimuth) < 10:
        azimuth = "00"+ str(int(azimuth))

    return elev, azimuth, flip



def read(elev, azimuth, N=128):
    """ Accepts elev and azimuth in degrees, and returns closest impulse response and
    transfer function to that combination from compact KEMAR HRTF measurements"""

    elev, azimuth, flip = setangles(elev, azimuth)
    filename = "compact/elev"+str(elev)+"/H"+str(elev)+"e"+str(azimuth)+"a.dat"
    h_t = readHRTF_file(filename)

    h_t_l = transpose(transpose(h_t)[0])
    h_t_r = transpose(transpose(h_t)[1])
    if flip:
        return h_t_r, h_t_l
    return h_t_l, h_t_r


# this is the code that takes a mono signal and projects it to one 'location'
def project(sig, elev, azimuth):
    h_t_l, h_t_r = read(elev, azimuth)

    Hw_l = fft(h_t_l, len(sig))
    Hw_r = fft(h_t_r, len(sig))

    f_audio = fft(sig)
    f_audio_l = Hw_l*f_audio
    f_audio_r = Hw_r*f_audio
    t_audio_l = ifft(f_audio_l, len(sig)).real
    t_audio_r = ifft(f_audio_r, len(sig)).real
    return t_audio_l, t_audio_r


# Bard's code for spining one audio clip around.  I'm not using it but, still leaving in
def path(t_sig, infile, sr, start, end, duration=0, window_size=1024, fs=44100):
    """ Moves a sound from start to end positions over duration (Seconds)"""
    M = (fs/2.) / window_size
    w = r_[:fs/2.:M]
    N = len(w)

    window = hamming(N)   # this is for overlapping multiple 'locations' into one long clip
    #window = r_[:window_size])

    i = 1
    elev = start[0]
    elev_end = end[0]

    if duration == 0:
        duration = len(t_sig)/fs

    azimuth = start[1]
    azimuth_end = end[1]
    N_steps = int(len(t_sig) * 2 / window_size)
    elev_delta = float((elev_end - elev) / float(N_steps)) #deg/half-window
    azimuth_delta = float((azimuth_end - azimuth) / float(N_steps))

    output_l = zeros( len(t_sig) )
    output_r = zeros( len(t_sig) )

    outpath = "./"
    while i*(window_size) < len(t_sig):
        ind_min = int( (i-1)*window_size)
        ind_max = int( (i)*window_size )
        t_sig_w = t_sig[ind_min:ind_max] * window
        t_output_l, t_output_r = project(t_sig_w, elev, azimuth)

        output_l[ind_min:ind_max] += t_output_l
        output_r[ind_min:ind_max] += t_output_r

        elev = elev + elev_delta
        azimuth = azimuth + azimuth_delta

        i = i + 0.5

    return output_l, output_r



# takes a single file's worth of mono and generates multiple files at different locations
def project_multi(mono_sig, infile, sr, start, end, steps, quiet=False):
    elev_bgn = start[0]
    az_bgn = start[1]
    elev_end = end[0]
    az_end = end[1]
    steps_elev = steps[0]
    steps_az = steps[1]
    elev_delta = float((elev_end - elev_bgn) / float(steps_elev)) #deg/half-window
    az_delta = float((az_end - az_bgn) / float(steps_az))

    total = steps_elev * steps_az

    outpath = "./"
    count = 0
    for i in range(steps_elev):
        elev = elev_bgn + i * elev_delta
        for j in range(steps_az):
            count += 1
            az = az_bgn + j * az_delta

            stereo_l, stereo_r = project(mono_sig, elev, az)
            stereo_sig = np.vstack( (stereo_l, stereo_r))

            # save to file
            classname = "class"+str(count).zfill(2)+"-"+"a"+str(az)#+"e"+str(elev)+ # can add elevation if you want
            if not os.path.exists(outpath+classname):
                os.mkdir( outpath+classname)
            filename_no_ext = os.path.splitext(infile)[0]
            ext = os.path.splitext(infile)[1]
            outfile = classname+'/'+filename_no_ext+'_'+classname+ext
            if not quiet:
                print("\r    elev, az = ",elev,az,", outfile = ",outfile,"             ")
            librosa.output.write_wav(outfile,stereo_sig,sr)
    if not quiet:
        print("")
    return

def binauralify_one_file(file_list, n_az, quiet,file_index):

    infile = file_list[file_index]
    if os.path.isfile(infile):
        print("   Binauralifying file",infile,"...")
        mono, sr = librosa.load(infile, sr=None)   # librosa naturally makes mono from stereo btw
        project_multi(mono, infile, sr, (0,-180), (0, 180), (1,n_az), quiet=quiet)
    else:
        print("   *** File",infile,"does not exist.  Skipping.")
    return

def main(args):

    download_if_missing()                # make sure we've got the hrtf data we need
    cpu_count = os.cpu_count()
    print("",cpu_count,"CPUs detected: Parallel execution across",cpu_count,"CPUs")
    file_indices = tuple( range(len(args.file)) )

    pool = Pool(cpu_count)
    pool.map(partial(binauralify_one_file, args.file, args.n_az, args.quiet), file_indices)
    '''
    for infile in args.file:
        if os.path.isfile(infile):
            print("   Binauralifying file",infile,"...")
            mono, sr = librosa.load(infile, sr=None)   # librosa naturally makes mono from stereo btw
            project_multi(mono, infile, sr, (0,-180), (0, 180), (1,args.n_az), quiet=args.quiet)
        else:
            print("   *** File",infile,"does not exist.  Skipping.")
    '''


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="binauralify: generate binaural samples from mono audio")
#   parser.add_argument("n_elev", help="number of discrete poitions of elevation",type=int)
    parser.add_argument("-q", "--quiet", help="quiet mode; reduce output",
                    action="store_true")
    parser.add_argument("n_az", help="number of discrete poitions of azimuth",type=int)
    parser.add_argument('file', help="mono wav file(s) to binauralify", nargs='+')
    args = parser.parse_args()
    main(args)
