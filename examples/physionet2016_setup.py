#! /usr/bin/env python

'''
 Set up the heart sounds database for the physionet 2016 challenge
 See https://www.physionet.org/challenge/2016/

NOTE: The PhysioNet 2016 Challenge dataset has a "validation" section, but their 
    'validation' data is INCLUDED in their training set. 
    So what we need do is regard the val set as val data, but DELETE instances of 
    val data that appear in the training set

Requirements:  
    The dataset, which the program will try to download from the following URLs:
        https://www.physionet.org/physiobank/database/challenge/2016/training.zip
        https://www.physionet.org/physiobank/database/challenge/2016/validation.zip
'''

from __future__ import print_function
import os
from os.path import isfile
import glob
import pandas as pd
import shutil
import sys
from subprocess import call
import glob
import subprocess

sys.path.insert(0, '../utils')
from split_audio import *

mainpath = "physionet2016"
samplepath = "Samples/"
trainpath = "Samples/Train/"
testpath = "Samples/Test/"

class_names = ('normal','abnormal')
set_names = ('training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f','validation') # no point getting their val set


# TODO: this never checks in case one of the operations fails
def download_if_missing(dirname="training-f", filename="training.zip", 
    url="https://www.physionet.org/physiobank/database/challenge/2016/training.zip",tar=False):

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


            print("   Uncompressing archive...",end="")
            if (tar):
                call(["tar","-zxf",filename])
            else:
                call(["unzip",filename])
            print(" done.")
    return


# create the directories we need
def make_dirs():
    if not os.path.exists(mainpath):
        print("Creating directory "+mainpath+"...")
        os.mkdir( mainpath )
        print("Changing directory to "+mainpath+"...")
        os.chdir(mainpath)
    if not os.path.exists(samplepath):
        os.mkdir( samplepath )
        os.mkdir( trainpath )
        os.mkdir( testpath )
        for classname in class_names:
            os.mkdir( trainpath+classname );   
            os.mkdir( testpath+classname );
    return  


# read in a text file as a list of lines
def slurp_file(filepath):
    line_list = open(filepath).readlines()
    return line_list


# any files with same names in both Test/ and Train/ get deleted
def delete_test_dupes(class_names):
    for classname in class_names:
        print("Deleting duplicates in "+testpath+classname+"/ & "+trainpath+classname+"...")
        trainlist = [os.path.basename(x) for x in glob.glob(trainpath+classname+"/*")]
        testlist = [os.path.basename(x) for x in glob.glob(testpath+classname+"/*")]
        both = set(trainlist).intersection(testlist)
        #print(" both = ",both)
        for filename in both:
            path = trainpath+classname+"/"+filename
            os.remove(path)
    return


def chopup_clips(class_names):
    dur = 5;
    print("Chopping audio files into",dur,"second clips...")
    file_list = glob.glob("Samples/*/*/*.wav")
    split_audio(file_list, clip_dur=dur, remove_orig=True)
    return


def main():
    make_dirs()
    download_if_missing()
    download_if_missing(dirname="validation", filename="validation.zip", 
           url="https://www.physionet.org/physiobank/database/challenge/2016/validation.zip")

    for set_idx, setname in enumerate(set_names):
     
        if (setname != 'validation'):
            destpath = trainpath
        else:
            destpath = testpath

        ref_filename = setname+'/'+'REFERENCE.csv'
        df = pd.read_csv(ref_filename, names=("file","code"))
        df["code"] = df["code"].replace([-1,1],['normal','abnormal'])   
        df["file"] = df["file"].replace([r"$"],[".wav"],regex=True)

        for index, row in df.iterrows():
            this_file = row["file"]
            this_class = row["code"]
            src = setname+'/'+this_file
            dst = destpath + this_class + '/cl-'+this_class+"-"+this_file
            print("src, dst =  ",src,dst)
            shutil.copyfile(src,dst)

    delete_test_dupes(class_names)
    chopup_clips(class_names)

    print("\nFINISHED.")
    print("Now run the following command:\n cd physionet2016; ../../preprocess_data.py --already") 
    return


if __name__ == '__main__':
    main()
