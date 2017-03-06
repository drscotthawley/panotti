#! /usr/bin/env python

'''
 Set up the physionet 2016 challenge heart sounds database 
 See https://www.physionet.org/challenge/2016/

 Requirements:  
    The dataset, which it will try to download from
        https://www.physionet.org/physiobank/database/challenge/2016/training.zip
        https://www.physionet.org/physiobank/database/challenge/2016/validation.zip

TODO: Currently -- because it's easier right now -- the validation & training sets get 
   concatenated, shuffled, and then Panotti does its own 80-20 split off that.
   Gotta fix that so training stays training, and val stays val
'''

from __future__ import print_function
import os
from os.path import isfile
import pandas as pd
import shutil
import sys

trainpath = "Train/"
testpath = "Test/"
samplepath = "Samples/"

class_names = ('normal','abnormal')
set_names = ('training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f','validation')


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

            from subprocess import call
            print("   Uncompressing archive...",end="")
            if (tar):
                call(["tar","-zxf",filename])
            else:
                call(["unzip",filename])
            print(" done.")
    return



def make_dirs():
    if not os.path.exists(samplepath):
        os.mkdir( trainpath )
        os.mkdir( testpath )
        os.mkdir( samplepath )
        for classname in class_names:
            os.mkdir( trainpath+classname );   
            os.mkdir( testpath+classname );
            os.mkdir( samplepath+classname );
    return  


def slurp_file(filepath):
    line_list = open(filepath).readlines()
    return line_list


def main():
    make_dirs()

    download_if_missing()
    download_if_missing(dirname="validation", filename="validation.zip", 
    url="https://www.physionet.org/physiobank/database/challenge/2016/validation.zip")

    for set_idx, setname in enumerate(set_names):
     
        # TODO: come back and fix this...
        #if (setname != 'validation'):
        #    destpath = trainpath
        #else:
        #    destpath = testpath
        destpath = samplepath   # TODO: for now

        ref_filename = setname+'/'+'REFERENCE.csv'
        df = pd.read_csv(ref_filename, names=("file","code"))
        df["code"] = df["code"].replace([-1,1],['normal','abnormal'])   
        df["file"] = df["file"].replace([r"$"],[".wav"],regex=True)
        #print("\nFile ",ref_filename,":\n", df,sep="")
        for index, row in df.iterrows():
            this_file = row["file"]
            this_class = row["code"]
            src = setname+'/'+this_file
            dst = destpath + this_class + '/cl-'+this_class+"-"+ setname+"-"+this_file
            print("src, dst =  ",src,dst)
            shutil.copyfile(src,dst)

if __name__ == '__main__':
    main()
