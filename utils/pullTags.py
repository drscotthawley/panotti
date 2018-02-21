#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 14:09:00 2018

@authors: Braden Carei (initial), Scott Hawley (revisions/parallelism)

Extracts metadata from Apple Loops .caf files
"""

import os
import subprocess
import errno
import shutil
from sys import version_info
import glob
from multiprocessing import Pool
from functools import partial
if version_info >= (3, 4):
    from plistlib import loads as readPlistFromString
else:
    from plistlib import readPlistFromString
import re


keys_to_use = ['kMDItemMusicalInstrumentName',
               'kMDItemMusicalInstrumentCategory',
               'kMDItemMusicalGenre',
               'kMDItemAppleLoopDescriptors',
               'kMDItemTimeSignature']


def make_dir(directory):  # makes a directory if it doesn't exist
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

def make_link(source, link_name):  # makes a symbolic link (unix) or shortcut (Windows):
    # TODO: how to do on windows?
    try:
        os.stat(link_name)
    except:
        os.symlink(source, link_name)


def pullTags_one_file(file_list, new_main_folder, file_index):
    # We actually pull the tags and create the directories & symbolic links as we go,
    #   and let the OS/filesystem handle 'collisions' if more than one process is trying to create the same thing
    infile = file_list[file_index]
    #print('infile = ',infile)
    output = subprocess.check_output(['mdls','-plist','-', infile])  # mdls is an Apple command-line utility
    plist = readPlistFromString(output)
    #print(plist)
    print_every = 100
    if (0 == file_index % print_every):
        print("pullTags:  File ",file_index,"/",len(file_list),": ",infile,sep="")
    for key in keys_to_use:
        #print(" Checking key = ",key)
        try:
            tags = plist[key]
            if tags:                       # guard against blank tags
                if isinstance(tags, str):
                    tags = [tags]      # just make it a length-1 list
                #print("               key = ",key,",   tags (list):")
                for tag in tags:
                    tag = re.sub('[^a-zA-Z\d\ ]|( ){2,}','_',tag )  # whitelist only certain characters. e.g. "4/4"->"4_4"
                    #tag = tag.replace("/", "-").replace(";", "-").replace("\\", "-").replace(" ", "_").replace
                    #print("                                                      [",tag,']',sep="")
                    if tag:              # guard against blank tags
                        new_folder = new_main_folder+'/'+tag
                        make_dir(new_folder)
                        link_name = new_folder+'/'+os.path.basename(infile)
                        make_link(infile, link_name)
        except:
            #print("Key error: File",infile,"doesn't contain key",key)
            pass
    return


def hawleys_way(args):
    search_dirs = args.dir
    new_main_folder='Samples/'
    make_dir(new_main_folder)

    print("Searching for .caf files in starting directories ",search_dirs)
    # Rescursively search for .caf files starting in various starting directories
    file_list = []
    for search_dir in search_dirs:
        for filename in glob.iglob(search_dir+'/**/*.caf', recursive=True):
            file_list.append(filename)

    # farm out the pulling of tags to many files simultaneously across all processors
    file_indices = tuple( range(len(file_list)) )
    cpu_count = os.cpu_count()
    if (False):                              # simple single-processor execution for testing
        for file_index in file_indices:
            pullTags_one_file(file_list,new_main_folder,file_index)
    else:
        pool = Pool(cpu_count)
        print("Mapping to",cpu_count,"processes")
        pool.map(partial(pullTags_one_file, file_list, new_main_folder), file_indices)
    return



def bradens_way():
    # Current working directory
    CWD = os.getcwd()

    UNKNOWN_FOLDER = "New Folder"

    # Folders
    folders = {
        UNKNOWN_FOLDER: []
    }

    # get a list of all the names of new folders
    for relative_path in os.listdir(CWD):
        full_path = os.path.join(CWD, relative_path)

        # If running from a script, skip self.
        #if os.path.samefile(full_path, __file__):
        #  continue

        try:
            output = subprocess.check_output(['mdls','-plist','-', full_path])

            plist = readPlistFromString(output)

            if (('kMDItemMusicalInstrumentName' in plist) or ('kMDItemMusicalInstrumentCategory' in plist)
                or ('kMDItemMusicalGenre' in plist) or ('kMDItemAppleLoopDescriptors' in plist)):

                tag1 = plist['kMDItemMusicalInstrumentName']
                tag2 = plist['kMDItemMusicalInstrumentCategory']
                tag3 = plist['kMDItemMusicalGenre']
                tag4 = plist['kMDItemAppleLoopDescriptors']

                if tag1 not in folders:
                        folders[tag1] = []

                if tag2 not in folders:
                        folders[tag2] = []

                if tag3 not in folders:
                        folders[tag3] = []

                if tag4 not in folders:
                        folders[tag4] = []

                new_path = os.path.join(CWD, tag1, relative_path)
                folders[tag1].append([full_path, new_path])

                new_path = os.path.join(CWD, tag2, relative_path)
                folders[tag2].append([full_path, new_path])

                new_path = os.path.join(CWD, tag3, relative_path)
                folders[tag3].append([full_path, new_path])

                new_path = os.path.join(CWD, tag4, relative_path)
                folders[tag4].append([full_path, new_path])

            else:
                # Move file to the catch-all folder
                new_path = os.path.join(CWD, UNKNOWN_FOLDER, relative_path)
                folders[UNKNOWN_FOLDER].append([full_path, new_path])

        except:
            print("Could not process: %s" % full_path)

    #Create folders and move files
    for (folder, tuples) in folders.items():
        folder_path = os.path.join(CWD, folder)
                #print(folder_path)
                # Create folder if it does not exist
        try:
            os.makedirs(folder_path)
            print("Created folder: %s" % folder_path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        # Move files
        for t in tuples:
            try:
                shutil.copy(t[0],t[1])
                shutil.copy(t[0],t[2])
                shutil.copy(t[0],t[3])
                shutil.copy(t[0],t[4])
            except:
                print("Could not move file: %s" % t[0])
    return



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pull metadata tags from Apple Loops .caf files')
    parser.add_argument('dir', help="directory/ies to search in", nargs='*', default=['/Library/Audio'])
    args = parser.parse_args()
    hawleys_way(args)
