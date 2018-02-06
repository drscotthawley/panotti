#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 14:09:00 2018

@author: Braden Carei

Extracts metadata from Apple Loops .caf files
"""

import os
import subprocess
import errno
import shutil
from sys import version_info

if version_info >= (3, 4):
    from plistlib import loads as readPlistFromString
else:
    from plistlib import readPlistFromString
    
    
# Current working directory
CWD = os.getcwd()

UNKNOWN_FOLDER = "New Folder"

# Folders
folders = {
    UNKNOWN_FOLDER: []
    
}

for relative_path in os.listdir(CWD):
    full_path = os.path.join(CWD, relative_path)

    # If running from a script, skip self.
    #if os.path.samefile(full_path, __file__):
    #  continue

    try:
        output = subprocess.check_output(['mdls','-plist','-', full_path])
        
        
        
        plist = readPlistFromString(output)
        
        #print(output)
        
      
          # Skip invisible files
       # if plist['kMDItemFSInvisible'] is True:
         #   continue

        # Skip folders and symbolic links. Make an exception for Application bundles,
        # which in OSX are just folders with an .app extension.
#        if not os.path.isfile(full_path) or os.path.islink(full_path):
#            if 'com.apple.application' not in plist.get('kMDItemContentTypeTree', []):
#                continue
        
        if ('kMDItemMusicalInstrumentName' in plist) or ('kMDItemMusicalInstrumentCategory' in plist) or ('kMDItemMusicalGenre' in plist):
           
            
            tag1 = plist['kMDItemMusicalInstrumentName']
            
            tag2 = plist['kMDItemMusicalInstrumentCategory']
        
            tag3 = plist['kMDItemMusicalGenre']
                
            if tag1 not in folders:
                    folders[tag1] = []
                    
            if tag2 not in folders:
                    folders[tag2] = []
                    
            if tag3 not in folders:
                    folders[tag3] = []
                
            new_path = os.path.join(CWD, tag1, relative_path)
            folders[tag1].append([full_path, new_path])
        
            new_path = os.path.join(CWD, tag2, relative_path)
            folders[tag2].append([full_path, new_path])
            
            new_path = os.path.join(CWD, tag3, relative_path)
            folders[tag3].append([full_path, new_path])
        
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
