#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 07:41:46 2018

@author: TheRealest

Windows utility for creating new file shortcuts based on data.json file 
"""

import json
import platform
import os

sys = platform.system()
print(sys)

if sys == "Windows":
    import winshell
    from win32com.client import Dispatch
    
json_file = input(str("Enter JSON file path (make sure it is in the same dir): \n"))
print(json_file)
path_txt = input("Enter the main path FOOL (make sure it is in the same dir):  \n")
print(path_txt)

main_directory = input("put new folder path here")
print(main_directory)
with open(json_file) as json_data:

   file_paths = []
   for folder, subs, files in os.walk(path_txt):
       for filename in files:
           file_paths.append(os.path.abspath(os.path.join(folder, filename)))

   # load JSON file
   d = json.load(json_data)
   items = d["items"]
   num_items = len(items)

   # hrs, mins, secs = tim.split(':')
   
   for x in range(num_items):
       # get old path name for every file to put in symbolic links
       old_path_name = file_paths[x]
       print(old_path_name)

       # make new filename for each element: name + id
       item = items[x]
       if sys == "Windows":
           old_path_name, tail = old_path_name.split('.')
           new_path_name = old_path_name + "_" + item["id"] + "." + tail
       else:
           old_path_name1, tail = old_path_name.split('.')
           new_path_name = old_path_name1 + "_" + item["id"] + "." + tail
           

       # askes for input and makes new folders(if needed)

       tags = item["tags"]
       tag = tags[0]
       if sys == "Windows":
           new_directory = main_directory + "\\" + tag
       else:
           new_directory = main_directory + "/" + tag
       
       subfolders = os.listdir(main_directory)
       if (tag not in subfolders):
           os.mkdir(new_directory)
           
            # create a symbolic link
       pre_dest = item["name"] + "_" + item["id"] + tail
       src = old_path_name
       if sys == "Windows":
           dst = "Desktop\\" + pre_dest
       else:
           pre_dest = item["name"] + "_" + item["id"] + "." + tail
           dst = new_directory + "/" + pre_dest

       tail_len = len(item["name"])
       old_path_name_temp = old_path_name[:-tail_len]
       
       os.symlink(src, dst)

       # Creates ShortCuts
       if sys == "Windows":
           desktop = winshell.desktop()
           path = os.path.join(new_directory, item["name"] + "_" + item["id"] + ".lnk")
           target = old_path_name + "." + tail
           wDir = new_path_name
       
      
           icon = r"C:\WINDOWS\system32\notepad.exe"

           shell = Dispatch('WScript.Shell')
           shortcut = shell.CreateShortCut(path)
           shortcut.Targetpath = target
           shortcut.WorkingDirectory = wDir
           shortcut.IconLocation = icon
           shortcut.save()
