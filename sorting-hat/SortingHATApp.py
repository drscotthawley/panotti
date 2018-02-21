#!/usr/bin/env python3
'''
SortingHatApp  - Desktop version of Sorting H.A.T. program.

Author: Scott H. Hawley @drscotthawley

TODO:
    - Speed up sort. It's fast enough to run on a laptop, so we do, but it could be even faster.
    - Should create a "ButtonBarAndStatus" class that can be reused multiple times

Requirements:
    - Panotti on both local machine ("laptop") and server
    - Valid SSH login on server.  TODO: generalize from SSH to HTTP, or some AWS and/or Google Cloud APis
    $ pip install kivy kivy-garden scandir functools paramiko git+https://github.com/jbardin/scp.py.git
    $ garden install filebrowser scrolllabel

============
'''
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add panotti to Python path
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.progressbar import ProgressBar
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty
from kivy.garden.filebrowser import FileBrowser
from kivy.config import Config
from settingsjson import settings_json
import time
import subprocess
import threading
from scp_upload import scp_upload
from functools import partial
import re
from pathlib import Path
from panotti.datautils import *
import csv

PANOTTI_HOME = os.path.expanduser("~")+"/panotti/"
PREPROC_DIR = "Preproc"
ARCHIVE_NAME = PREPROC_DIR+".tar.gz"
REMOTE_RUN_DIR = "~/SortingHatRuns"
SORTED_DIR = "Sorted"

def count_files(folder, skip_csv=True):
    total = 0
    for root, dirs, files in os.walk(folder):
        files = [f for f in files if not f[0] == '.']     # ignore hidden files
        if (skip_csv):
            files = [f for f in files if not '.csv'==os.path.splitext(f)[1]]     # ignore csv
        dirs[:] = [d for d in dirs if not d[0] == '.']    # ignore hidden directories
        total += len(files)
    return total


def file_len(filename):
    count = 0
    for l in open(filename):
        if ('#' != l[0]):   # ignore comments
            count +=1
    return count

def folder_size(path):   # get bytes
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += folder_size(entry.path)
    return total


# check to see if a thread is still running, and if not call completion
def check_for_completion(thread, completion, t):
    if not thread.isAlive():
        completion()
        return False    # cancels any Kivy scheduler
    return True         # keep the scheduler going

# Generic utility to run things ('threads' or 'processes') in the background
#   routine can be a string (for shell command) or another python function
#   progress and oompletion are callbacks, i.e. should point to other functions
#  Note: the progress callback is actually what handles the completion
def spawn(routine, progress=None, interval=0.1, completion=None):

    def runProcInThread(cmd, completion_in_thread):  # cmd is a string for a shell command, usually completion_in_thread will be None
        #TODO: remove security risk associated w/ shell=True flag below, by parsing cmd string for pipe symbols and semicolons,
        #          generating multiple subprocess calls as appropriate
        proc = subprocess.Popen(cmd, shell=True)     # SECURITY RISK: cmd better not have any semicolons you don't want in there. use whitelist_string on any user-supplied string inputs
        proc.wait()                          # make sure it's finished
        if (completion_in_thread is not None):
            completion_in_thread()
        return

    # spawn the process/thread
    if isinstance(routine, str):  # routine is a string, spawn a shell command
        thread = threading.Thread(target=runProcInThread, args=(routine,None) )
    elif callable(routine):       # routine is another Python function
        thread = threading.Thread(target=routine)
    else:
        print("*** spawn: Error: routine = ",routine," is neither string nor callable. Exiting.")
        return           # Leave
    thread.start()

    # schedule a Kivy clock event to repeatedly call the progress-query routine (& check for completion of process)
    #   Note: these routines that get called by Clock.schedule_interval need to return False in order to kill the schedule
    if (progress is not None):
        progress_clock_sched = Clock.schedule_interval(partial(progress, thread, completion), interval)
    elif (completion is not None):          # no progress per se, but still wait on completion
        completion_clock_sched = Clock.schedule_interval(partial(check_for_completion, thread, completion), interval)

    return

def whitelist_string(string):
    string = re.sub('[^a-zA-Z\d\ ]|( ){2,}','-',string )
    return string

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class ScrollableLabel(ScrollView):
    text = StringProperty('')


#============== Main Widget =================

class SHPanels(TabbedPanel):
    def __init__(self, **kwargs):
        super(SHPanels, self).__init__(**kwargs)
        Clock.schedule_once(self.on_tab_width, 0.1)
        Window.bind(on_dropfile=self._on_file_drop)
        Clock.schedule_once(partial(self.switch, self.ids['buildPanel']), 0)
        self.last_drop_time = time.time()-10000
        self.progress_events = {}
        self.samplesDir = ''
        self.parentDir = ''
        self.indexFile = ''
        self.ready_to_preproc = False
        self.ready_to_upload = False
        self.ready_to_train = False
        self.sortFileList = []
        self.sortWeightsFile = ''
        self.autoRunning = False

    def switch(self, tab, *args):   # panel switching
        self.switch_to(tab)

    #----------- Testing stuffffff-------
        self.count = 0
        self.maxval = 0

    def finished(self):
        print("\n\n*** Finished.")

    def count_to(self, maxval):   # this will be our test process
        print("Begin the counting, maxval = ",maxval)
        self.maxval = maxval
        for self.count in range(maxval):
            pass

    def test_spawn(self):
        spawn(partial(self.count_to,50000000), progress=self.progress_display, completion=self.finished)
        #time.sleep(10)  # just keep the program from terminating so we can see what happens!

    #-------------- Generic Utilities for testing / mock-up  ------------
    def done_fake(self):
        self.ids['trainStatus'].text = "Server is up and running."
        self.ids['serverButton'].state = "down"
        return False

    def progress_display(self, barname, thread, completion, t):
        percent = int((self.count+1) / self.maxval * 100)
        self.ids[barname].value = percent
        # Test for completion:
        if not thread.isAlive():         # if the thread has completed
            if (percent >=100):          # Yay
                Clock.schedule_once(lambda dt: completion(), 0.5)   # go on to the final state
            else:
                print("\nError: Process died but progress < 100% ")
            return False                 # Either way, cancel the Kivy clock schedule
        return True                      # keep the progress-checker rolling

    def count_up(self,maxval=100):     # some 'fake' task for testing purposes
        self.maxval = maxval
        for self.count in range(maxval):
            time.sleep(0.01)


    def start_prog_anim(self, barname):
        self.ready_to_preproc = ('No folder selected' != self.ids['samplesDir'].text)
        self.ready_to_upload = (self.ids['preprocProgress'].value >= 100) and (self.ids['serverProgress'].value >= 100)
        self.ready_to_train = (self.ids['uploadProgress'].value >= 100) and (self.ready_to_upload)
        if (('serverProgress' == barname) or
            (('preprocProgress' == barname) and self.ready_to_preproc) or
            (('uploadProgress' == barname) and self.ready_to_upload) or
            (('trainProgress' == barname) and self.ready_to_train) ):
            #self.ids[barname].value = 0
            spawn(self.count_up, progress=partial(self.progress_display,barname),interval=0.1, completion=self.done_fake)


    #-------------- File/Folder Selection --------------

    # TODO: does this get used?  copied code from elsewhere
    def open(self, path, filename):
        with open(os.path.join(path, filename[0])) as f:
            print(f.read())

    # TODO: does this get used?  copied code from elsewhere
    def selected(self, filename):
        print("selected: %s" % filename[0])

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self,origin_button):
        content = LoadDialog(load=partial(self.load,origin_button), cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    # either by drag-dropping or by using popup dialog, we now have one or more filenames/directory names
    def got_filenames(self, filenames, origin_button):
        if (self.current_tab == self.ids['buildPanel']):
            self.samplesDir = filenames[0]
            if (os.path.isdir(self.samplesDir)):
                self.parentDir =  str(Path(self.samplesDir).parent) + '/'
                print("self.parentDir = "+self.parentDir)
                self.ids['samplesDir'].text = self.samplesDir
                self.totalClips = count_files(self.samplesDir)
                text = "Contains "+str(self.totalClips)+" files"
                self.ids['buildStatus'].text = text
                self.ids['samplesButton'].state = "down"
        elif (self.current_tab == self.ids['trainPanel']):
            self.indexFile = filenames[0]
            if ('' == self.samplesDir):
                self.samplesDir =  os.path.dirname(self.indexFile)
                self.parentDir =  str(Path(self.samplesDir).parent) + '/'
            self.ids['indexFile'].text = self.indexFile
            self.ids['indexSelectButton'].state = "down"
            self.totalClips = file_len(self.indexFile)
            self.ids['trainStatus'].text = "References "+str(self.totalClips)+" files"
        elif ('sort_weights' == origin_button):
            self.sortWeightsFile = filenames[0]
            self.ids['sortWeightsLabel'].text = self.sortWeightsFile
            self.ids['sortWeightsButton'].state = 'down'
        elif ('sort_files' == origin_button):#(self.current_tab == self.ids['sortPanel']):
            '''  # removed sort files button because of ambiguous/confusing filebrowser behavior.
            # files to be sorted are now drag-only
            self.sortFileList = filenames
            self.ids['sortFilesDisplay'].text  = ''
            for name in filenames:
                if (not os.path.isdir(name)):         # single files
                    self.ids['sortFilesDisplay'].text += str(name) + '\n'
                else:       # TODO: what to do, decsend into the directory?  recursively?
                    pass
            '''
        return

    # this doesn't actually load file, it's just the result of the file selector gui
    def load(self, origin_button, path, filenames):
        self.dismiss_popup()
        if (filenames):
            self.got_filenames(filenames, origin_button )

    # sometimes you just want a widget id (name)
    def get_id(self,  instance):
        for id, widget in instance.parent.ids.items():
            if widget.__self__ == instance:
                return id


    # if you drag & drop multiple files, it treats them as separate events; but we want one list o files
    def consolidate_drops(self, file_path):
        now = time.time()
        tolerance = 1   # 1 second
        if (now - self.last_drop_time > tolerance):         # this is (the beginning of) a completely new drag-drop event
            name, extension = os.path.splitext(file_path)
            if ('.hdf5' == extension):
                self.sortWeightsFile = file_path
                self.ids['sortWeightsLabel'].text = self.sortWeightsFile
                self.ids['sortWeightsButton'].state = 'down'
            else:                                           # some other file (non-hdf5)
                self.sortFileList=[file_path]               #TODO: design descision: overwrite sortFileList (not append?)
                self.ids['sortFilesDisplay'].text = file_path
                self.ids['sortButton'].state = 'normal'         # reset the sort button state when files are dropped in

        else:                                               # this is yet another in a series of events triggered by dragging multiple files
            self.sortFileList.append(file_path)
            self.ids['sortFilesDisplay'].text += '\n'+file_path
            self.ids['sortStatus'].text = str(len(self.sortFileList))+" files selected"
        self.last_drop_time = now
        #print("Sort!  self.sortFileList = ",self.sortFileList)

    # this fires multiple times if multiple files are dropped
    def _on_file_drop(self, window, file_path):
        if (self.current_tab == self.ids['buildPanel']):
            self.got_filenames( [file_path.decode('UTF-8')], 'train_button' )
        elif (self.current_tab == self.ids['trainPanel']):
            self.got_filenames( [file_path.decode('UTF-8')], 'indexSelectButton' )
        elif (self.current_tab == self.ids['sortPanel']):
            self.consolidate_drops(file_path.decode('UTF-8'))

    #-------------- Indexing Samples folder -------------

    # for each file, in each directory in Samples/ list the classes (directories) it belongs to...
    def gen_index(self):
        self.ready_to_index = (self.samplesDir != '')
        if self.ready_to_index:
            self.ids['indexProgress'].value=10

            class_names = get_class_names(path=self.samplesDir) # list of subdirectories, which are the classes
            print("class_names = ",class_names)
            indexDict = dict()                                     # we'll use a dictionary to store names
            for classname in class_names:                          # go through each directory
                files_in_dir = listdir_nohidden(self.samplesDir+'/'+classname+'/')
                for filename in files_in_dir:
                    if filename in indexDict:
                        indexDict[filename].append(classname)
                    else:
                        indexDict[filename] = [classname]
            #print("indexDict = ",indexDict)
            self.ids['indexProgress'].value=50

            # ...and now somehow we save it as a CSV file...
            self.indexFile = self.samplesDir+'/'+'IndexFile.csv'
            with open(self.indexFile, 'w') as f:
                print('#file, tags', file=f)
                for key, tags in indexDict.items():
                    tagString = ",".join(tags)
                    print(key+","+tagString,file=f)
                f.close()
            self.ids['indexProgress'].value=100
            self.ids['buildStatus'].text='Dataset index generated: '+self.indexFile
            self.ids['indexFile'].text = self.indexFile
            self.ids['indexGenButton'].state = 'down'
            self.ids['indexSelectButton'].state = 'down'

    #-------------- Auto Mode --------------
    # calls all of the various steps in succession

    def auto(self):
        self.autoRunning = True
        Clock.schedule_once(lambda dt: partial(self.start_prog_anim, 'serverProgress')(), 0)
        self.preproc()
        #Clock.schedule_once(lambda dt: self.preproc(), 0.1)
        return

    #-------------- Preprocessing --------------

    def monitor_preproc(self, folder, thread, completion, dt):
        files_processed = count_files(folder)      # Folder should be Preproc
        percent = max(3, int(files_processed / self.totalClips * 100))
        self.ids['preprocProgress'].value = percent
        self.ids['trainStatus'].text = str(files_processed)+"/"+str(self.totalClips)+" files processed ("+str(percent)+"%)"
        if (self.ids['preprocProgress'].value >= 100):  # finished
            self.ids['preprocButton'].state = "down"
            if (self.autoRunning):
                #self.upload()
                Clock.schedule_once(lambda dt: self.upload(), 1)    # call the next stage
            return False              # this just cancels the clock schedule
        return

    def preproc(self):
        if ('' != self.samplesDir) and ('' != self.parentDir):
            self.ready_to_preproc = True
        if not self.ready_to_preproc:
            return
        #cmd = 'cd '+self.parentDir+'; rm -rf '+PREPROC_DIR
        #p = subprocess.call(cmd, shell=True)                       # blocking
        cmd = 'cd '+self.parentDir+'; rm -rf '+PREPROC_DIR+' '+ARCHIVE_NAME+'; '+PANOTTI_HOME+'/preprocess_data.py '
        if App.get_running_app().config.get('preproc', 'sequential'):   # ordering
            cmd += '-s '
        clean =  App.get_running_app().config.get('preproc', 'clean')
        if (1 == int(clean)):  # clean mode overrides some other settings
            cmd += '--clean '
        else:
            if (1== int(App.get_running_app().config.get('preproc', 'mono'))):
                cmd += '-m '
            cmd += '--dur='+App.get_running_app().config.get('preproc','duration')+' '
        cmd += '-r='+App.get_running_app().config.get('preproc','sampleRate')+' '
        cmd += '-i='+self.samplesDir+' '
        cmd += '--format='+App.get_running_app().config.get('preproc','specFileFormat')+' '
        #cmd += ' | tee log.txt '
        print('Preproc. Executing command: ',cmd)
        spawn(cmd, progress=partial(self.monitor_preproc,self.parentDir+PREPROC_DIR), interval=0.2, completion=None )
        return


    #-------------- Uploading Preproc --------------
    # for progress purposes, we'll split the percentage 40/60 between archive & upload

    # status messages , progress and such
    def my_upload_callback(self, filename, size, sent):
        percent = min( 25 + int( float(sent)/float(size)*75),  100)  # start at 25%, go up to 100
        prog_str = 'Uploading progress: '+str(percent)+' %'
        self.ids['trainStatus'].text = prog_str
        self.ids['uploadProgress'].value = percent
        if (percent >= 99):  # Finished
            self.ids['uploadButton'].state = "down"
            if (self.autoRunning):
                Clock.schedule_once(lambda dt: self.train(), 1)    # call the next stage
                self.autoRunning = False  # no more need after we call train
            return False    # kill the schedule


    # TODO: decide on API for file transfer. for now, we use scp
    def actual_upload(self, archive_path):
        self.server = whitelist_string( App.get_running_app().config.get('network', 'server') )
        self.username = whitelist_string( App.get_running_app().config.get('network', 'username') )
        scp_upload( src_blob=archive_path, options={'hostname': self.server, 'username': self.username}, progress=self.my_upload_callback )

    # Watches progress of packaging the Preproc/ directory.
    def monitor_archive(self, archive_file, orig_size, thread, completion, dt):
        #TODO: ok this is sloppy but estimating compression is 'hard'; we generally get around a factor of 10 in compression
        percent = 0
        if (os.path.isfile(archive_file) ):  # problem with this is, zip uses an alternate name until it's finished
            '''archive_size = os.path.getsize(archive_file)
            est_comp_ratio = 8
            est_size = orig_size/est_comp_ratio
            percent = min( int(archive_size / est_size * 50), 100 )'''
            cmd = "cd "+self.parentDir+"; tar -tvf "+ARCHIVE_NAME+" | wc | awk '{print $1}'"  # get the number of files in the archive
            tar_check = subprocess.check_output(cmd, shell=True)
            files_processed = int(tar_check)
            files_to_be_processed = count_files(self.parentDir+PREPROC_DIR)
            percent = int( files_processed / files_to_be_processed * 100)
            self.ids['uploadProgress'].value = int(percent * .25)  # make 30% of the upload bar be the archiving
            self.ids['trainStatus'].text = "Archiving... "+str(percent)+" %"

        if not thread.isAlive():           # archive process completed
            if (percent < 99):
                print("  Warning: archive finished with less than 100% complete")
            self.ids['trainStatus'].text = "Now Uploading..."
            Clock.schedule_once(lambda dt: completion(), 0.5)
            return False            # cancels scheduler
        return

    # this actually initiates "archiving" (zip/tar) first, and THEN uploads
    def upload(self):
        archive_path =  self.parentDir+PREPROC_DIR+'.tar.gz'
        self.ready_to_upload = (os.path.exists(self.parentDir+PREPROC_DIR) and (self.ids['serverProgress'].value >= 99) and
            ((self.ids['preprocProgress'].value >= 99) or (0 == self.ids['preprocProgress'].value)) )  # don't allow upload in the middle of processing
        if not self.ready_to_upload:
            return
        self.ids['trainStatus'].text = "Archiving "+PREPROC_DIR+"..."
        cmd = 'cd '+self.parentDir+'; rm -f '+archive_path+';  tar cfz '+archive_path+' '+PREPROC_DIR
        orig_size = folder_size(self.parentDir+PREPROC_DIR)
        spawn(cmd, progress=partial(self.monitor_archive,archive_path,orig_size), interval=0.5, completion=partial(self.actual_upload,archive_path) )

        return


    #-------------- Training --------------
    def train_is_complete(self,dst):
        self.ids['trainStatus'].text = '                   Training is complete!\nWeights file in '+dst
        self.ids['trainProgress'].value = 100
        self.ids['trainButton'].state = "down"
        self.sortWeightsFile = dst+'weights.hdf5'
        self.ids['sortWeightsLabel'].text = self.sortWeightsFile
        self.ids['sortWeightsButton'].state = 'down'
        self.autoRunning = False
        return False    # cancel clock schedule

    def download_weights(self):
        print("\n\nDownloading weights...")
        if ('' == self.parentDir):
            dst = "~/Downloads"
        else:
            dst = self.parentDir
        cmd = "scp "+self.username+'@'+self.server+':weights.hdf5 '+dst
        print("Executing command cmd = [",cmd,"]")
        spawn(cmd, progress=None, completion=partial(self.train_is_complete,dst))
        return

    def monitor_train(self, thread, completion, t):
        self.ids['trainProgress'].value = 10   # TODO: find a way to keep track of training. length of log file?
        # spawn a (blocking) process that checks the status of the training log file
        #cmd = "grep Epoch log.txt | grep / | tail -1 | awk '{print $2}'"
        #print("monitor_train: cmd = ",cmd)
        #p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        retval = 1#p.wait()
        if (0 == retval):
            epochstr1 = p.stdout.readlines()[0]
            epochstr2 = epochstr1.decode('UTF-8').replace('\n','')  # e.g. "12/20"
            print("epochstr2 = ",epochstr2)
            if (epochstr1 != '') and ('No such file' not in epochstr2):
                try:
                    epoch, maxepoch = map(int, epochstr2.split('/'))
                    percent = epoch / maxepoch * 100
                    self.ids['trainProgress'].value = percent
                except:
                    return
        if not thread.isAlive():
            self.ids['trainStatus'].text = "Train thread finished.\nDownloading weights..."
            Clock.schedule_once(lambda dt: completion(), 0.5)
            return False    # cancel schedule

    def change_train_status_msg(self,t):
        self.ids['trainStatus'].text = "Training, please wait..."

    def train(self, method='ssh'):
        self.ready_to_train = True#(self.ids['uploadProgress'].value >= 100)
        if not self.ready_to_train:
            return
        Clock.schedule_once(self.change_train_status_msg, 0)  # gotta delay by a bit to update the message

        self.server = whitelist_string( App.get_running_app().config.get('network', 'server') )
        self.username = whitelist_string( App.get_running_app().config.get('network', 'username') )

        if ('ssh' == method):
        # remote code execution via SSH server. could use sorting-hat HTTP server instead
            cmd = 'ssh -t '+self.username+'@'+self.server+' "rm -rf '+PREPROC_DIR+' log.txt; tar xfz '+ARCHIVE_NAME+';'
            if ('Random' == App.get_running_app().config.get('train', 'weightsOption')) and (self.ids['trainProgress'].value < 99):
                cmd+= ' rm -f weights.hdf5;'   # reset weights if first time pressing the train button, otherwise reuse what's on the server
            cmd += ' ~/panotti/train_network.py'
            cmd += ' --epochs='+App.get_running_app().config.get('train','epochs')
            cmd += ' --val='+App.get_running_app().config.get('train','val_split')
            cmd += ' | tee log.txt"'
            print("Executing command cmd = [",cmd,"]")
            spawn(cmd, progress=self.monitor_train, interval=1.5, completion=self.download_weights)
        elif ('http' == method):
            print("Yea, haven't done that yet")
        else:
            print("Error: Unrecognized API method '",method,"''")

    #-------------- Sort --------------
    def make_links(self, json_string, numIDs):
        import json
        parsed_json = json.loads(json_string)
        #print("parsed_json = ",parsed_json)
        self.sortedDir = self.parentDir+SORTED_DIR+'/'
        if not os.path.exists(self.sortedDir):
            os.mkdir(self.sortedDir)
        item_list = parsed_json["items"]
        for item in item_list:
            idnum = item["id"]   # idnum is the name b/c id is a builtin Python function
            src = item["name"]
            tags = item["tags"]
            #print("idnum = ",idnum,", src = ",src,", tags = ",tags)
            for tag in tags:
                dst = self.sortedDir+tag
                if not os.path.exists(dst):
                    os.mkdir(dst)
                dst += '/'+os.path.basename(src)  # need the actual filename for os.symlink (unlike "ln -s")
                if os.path.exists(dst):     # delete any previous link/file if it's there
                    if os.path.islink(dst):
                        os.unlink(dst)
                    else:
                        os.remove(dst)
                os.symlink(src,dst)
        self.ids['sortProgress'].value = 100
        self.ids['sortButton'].state = 'down'
        self.ids['sortStatus'].text = "Sorted!  Look in "+self.sortedDir

    def query_json_file(json_filename):
        return numIDs, json_string

    def monitor_sort(self, numfiles, thread, completion, t):
        # count the number of answers in the json file
        json_filename = self.parentDir+'data.json'
        print("monitor_sort: json_filename = ",json_filename)
        if not os.path.isfile(json_filename):
            print("monitor_sort: json file doesn't exist yet")
            return
        numIDs = 0
        json_string = ''
        with open(json_filename) as f:
            for i, line in enumerate(f):
                #print("i, line = ",i,line,end="")
                json_string += line
                if '\"id\"' in line:
                    numIDs += 1
                    #print(" -- got one!")
        finished_val = 90
        percent = int(numIDs / numfiles * finished_val)   # 90% is the ML sorting, last 10% is the linking (filesystem sorting)
        self.ids['sortStatus'].text = "Sorted "+str(numIDs)+'/'+str(numfiles)+' files'
        self.ids['sortProgress'].value = min( percent, finished_val)
        if (percent >= .99*finished_val):
            self.make_links(json_string, numIDs)
            return False

    def sort(self):
        self.ready_to_sort = os.path.isfile(self.sortWeightsFile) and (self.sortFileList is not None) and (self.sortFileList != [])
        if not self.ready_to_sort:
            return
        if not self.parentDir:
            self.parentDir = os.path.dirname(self.sortWeightsFile)+'/'
        cmd = "cd "+self.parentDir+"; rm data.json; "+PANOTTI_HOME+'predict_class.py '
        numfiles = len(self.sortFileList)
        for name in self.sortFileList:   # escape spaces
            cmd += name.replace(" ","\\ ")+" "
        #cmd += " | tee log.txt"
        self.ids['sortStatus'].text = "Sorting "+str(numfiles)+" files..."

        print("Sorting",numfiles,"files.   cmd = ",cmd)
        self.ids['sortProgress'].value = 10
        spawn(cmd, progress=partial(self.monitor_sort, numfiles), interval=1, completion=self.make_links)


    #-------------- Settings --------------

    # programaticaly change to tab_state (i.e. tab instance)
    def change_to_tab(self, tab_state, t):
        self.switch_to(tab_state)

    # opens settings view, returns to tab you were on before
    def my_handle_settings(self):
        tab_state = self.current_tab            # remember what tab we're on
        App.get_running_app().open_settings()
        Clock.schedule_once(partial(self.change_to_tab, tab_state), 0.1)  # switch back to orig tab after a slight delay

#============== End of main widget ==============

class SortingHATApp(App):
    def build(self):
        Window.size = (700, 400)
        self.icon = 'static/sorting-hat-logo.png'
        self.use_kivy_settings = False
        return SHPanels()

    def build_config(self, config):
        config.setdefaults('network', {
            'server': 'lecun',
            'username': os.getlogin(),      # default is that they have the same username on both local & server
            'sshKeyPath': '~/.ssh/id_rsa.pub'})
        config.setdefaults('preproc', {
            'mono': True,
            'clean': False,
            'sequential': False,   # for test/train split. True: preserve order and send last files (per class) to Test.  False=shuffle first
            'duration': 3,
            'sampleRate': 44100,
            'specFileFormat': 'npz', # note, color png supports only up to 4 channels of audio, npz is arbitrarily many, jpeg is lossy
            'split_audio': False,  # truncate files that are too long, rather than send multiple chopped-up parts through
                                   # Note: if split_audio=True and sequential=False, you'll likely end up corrupting the Testing set
            })
        config.setdefaults('train', {
            'weightsOption': 'Default',
            'server': 'lecun.belmont.edu',
            'sshKeyPath': '~/.ssh/id_rsa.pub',
            'epochs': 20,
            'val_split': 0.1,
            'aug_fac': 1.0,
            })

    def build_settings(self, settings):
        settings.add_json_panel('Settings',
                                self.config,
                                data=settings_json)

    def on_config_change(self, config, section,
                         key, value):
        print(config, section, key, value)



if __name__ == '__main__':
    SortingHATApp().run()
