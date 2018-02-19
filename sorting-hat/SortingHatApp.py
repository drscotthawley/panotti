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
    $ garden install filebrowser

============
'''

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
import os
import re
from pathlib import Path

PANOTTI_HOME = os.path.expanduser("~")+"/panotti/"
PREPROC_DIR = "Preproc"
ARCHIVE_NAME = PREPROC_DIR+".tar.gz"
REMOTE_RUN_DIR = "~/SortingHatRuns"
SORTED_DIR = "Sorted"

def count_files(folder):
    total = 0
    for root, dirs, files in os.walk(folder):
        files = [f for f in files if not f[0] == '.']     # ignore hidden files
        dirs[:] = [d for d in dirs if not d[0] == '.']    # ignore hidden directories
        total += len(files)
    return total


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

Builder.load_string("""
#:import expanduser os.path.expanduser

<ScrollableLabel>:  # https://github.com/kivy/kivy/wiki/Scrollable-Label
    Label:
        size_hint_y: None
        height: self.texture_size[1]
        text_size: self.width, None
        text: root.text
        halign: 'center'
        valign: 'center'


<CustomWidthTabb@TabbedPanelItem>
    width: self.texture_size[0]
    padding: 20, 0
    size_hint_x: None

<ButtonWithState@Button>
    background_color: 0,0,0,0  # the last zero is the critical on, make invisible
    canvas.before:
        Color:
            rgba: (.35,.35,.35,1) if self.state=='normal' else (0,.5,.0,1)  # visual feedback of press
        RoundedRectangle:
            size: (self.width -2.0, self.height - 2.0)
            pos: ((self.right - self.width + 2.0),(self.top - self.height + 2.0))
            radius: [3,]

<TextInputLikeLabel@TextInput>
    # Really I just want a Label that supports text highlighting.  But kivy doesn't have that, so...
    background_color: (.35,.35,.35,1)
    foreground_color: (1, 1, 1, 1)
    readonly: True
    ## kivy doesn't support centering text in a TextInput, only in a Label
    # left, right   https://stackoverflow.com/questions/40477956/kivy-textinput-horizontal-and-vertical-align-centering-text
    padding_x: [self.center[0] - self._get_text_width(max(self._lines, key=len), self.tab_width, self._label_cached) / 2.0,0] if self.text else [self.center[0], 0]
    # top, bottom
    padding_y: [self.height / 2.0 - (self.line_height / 2.0) * len(self._lines), 0]

<SHPanels>:
    id: SH_widget
    size_hint: 1,1
    do_default_tab: False
    tab_width: None
    CustomWidthTabb:
        id: trainPanel
        text: 'Train the Neural Net'
        BoxLayout:
            orientation: 'vertical'
            BoxLayout:
                orientation: 'horizontal'
                ButtonWithState:
                    id: samplesButton
                    text: 'Select Samples Folder'
                    on_release: SH_widget.show_load('train')
                Label:
                    id: samplesDir
                    text: 'No folder selected'
                    size: self.texture_size
            BoxLayout:
                orientation: 'horizontal'
                ButtonWithState:
                    text: "Start Server"
                    id: serverButton
                    on_release: root.start_prog_anim('serverProgress')
                ProgressBar:
                    id: serverProgress
                    value: 0
            BoxLayout:
                orientation: 'horizontal'
                ButtonWithState:
                    id: preprocButton
                    text: "Create Spectrograms"
                    on_release: root.preproc('preprocProgress')
                ProgressBar:
                    id: preprocProgress
                    value: 0
            BoxLayout:
                orientation: 'horizontal'
                ButtonWithState:
                    text: "Upload"
                    id: uploadButton
                    on_release: root.upload('uploadProgress')
                ProgressBar:
                    id: uploadProgress
                    value: 0
            BoxLayout:
                orientation: 'horizontal'
                ButtonWithState:
                    text: "Train"
                    id: trainButton
                    on_release: root.train('trainProgress')
                ProgressBar:
                    id: trainProgress
                    value: 0
            TextInputLikeLabel:
                id: statusMsg
                text: "Status: Initial State"
                center: self.parent.center

    CustomWidthTabb:
        text: 'Sort Your Library'
        id: sortPanel
        BoxLayout:
            orientation: 'vertical'
            BoxLayout:
                size_hint_y: 0.3
                ButtonWithState:
                    id: sortWeightsButton
                    text: 'Select Weights File'
                    on_release: SH_widget.show_load('sort_weights')
                Label:
                    id: sortWeightsLabel
                    text: 'No weights file'
            ScrollableLabel:
                size_hint_y: 0.9
                id: sortFilesDisplay
                text: '\\n\\n\\n\\n[Drag in files to be sorted]'
            BoxLayout:
                size_hint_y: 0.3
                ButtonWithState:
                    text: 'Sort!'
                    id: sortButton
                    on_release: root.sort()
                ProgressBar:
                    id: sortProgress
                    value: 0
            TextInputLikeLabel:
                id: sortStatus
                size_hint_y: 0.3
                text: "Status: Initial State"
                center: self.parent.center
    CustomWidthTabb:
        text: 'About'
        BoxLayout:
            RstDocument:
                text:
                    '\\n'.join(("About", "-----------",
                    "Sorting H.A.T.* - Organize your audio library with the help of neural nets.\\n",
                    "Built on `Panotti <http://github.com/drscotthawley/panotti>`_ by @drscotthawley"))

            Image:
                source: 'static/sorting-hat-logo.png'
                canvas.before:
                    Color:
                        rgba: .9, .9, .9, 1
                    Rectangle:
                        pos: self.pos
                        size: self.size

    CustomWidthTabb:
        id: settingsPanel
        text: 'Settings'
        on_press: SH_widget.my_handle_settings()
        Button:
            text: 'Press to go to Train'
            on_release: root.switch_to(trainPanel)



<LoadDialog>:
    BoxLayout:
        size: root.size
        pos: root.pos
        orientation: "vertical"
        FileBrowser:
            id: filechooser
            #multiselect: True.  # Nope, bad/ambigious. Kivy filebrowser needs to be rewritten.
            dirselect: True
            path: expanduser("~")
            on_canceled: root.cancel()
            on_success: root.load(filechooser.path, filechooser.selection)
""")

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
        self.last_drop_time = time.time()-10000
        self.progress_events = {}
        self.samplesDir = ''
        self.parentDir = ''
        self.ready_to_preproc = False
        self.ready_to_upload = False
        self.ready_to_train = False
        self.sortFileList = []
        self.sortWeightsFile = ''

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
        self.ids['statusMsg'].text = "Server is up and running."
        self.ids['serverButton'].state = "down"

    def progress_display(self, barname, thread, completion, t):
        percent = int((self.count+1) / self.maxval * 100)
        self.ids[barname].value = percent
        # Test for completion:
        if not thread.isAlive():         # if the thread has completed
            if (percent >=100):          # Yay
                completion()    # go on to the final state
            else:
                print("\nError: Process died but progress < 100% ")
            return False                 # Either way, cancel the Kivy clock schedule
        return True                      # keep the progress-checker rolling

    def count_up(self,maxval=100):     # some 'fake' task for testing purposes
        self.maxval = maxval
        for self.count in range(maxval):
            time.sleep(0.05)


    def start_prog_anim(self,barname):
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
        if (self.current_tab == self.ids['trainPanel']):
            self.samplesDir = filenames[0]
            if (os.path.isdir(self.samplesDir)):
                self.parentDir =  str(Path(self.samplesDir).parent) + '/'
                print("self.parentDir = "+self.parentDir)
                self.ids['samplesDir'].text = self.samplesDir
                self.totalClips = count_files(self.samplesDir)
                text = "Contains "+str(self.totalClips)+" files"
                self.ids['statusMsg'].text = text
                self.ids['samplesButton'].state = "down"
        elif ('sort_weights' == origin_button):
            self.sortWeightsFile = filenames[0]
            self.ids['sortWeightsLabel'].text = self.sortWeightsFile
            self.ids['sortWeightsButton'].state = 'down'
        elif ('sort_files' == origin_button):#(self.current_tab == self.ids['sortPanel']):
            '''  # removed sort files button because it was ambiguous
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
        else:                                               # this is yet another in a series of events triggered by dragging multiple files
            self.sortFileList.append(file_path)
            self.ids['sortFilesDisplay'].text += '\n'+file_path
            self.ids['sortStatus'].text = str(len(self.sortFileList))+" files selected"
        self.last_drop_time = now
        #print("Sort!  self.sortFileList = ",self.sortFileList)

    # this fires multiple times if multiple files are dropped
    def _on_file_drop(self, window, file_path):
        if (self.current_tab == self.ids['trainPanel']):
            self.got_filenames( [file_path.decode('UTF-8')], 'train_button' )
        elif (self.current_tab == self.ids['sortPanel']):
            self.consolidate_drops(file_path.decode('UTF-8'))


    #-------------- Preprocessing --------------

    def monitor_preproc(self, folder, thread, completion, dt):
        files_processed = count_files(folder)      # Folder should be Preproc
        self.ids['preprocProgress'].value = max(3, int(files_processed / self.totalClips * 100))
        self.ids['statusMsg'].text = str(files_processed)+"/"+str(self.totalClips)+" files processed"
        if (self.ids['preprocProgress'].value >= 99.4):  # finished
            self.ids['preprocButton'].state = "down"
            return False              # this just cancels the clock schedule
        return

    def preproc(self,barname):
        if ('' != self.samplesDir) and ('' != self.parentDir):
            self.ready_to_preproc = True
        if not self.ready_to_preproc:
            return
        #cmd = 'cd '+self.parentDir+'; rm -rf '+PREPROC_DIR
        #p = subprocess.call(cmd, shell=True)                       # blocking
        cmd = 'cd '+self.parentDir+'; rm -rf '+PREPROC_DIR+' '+ARCHIVE_NAME+'; '+PANOTTI_HOME+'/preprocess_data.py '
        if App.get_running_app().config.get('example', 'sequential'):
            cmd += '-s '
        if App.get_running_app().config.get('example', 'mono'):
            cmd += '-m '
        cmd += '--dur='+App.get_running_app().config.get('example','duration')+' '
        cmd += '-r='+App.get_running_app().config.get('example','sampleRate')+' '
        cmd += '--format='+App.get_running_app().config.get('example','specFileFormat')+' '
        #cmd += ' | tee log.txt '
        print('Executing command: ',cmd)
        spawn(cmd, progress=partial(self.monitor_preproc,self.parentDir+PREPROC_DIR), interval=0.2, completion=None )
        return


    #-------------- Uploading Preproc --------------
    # for progress purposes, we'll split the percentage 40/60 between archive & upload

    # status messages , progress and such
    def my_upload_callback(self, filename, size, sent):
        percent = min( 60 + int( float(sent)/float(size)*60),  100)
        prog_str = 'Uploading progress: '+str(percent)+' %'
        self.ids['statusMsg'].text = prog_str
        barname = 'uploadProgress'
        self.ids[barname].value = percent
        if (percent >= 99):  # Finished
            self.ids['uploadButton'].state = "down"
            return False    # kill the schedule


    # TODO: decide on API for file transfer. for now, we use scp
    def actual_upload(self, archive_path):
        self.server = whitelist_string( App.get_running_app().config.get('example', 'server') )
        self.username = whitelist_string( App.get_running_app().config.get('example', 'username') )
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
            self.ids['statusMsg'].text = "Archiving... "+str(percent)+" %"

        if not thread.isAlive():           # archive process completed
            if (percent < 99):
                print("  Warning: archive finished with less than 100% complete")
            self.ids['statusMsg'].text = "Now Uploading..."
            completion()
            return False            # cancels scheduler
        return

    # this actually initiates "archiving" (zip/tar) first, and THEN uploads
    def upload(self,barname):
        archive_path =  self.parentDir+PREPROC_DIR+'.tar.gz'
        self.ready_to_upload = os.path.exists(self.parentDir+PREPROC_DIR) and (self.ids['serverProgress'].value >= 99) and (self.ids['preprocProgress'].value >= 99)
        if not self.ready_to_upload:
            return
        self.ids['statusMsg'].text = "Archiving "+PREPROC_DIR+"..."
        cmd = 'cd '+self.parentDir+'; rm -f '+archive_path+';  tar cfz '+archive_path+' '+PREPROC_DIR
        orig_size = folder_size(self.parentDir+PREPROC_DIR)
        spawn(cmd, progress=partial(self.monitor_archive,archive_path,orig_size), interval=0.2, completion=partial(self.actual_upload,archive_path) )

        return


    #-------------- Training --------------
    def train_is_complete(self,dst):
        self.ids['statusMsg'].text = 'Training is complete!\nWeights file in '+dst
        self.ids['trainProgress'].value = 100
        self.ids['trainButton'].state = "down"
        self.sortWeightsFile = dst+'weights.hdf5'
        self.ids['sortWeightsLabel'].text = self.sortWeightsFile
        self.ids['sortWeightsButton'].state = 'down'
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
        if not thread.isAlive():
            self.ids['statusMsg'].text = "Train thread finished.\nDownloading weights..."
            completion()
            return False    # cancel schedule

    def change_train_status_msg(self,t):
        self.ids['statusMsg'].text = "Training, please wait..."

    def train(self, barname, method='ssh'):
        self.ready_to_train = (self.ids['uploadProgress'].value >= 100)
        if not self.ready_to_train:
            return
        Clock.schedule_once(self.change_train_status_msg, 0.5)  # gotta delay by a bit to update the message

        self.server = whitelist_string( App.get_running_app().config.get('example', 'server') )
        self.username = whitelist_string( App.get_running_app().config.get('example', 'username') )

        if ('ssh' == method):
        # remote code execution via SSH server. could use sorting-hat HTTP server instead
            cmd = 'ssh -t '+self.username+'@'+self.server+' "tar xvfz Preproc.tar.gz;'
            if ('Random' == App.get_running_app().config.get('example', 'weightsOption')):
                cmd+= ' rm -f weights.hdf5;'
            cmd += ' ~/panotti/train_network.py'
            cmd += ' --epochs='+App.get_running_app().config.get('example','epochs')
            cmd += ' --val='+App.get_running_app().config.get('example','val_split')
            cmd += ' | tee log.txt"'
            print("Executing command cmd = [",cmd,"]")

            spawn(cmd, progress=self.monitor_train, interval=1, completion=self.download_weights)
            #p = subprocess.call(cmd, shell=True)   # blocking  TODO: make non-blocking
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

class SortingHatApp(App):
    def build(self):
        Window.size = (600, 400)
        self.icon = 'static/sorting-hat-logo.png'
        self.use_kivy_settings = False

        return SHPanels()

    def build_config(self, config):
        config.setdefaults('example', {
            'server': 'lecun',
            'username': os.getlogin(),      # default is that they have the same username on both local & server
            'sshKeyPath': '~/.ssh/id_rsa.pub',
            'mono': True,
            'sequential': True,
            'duration': 3,
            'sampleRate': 44100,
            'specFileFormat': 'npz',   # note, color png supports only up to 4 channels of audio, npz is arbitrarily many, jpeg is lossy
            'weightsOption': 'Default',
            'server': 'lecun.belmont.edu',
            'sshKeyPath': '~/.ssh/id_rsa.pub',
            'epochs': 20,
            'val_split': 0.1,
            })

    def build_settings(self, settings):
        settings.add_json_panel('Settings',
                                self.config,
                                data=settings_json)

    def on_config_change(self, config, section,
                         key, value):
        print(config, section, key, value)


if __name__ == '__main__':
    SortingHatApp().run()
