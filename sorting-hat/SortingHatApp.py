#!/usr/bin/env python3
'''
SortingHatApp  - Desktop version of Sorting H.A.T. program.

Author: Scott H. Hawley @drscotthawley

TODO:
    - Everything.  Still just a facade, doesn't actually work at all.
    - Just learning Kivy as I write this. Still quite confused.
    - Should create a "ButtonBarAndStatus" class that can be reused multiple times

Requirements:
    $ pip install kivy kivy-garden scandir functools
    $ garden install filebrowser
============
'''

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.progressbar import ProgressBar
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.garden.filebrowser import FileBrowser
import subprocess
#import scandir

from functools import partial
import os

PANOTTI_HOME = os.path.expanduser("~")+"/github/panotti"

def count_files(folder):
    total = 0
    for root, dirs, files in os.walk(folder):
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

Builder.load_string("""
#:import expanduser os.path.expanduser

<SHPanels>:
    id: SH_widget
    #size_hint: .5, .5
    pos_hint: {'center_x': .5, 'center_y': .5}
    do_default_tab: False
    tab_width: self.width/4
    TabbedPanelItem:
        text: 'Train the Neural Net'
        BoxLayout:
            orientation: 'vertical'
            BoxLayout:
                orientation: 'horizontal'
                Button:
                    text: 'Select Samples Folder'
                    on_release: SH_widget.show_load()
                Label:
                    id: samplesDir
                    text: 'No folder selected'
                    size: self.texture_size
            BoxLayout:
                orientation: 'horizontal'
                Button:
                    text: "Start Server"
                    id: serverButton
                    on_release: root.start_prog_anim('serverProgress')
                ProgressBar:
                    id: serverProgress
                    value: 0
            BoxLayout:
                orientation: 'horizontal'
                Button:
                    text: "Create Spectrograms"
                    on_release: root.preproc('preprocProgress')
                ProgressBar:
                    id: preprocProgress
                    value: 0
            BoxLayout:
                orientation: 'horizontal'
                Button:
                    text: "Upload"
                    id: uploadButton
                    on_release: root.upload('uploadProgress')
                ProgressBar:
                    id: uploadProgress
                    value: 0
            BoxLayout:
                orientation: 'horizontal'
                Button:
                    text: "Train"
                    id: trainButton
                    on_release: root.start_prog_anim('trainProgress')
                ProgressBar:
                    id: trainProgress
                    value: 0
            Label:
                id: statusMsg
                text: "Status: Amazing statuses coming your way..."
                center: self.parent.center

    TabbedPanelItem:
        text: 'Sort Your Library'
        BoxLayout:
            Label:
                text: 'Second tab content area'
            Button:
                text: 'Button that does nothing'
    TabbedPanelItem:
        text: 'About'
        RstDocument:
            text:
                '\\n'.join(("About", "-----------",
                "Sorting H.A.T.* - Organize your audio library with the help of neural nets."))
    TabbedPanelItem:
        text: 'Settings'
        Label:
            text: 'There is a way to have Kivy do settings automatically...'


<LoadDialog>:
    BoxLayout:
        size: root.size
        pos: root.pos
        orientation: "vertical"
        FileBrowser:
            id: filechooser
            dirselect: True
            path: expanduser("~")
        BoxLayout:
            size_hint_y: None
            height: 30
            Button:
                text: "Cancel"
                on_release: root.cancel()

            Button:
                text: "Load"
                on_release: root.load(filechooser.path, filechooser.selection)

""")

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SHPanels(TabbedPanel):
    def __init__(self, **kwargs):
        super(SHPanels, self).__init__(**kwargs)
        Window.bind(on_dropfile=self._on_file_drop)
        self.progress_events = {}
        self.ready_to_preproc = False
        self.ready_to_upload = False
        self.ready_to_train = False

    server_progress = ObjectProperty()
    def next(self, barname, dt):         # little updater for 'fake' progress bar updates
        self.ids[barname].value += 1
        if (self.ids[barname].value >= 100):
            self.progress_events[barname].cancel()
            if ('trainProgress' == barname):
                self.ids['statusMsg'].text = "Downloading weights..."

    def start_prog_anim(self,barname):
        self.ready_to_preproc = ('No folder selected' != self.ids['samplesDir'].text)
        self.ready_to_upload = (self.ids['preprocProgress'].value >= 100) and (self.ids['serverProgress'].value >= 100)
        self.ready_to_train = (self.ids['uploadProgress'].value >= 100) and (self.ready_to_upload)
        if (('serverProgress' == barname) or
            (('preprocProgress' == barname) and self.ready_to_preproc) or
            (('uploadProgress' == barname) and self.ready_to_upload) or
            (('trainProgress' == barname) and self.ready_to_train) ):
            self.ids[barname].value = 0
            self.progress_events[barname] = Clock.schedule_interval(partial(self.next,barname), 1/50)


    def monitor_preproc_progress(self, folder, p, dt):
        files_processed = count_files(folder)      # Folder should be Preproc
        self.ids['preprocProgress'].value = max(5, int(files_processed / self.totalClips * 100))
        self.ids['statusMsg'].text = str(files_processed)+"/"+str(self.totalClips)+" files processed"
        if (self.ids['preprocProgress'].value >= 99.4) or (p.poll() is not None):
            self.preproc_sched.cancel()

    def preproc(self,barname):
        cmd = 'cd '+self.samplesDir+'/..; rm -rf Preproc'
        p = subprocess.call(cmd, shell=True)                       # blocking
        cmd = 'cd '+self.samplesDir+'/..; '+PANOTTI_HOME+'/preprocess_data.py -s -m --dur=4.0 -r=44100 | tee log.txt '
        p = subprocess.Popen(cmd, shell=True)                      # non-blocking
        self.preproc_sched = Clock.schedule_interval(partial(self.monitor_preproc_progress,self.samplesDir+'/../Preproc', p), 0.2)

        return

    def actual_upload(self, archive_file):   #
        return

    def monitor_archive_progress(self, archive_file, orig_size, p, dt):   # archive as in zip or tar
        #TODO: ok this is sloppy but estimating compression is 'hard'; we generally get around a factor of 10 in compression
        if (os.path.isfile(archive_file) ):  # problem with this is, zip uses an alternate name until it's finished
            archive_size = os.path.getsize(archive_file)
            est_comp_ratio = 10
            est_size = orig_size/est_comp_ratio
            percent = int(archive_size / est_size * 100)
            self.ids['statusMsg'].text = "Archiving... "+str(percent)+" %"

        if (p.poll() is not None):           # archive process completed
            self.archive_sched.cancel()
            self.ids['statusMsg'].text = "Now Uploading..."
            barname = 'uploadProgress'
            self.progress_events[barname] = Clock.schedule_interval(partial(self.next,barname), 1/50)
        return

    def upload(self,barname):           # this actually initiates "archiving" (zip/tar), and THEN uploads
        self.ready_to_upload = (self.ids['preprocProgress'].value >= 100) and (self.ids['serverProgress'].value >= 100)
        if (self.ready_to_upload):
            archive_file = self.samplesDir+'/../Preproc.tar.gz'
            self.ids['statusMsg'].text = "Archiving Preproc...)"
            cmd = 'cd '+self.samplesDir+'/..; rm -f '+archive_file+';  tar cfz '+archive_file+' Preproc/'
            p = subprocess.Popen(cmd, shell=True)
            orig_size = folder_size(self.samplesDir+'/../Preproc')
            self.archive_sched = Clock.schedule_interval(partial(self.monitor_archive_progress, archive_file, orig_size, p), 0.2)
        return # self.start_prog_anim(barname)

    def open(self, path, filename):
        with open(os.path.join(path, filename[0])) as f:
            print(f.read())

    def selected(self, filename):
        print("selected: %s" % filename[0])


    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def got_samplesDir(self):
        self.ids['samplesDir'].text = self.samplesDir
        self.totalClips = count_files(self.samplesDir)
        text = "Contains "+str(self.totalClips)+" files"
        self.ids['statusMsg'].text = text
        return

    def load(self, path, filename):
        print("filename = ",filename)
        self.samplesDir = str(filename[0])
        self.dismiss_popup()
        self.got_samplesDir()
        return

    def _on_file_drop(self, window, file_path):
        self.samplesDir = file_path.decode('UTF-8')
        self.got_samplesDir()
        return


class SortingHatApp(App):
    def build(self):
        return SHPanels()


if __name__ == '__main__':
    SortingHatApp().run()
