#!/usr/bin/env python3
'''
SortingHatApp  - Desktop version of Sorting H.A.T. program.

Author: Scott H. Hawley @drscotthawley

TODO: - Everything.  Still just a facade, doesn't actually work at all.
      - Just learning Kivy as I write this. Still quite confused.
      - Should create a "ButtonBarAndStatus" class that can be reused multiple times

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

from functools import partial
import os

PANOTTI_HOME = os.path.expanduser("~")+"/github/panotti"

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
                    on_release: root.start_prog_anim('uploadProgress')
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
    def next(self, barname, dt):
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

    def preproc(self,barname):
        self.start_prog_anim(barname)

    def open(self, path, filename):
        with open(os.path.join(path, filename[0])) as f:
            print(f.read())

    def selected(self, filename):
        print("selected: %s" % filename[0])

    def _on_file_drop(self, window, file_path):
        self.samplesDir = file_path.decode('UTF-8')
        self.ids['samplesDir'].text = self.samplesDir
        return

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        print("filename = ",filename)
        self.samplesDir = str(filename[0])
        self.ids['samplesDir'].text = self.samplesDir

        self.totalClips = 0
        for root, dirs, files in os.walk(self.samplesDir):
            self.totalClips += len(files)
        self.ids['statusMsg'].text = "Number of clips = "+str(self.totalClips)

        self.dismiss_popup()


class SortingHatApp(App):
    def build(self):
        return SHPanels()


if __name__ == '__main__':
    SortingHatApp().run()
