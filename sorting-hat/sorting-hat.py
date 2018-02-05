#! /usr/bin/env python3

# Author: Braiden King
# Team 15
# HackMT 2018

import os
import subprocess
from flask import Flask, request, render_template, send_file, send_from_directory
import shutil
import time
from jinja2 import Markup
import os

app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


app.jinja_env.globals['include_file'] = lambda filename : Markup(app.jinja_loader.get_source(app.jinja_env, filename)[0])



@app.route("/")
def index():
    if os.path.exists('.lock'):
        return render_template('busy.html')   # tell user to come back if it's in use
    return render_template('index.html')


@app.route("/busy")
def busy():
    return render_template('busy.html')


# User uploads data for training
@app.route("/upload_train", methods=["POST"])
def upload_train():
    if os.path.exists('.lock'):
        return render_template('busy.html')   # tell user to come back if it's in use

    target = os.path.join(APP_ROOT, './')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):

        filename = str(file.filename)
        filename.replace(' ','\\ ')
        destination = "/".join([target, filename])
        print("filename = ",filename,", destination = ",destination)
        file.save(destination)

    cmd = 'touch .lock; rm -rf Preproc Samples; unzip '+destination+' | tee log.txt '
    print('cmd = [',cmd,']')
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    out,err = p.communicate()       # all the output and error will get sent to the browser
    print("out = ",str(out))
    print("err = ",str(err))
    return render_template('preproc.html')


@app.route("/upload_preproc",methods=['GET','POST'])
def upload_preproc():
    cmd = 'touch .lock; rm -f Samples.zip;  ../preprocess_data.py -s -m --dur=4.0 -r=44100 | tee -a log.txt'
    print('cmd = [',cmd,']')
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    out,err = p.communicate()       # all the output and error will get sent to the browser
    print("out = ",str(out))
    print("err = ",str(err))
    return render_template('train.html')

@app.route("/train",methods=['GET','POST'])
def train():
    cmd = 'touch .lock; ../train_network.py --epochs=10 --val=0 | tee -a log.txt;  rm -f .lock'
    print('cmd = [',cmd,']')
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    out,err = p.communicate()       # all the output and error will get sent to the browser
    print("out = ",str(out))
    print("err = ",str(err))
    return render_template('done_train.html')


@app.route("/download_train", methods=["GET", "POST"])
def download_train():
      try:
          #return send_file("weights.hdf5", attachment_filename= "weights.hdf5")
          return send_from_directory(".", "weights.hdf5", as_attachment=True)
      except Exception as e:
          return str(e)


@app.route("/upload_sort", methods=["POST"])
def upload_sort():

    target = os.path.join(APP_ROOT, 'Samples_sort/')

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):

        filename = file.filename
        filename = str(file.filename)
        if ('' != filename):
            filename.replace(' ','\\ ')
            destination = "/".join([target, filename])
            print(" in sort: filename = ",filename,", destination = ",destination)
            file.save(destination)

    cmd = 'touch .lock; rm -f data.json; cd Samples_sort; ../../predict_class.py -d=4.0 -r=44100 -m -w ../weights.hdf5 -c ../Samples * | tee -a ../log.txt; cp data.json ..; cd ..; rm -rf .lock Samples_sort'
    print('cmd = [',cmd,']')
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    out,err = p.communicate()       # all the output and error will get sent to the browser
    print("out = ",str(out))
    print("err = ",str(err))

    return render_template('done_sort.html')


@app.route("/download_sort", methods=["GET","PUT"])
def download_sort():
   #cmd= "echo hello;  pwd;  zip -r downloads.zip data.json"# file.exe"
   #print('cmd = [',cmd,']')
   #p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
   #out,err = p.communicate()       # all the output and error will get sent to the browser
   #print("out = ",str(out))
   #print("err = ",str(err))
   try:
       return send_from_directory(".", "data.json", as_attachment=True)
   except Exception as e:
       return str(e)




if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=8000, host='0.0.0.0')
