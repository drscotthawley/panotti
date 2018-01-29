# Author: Braiden King
# Team 15
# HackMT 2018

import os
import subprocess
from flask import Flask, request, render_template, send_file 
import shutil
import time
app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template('index.html')

#execution of the train submition
@app.route("/upload_train", methods=["POST"])
def upload_train():

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

    cmd = 'unzip '+destination+' | tee log.txt '
    print('cmd = [',cmd,']')
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    out,err = p.communicate()       # all the output and error will get sent to the browser
    print("out = ",str(out))
    print("err = ",str(err))
    return render_template('preproc.html')


@app.route("/upload_preproc",methods=['GET','POST'])
def upload_progress():
    cmd = '~/panotti/preprocess_data.py -s | tee -a log.txt'
    print('cmd = [',cmd,']')
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    out,err = p.communicate()       # all the output and error will get sent to the browser
    print("out = ",str(out))
    print("err = ",str(err))

    cmd = '~/panotti/train_network.py | tee -a log.txt'# ; ~/panotti/train_network.py | tee log.txt'
    print('cmd = [',cmd,']')
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    out,err = p.communicate()       # all the output and error will get sent to the browser
    print("out = ",str(out))
    print("err = ",str(err))
    return render_template('done_train.html')


@app.route("/download_train", methods=["POST"])
def download_train():
      try:
          return send_file("weights.hdf5", attacthment_filename= "weights.hdf5")
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

    cmd = 'cd Samples_sort; ~/panotti/predict_class.py -w ../weights.hdf5 -c ../Samples * | tee -a log.txt'
    print('cmd = [',cmd,']')
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    out,err = p.communicate()       # all the output and error will get sent to the browser
    print("out = ",str(out))
    print("err = ",str(err))

    return render_template('done_sort.html')


@app.route("/download_sort", methods=["GET","PUT"])
def download_sort():
   cmd= "echo hello;  pwd;  zip -r downloads.zip data.json"# file.exe"
   print('cmd = [',cmd,']')
   p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
   out,err = p.communicate()       # all the output and error will get sent to the browser
   print("out = ",str(out))
   print("err = ",str(err))
   try:
       return send_file("Samples_sort/data.json")#, attacthment_filename= "downloads.zip")
   except Exception as e:
       return str(e)


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=8000, host='0.0.0.0')
