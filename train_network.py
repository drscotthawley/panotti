#! /usr/bin/env python3
'''
Classify sounds using database
Author: Scott H. Hawley

This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
   and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

Trained using Fraunhofer IDMT's database of monophonic guitar effects,
   clips were 2 seconds long, sampled at 44100 Hz
'''
from __future__ import print_function
import sys
print(sys.path)
print(sys.version)
import numpy as np
from panotti.models import *
from panotti.datautils import *
#from keras.callbacks import ModelCheckpoint #,EarlyStopping
import os
from os.path import isfile
from timeit import default_timer as timer
from panotti.multi_gpu import MultiGPUModelCheckpoint
from panotti.mixup_generator import MixupGenerator
import math



def lr_sched(epoch):
    initial_rate = 0.5
    max_rate = 2.5
    epochs = 200
    b = epochs/3.0
    if epoch < b:
      lrate = initial_rate + (max_rate - initial_rate)/b*epoch
    else:
      lrate = max_rate*(1+np.cos((epoch-b)/(epochs-b)*np.pi))/2+1e-5
    return lrate


def get_1cycle_schedule(lr_max=1e-3, n_data_points=8000, epochs=200, batch_size=40, verbose=0):
  """
  Creates a look-up table of learning rates for 1cycle schedule with cosine annealing
  See @sgugger's & @jeremyhoward's code in fastai library: https://github.com/fastai/fastai/blob/master/fastai/train.py
  Wrote this to use with my Keras and (non-fastai-)PyTorch codes.
  Note that in Keras, the LearningRateScheduler callback (https://keras.io/callbacks/#learningratescheduler) only operates once per epoch, not per batch
      So see below for Keras callback

  Keyword arguments:
    lr_max            chosen by user after lr_finder
    n_data_points     data points per epoch (e.g. size of training set)
    epochs            number of epochs
    batch_size        batch size
  Output:
    lrs               look-up table of LR's, with length equal to total # of iterations
  Then you can use this in your PyTorch code by counting iteration number and setting
          optimizer.param_groups[0]['lr'] = lrs[iter_count]
  """
  if verbose > 0:
    print("Setting up 1Cycle LR schedule...")
  pct_start, div_factor = 0.3, 10.
  lr_start = lr_max/div_factor
  lr_end = lr_start/1e4
  n_iter = n_data_points * epochs // batch_size     # number of iterations
  a1 = int(n_iter * pct_start)
  a2 = n_iter - a1

  # make look-up table
  lrs_first = np.linspace(lr_start, lr_max, a1)            # linear growth
  lrs_second = (lr_max-lr_end)*(1+np.cos(np.linspace(0,np.pi,a2)))/2 + lr_end  # cosine annealing
  lrs = np.concatenate((lrs_first, lrs_second))
  return lrs


from keras.callbacks import Callback
import keras.backend as K

class OneCycleScheduler(Callback):
    """My modification of Keras' Learning rate scheduler to do 1Cycle learning
       which increments per BATCH, not per epoch
    Keyword arguments
        **kwargs:  keyword arguments to pass to get_1cycle_schedule()
        Also, verbose: int. 0: quiet, 1: update messages.

    Sample usage (from my train.py):
        lrsched = OneCycleScheduler(lr_max=1e-4, n_data_points=X_train.shape[0], epochs=epochs, batch_size=batch_size, verbose=1)
    """
    def __init__(self, **kwargs):
        super(OneCycleScheduler, self).__init__()
        self.verbose = kwargs.get('verbose', 0)
        self.lrs = get_1cycle_schedule(**kwargs)
        self.iteration = 0

    def on_batch_begin(self, batch, logs=None):
        lr = self.lrs[self.iteration]
        K.set_value(self.model.optimizer.lr, lr)         # here's where the assignment takes place
        if self.verbose > 0:
            print('\nIteration %06d: OneCycleScheduler setting learning '
                  'rate to %s.' % (self.iteration, lr))
        self.iteration += 1

    def on_epoch_end(self, epoch, logs=None):  # this is unchanged from Keras LearningRateScheduler
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)






def train_network(weights_file="weights.hdf5", classpath="Preproc/Train/",
    epochs=50, batch_size=20, val_split=0.2, tile=False, max_per_class=0):
    #np.random.seed(1)  # fix a number to get reproducibility

    # Get the data
    X_train, Y_train, paths_train, class_names = build_dataset(path=classpath,
        batch_size=batch_size, tile=tile, max_per_class=max_per_class)

    # Instantiate the model
    model, serial_model = setup_model(X_train, class_names, weights_file=weights_file)

    save_best_only = (val_split > 1e-6)

    split_index = int(X_train.shape[0]*(1-val_split))
    X_val, Y_val = X_train[split_index:], Y_train[split_index:]
    X_train, Y_train = X_train[:split_index-1], Y_train[:split_index-1]

    # shrink to simulate a small number of examples
    #new_size = len(class_names)*10
    #X_train, Y_train = X_train[:new_size], Y_train[:new_size]

    checkpointer = MultiGPUModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=save_best_only,
          serial_model=serial_model, period=1, class_names=class_names)
    #earlystopping = EarlyStopping(patience=12)
    #lr_scheduler = keras.callbacks.LearningRateScheduler(lr_sched, verbose=0)
    #lr_scheduler = OneCycleScheduler(lr_max=5e-3, n_data_points=X_train.shape[0], epochs=epochs, batch_size=batch_size, verbose=0)

    steps_per_epoch = X_train.shape[0] // batch_size
    if (len(class_names) > 2) or (steps_per_epoch > 1):
        training_generator = MixupGenerator(X_train, Y_train, batch_size=batch_size, alpha=0.25)()
        model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch,
              epochs=epochs, shuffle=True,
              verbose=1, callbacks=[checkpointer], validation_data=(X_val, Y_val))
    else:
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
              verbose=1, callbacks=[checkpointer], #validation_split=val_split)
              validation_data=(X_val, Y_val))

    # overwrite text file class_names.txt  - does not put a newline after last class name
    with open('class_names.txt', 'w') as outfile:
        outfile.write("\n".join(class_names))

    # Score the model against Test dataset
    X_test, Y_test, paths_test, class_names_test  = build_dataset(path=classpath+"../Test/", tile=tile)
    assert( class_names == class_names_test )
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="trains network using training dataset")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file (in .hdf5)', default="weights.hdf5")
    parser.add_argument('-c', '--classpath', #type=argparse.string,
        help='Train dataset directory with list of classes', default="Preproc/Train/")
    parser.add_argument('--epochs', default=20, type=int, help="Number of iterations to train for")
    parser.add_argument('--batch_size', default=40, type=int, help="Number of clips to send to GPU at once")
    parser.add_argument('--val', default=0.2, type=float, help="Fraction of train to split off for validation")
    parser.add_argument("--tile", help="tile mono spectrograms 3 times for use with imagenet models",action="store_true")
    parser.add_argument('-m', '--maxper', type=int, default=0, help="Max examples per class")
    args = parser.parse_args()
    train_network(weights_file=args.weights, classpath=args.classpath, epochs=args.epochs, batch_size=args.batch_size,
        val_split=args.val, tile=args.tile, max_per_class=args.maxper)
