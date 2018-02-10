from keras.layers import concatenate
from keras.layers.core import Lambda
from keras.models import Model
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import Callback, ModelCheckpoint
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from panotti import models

def make_serial(model, parallel=True):   # Undoes make_parallel, but keyword included in case it's called on a serial model
    if (parallel):
        return model.layers[-2]
    else:
        return model                    # if model's already serial, return original model


def get_available_gpus():  # from https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    local_device_protos = device_lib.list_local_devices()
    gpu_list =  [x.name for x in local_device_protos if x.device_type == 'GPU']
    count = len(gpu_list)
    print(" Available GPUs = ",gpu_list,", count = ",count)
    return count


def make_parallel(serial_model, gpu_count=-1):
    return multi_gpu_model(serial_model, gpus=gpu_count)



class MultiGPUModelCheckpoint(Callback):
    """Save the *serial version* of the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, serial_model=None, class_names=None):
        super(MultiGPUModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.class_names = class_names

        if (serial_model is None):
            self.serial_model = model
        else:
            self.serial_model = serial_model

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.serial_model.save_weights(filepath, overwrite=True)
                        else:
                            #self.serial_model.save(filepath, overwrite=True)
                            models.save_model_ext(self.serial_model, filepath, overwrite=True, class_names=self.class_names)

                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.serial_model.save_weights(filepath, overwrite=True)
                else:
                    #self.serial_model.save(filepath, overwrite=True)
                    models.save_model_ext(self.serial_model, filepath, overwrite=True, class_names=self.class_names)


'''
Old code, before FChollet integrated into Keras 2.0.9:
# from https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
# SHHawley updated to Keras 2, added make_serial & get_available_gpus, and gpu_count=-1 flag


def make_parallel(model, gpu_count=-1):
    if (gpu_count < 0):                 # gpu_count < 0 will mean  "use all available GPUs"
        gpus =  get_available_gpus()
        gpu_count = len(gpus)
        print("\n make_parallel:",gpu_count,"GPUs detected.")
    print(" make_parallel: Data parallel execution across",gpu_count,"GPUs")
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            #merged.append(merge(outputs, mode='concat', concat_axis=0))
            merged.append(concatenate(outputs, axis=0))

        return Model(inputs=model.inputs, outputs=merged)
'''
