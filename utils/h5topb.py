#! /usr/bin/env python3
# credit: John D'Souza, https://medium.com/@johnsondsouza23/export-keras-model-to-protobuf-for-tensorflow-serving-101ad6c65142

from keras import backend as K
from keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

# Function to export Keras model to Protocol Buffer format
# Inputs:
#       path_to_h5: Path to Keras h5 model
#       export_path: Path to store Protocol Buffer model

def export_h5_to_pb(path_to_h5, export_path):

    # Set the learning phase to Test since the model is already trained.
    K.set_learning_phase(0)

    # Load the Keras model
    keras_model = load_model(path_to_h5)

    # Build the Protocol Buffer SavedModel at 'export_path'
    builder = saved_model_builder.SavedModelBuilder(export_path)

    # Create prediction signature to be used by TensorFlow Serving Predict API
    signature = predict_signature_def(inputs={"images": keras_model.input},
                                      outputs={"scores": keras_model.output})

    with K.get_session() as sess:
        # Save the meta graph and the variables
        builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                         signature_def_map={"predict": signature})

    builder.save()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert Keras HDF5 model to Tensoflow')
    parser.add_argument('path_to_h5', help="hdf5 file(s) to convert")
    parser.add_argument('export_path', help="pb file(s) to generate")
    args = parser.parse_args()
    export_h5_to_pb(args.path_to_h5, args.export_path)
