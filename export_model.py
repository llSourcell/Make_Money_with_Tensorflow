#!/usr/bin/python
# (c)-2017 Amiya Patanaik amiyain@gmail.com
# Licensed under GPL v3

from keras import backend as K
from keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
import shutil, os, argparse

# reset learning phase
K.set_learning_phase(0)

# capture commandline arguments
parser = argparse.ArgumentParser(description='Export Keras models to Tensorflow Serving format.')
parser.add_argument('-p','--path', help='path to Keras model file', required=True)
parser.add_argument('-n','--name', help='model name', required=True)
parser.add_argument('-v','--version', help='version number', required=True)

args = vars(parser.parse_args())

export_version = args['version']
model_name = args['name']
model_path = args['path']

# loading models
model = load_model(model_path)

export_model_path = "models/" + model_name  + "/" + export_version 

# export models

builder = saved_model_builder.SavedModelBuilder(export_model_path)


# if there are multiple inputs/outputs
# signature_model = predict_signature_def(inputs={"input1": model.input[0], "input2": model.input[1]}, outputs={"output": model.output})

# please note that what you call the input and output is important
# the same keys must be used in the client as well
signature = predict_signature_def(inputs={'input': model.input},
                                  outputs={'output': model.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
    builder.save()
