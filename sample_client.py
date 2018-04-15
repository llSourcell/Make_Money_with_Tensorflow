#!/usr/bin/python
# This is directly adapted from:
# https://github.com/tobegit3hub/tensorflow_template_application/tree/master/python_predict_client

import numpy
from keras.datasets import mnist
from grpc.beta import implementations
import tensorflow as tf
from predict_client import predict_pb2
from predict_client import prediction_service_pb2

tf.app.flags.DEFINE_string("host", "0.0.0.0", "gRPC server host")
tf.app.flags.DEFINE_integer("port", 9000, "gRPC server port")
tf.app.flags.DEFINE_string("model_name", "mnist", "TensorFlow model name")
tf.app.flags.DEFINE_integer("model_version", -1, "TensorFlow model version")
tf.app.flags.DEFINE_float("request_timeout", 10.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(numpy.shape(x_train))
idx = 2324
img = x_train[idx,:,:]
label = y_train[idx]
img = numpy.resize(img, (1, 28, 28, 1))
print(label)


def main():
  host = FLAGS.host
  port = FLAGS.port
  model_name = FLAGS.model_name
  model_version = FLAGS.model_version
  request_timeout = FLAGS.request_timeout

  # Generate inference data
  keys = numpy.asarray([1, 2, 3])
  keys_tensor_proto = tf.contrib.util.make_tensor_proto(keys, dtype=tf.int32)
  features_tensor_proto = tf.contrib.util.make_tensor_proto(img,
                                                            dtype=tf.float32)

  # Create gRPC client and request
  channel = implementations.insecure_channel(host, port)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  if model_version > 0:
    request.model_spec.version.value = model_version
  request.inputs['inputs'].CopyFrom(features_tensor_proto)
  request.model_spec.signature_name = 'predict'
  #request.inputs['features'].CopyFrom(features_tensor_proto)

  # Send request
  result = stub.Predict(request, request_timeout)
  response = numpy.array(result.outputs['outputs'].float_val)
  prediction = numpy.argmax(response)

  print(prediction)


if __name__ == '__main__':
  main()