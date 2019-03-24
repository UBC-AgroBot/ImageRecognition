# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf

class ImageClassifier: 
  graph = None
  label_file = "/tmp/output_labels.txt"
  model_file = "/tmp/output_graph.pb"

  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255

  input_layer = "Placeholder"
  output_layer = "final_result"
  
  def __init__(self, model_file=model_file, label_file_path=label_file, input_height=input_height, 
              input_width=input_width, input_mean=input_mean, input_std=input_std,
              input_layer=input_layer, output_layer=output_layer):

    self.graph = self.load_graph(model_file)
    self.label_file = label_file_path
    self.input_height = input_height
    self.input_width = input_width
    self.input_mean = input_mean
    self.input_std = input_std

    self.input_layer = input_layer
    self.output_layer = output_layer

  
  def classify(self, file_name):
    t = self.read_tensor_from_image_file(
      file_name,
      input_height=self.input_height,
      input_width=self.input_width,
      input_mean=self.input_mean,
      input_std=self.input_std)

    input_name = "import/" + self.input_layer
    output_name = "import/" + self.output_layer
    input_operation = self.graph.get_operation_by_name(input_name)
    output_operation = self.graph.get_operation_by_name(output_name)
  
    with tf.Session(graph=self.graph) as sess:
      results = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t
      })

    results = np.squeeze(results)
  
    top_k = results.argsort()[-5:][::-1]
    labels = self.load_labels(self.label_file)
    for i in top_k:
      print(labels[i], results[i])

    return labels[0]


  def load_graph(self, model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph

  def load_labels(self, label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
      label.append(l.rstrip())
    return label

  def read_tensor_from_image_file(self, 
                                file_name,
                                input_height=input_height,
                                input_width=input_width,
                                input_mean=input_mean,
                                input_std=input_std):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
      image_reader = tf.image.decode_png(
          file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
      image_reader = tf.squeeze(
          tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
      image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
      image_reader = tf.image.decode_jpeg(
          file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
  
    return result