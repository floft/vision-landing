#!/usr/bin/env python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""objection_detection for tflite

Taken from:
https://github.com/freedomtan/tensorflow/blob/deeplab_tflite_python/tensorflow/contrib/lite/examples/python/object_detection.py

Used to plot results from concat and concat_1 outputs of the official TF Lite
implementation and my own to compare.

Loads one of three files -- change the "filename" variable:
    - official implementation: ./object_detection.py --offline --debug --model tflite_float.py
        - outputs tflite_official.npy
    - numpy implementation:    ./tflite_numpy.py
        - outputs tflite_manual.npy
    - OpenCL implementation:   ./tflite_opencl.py
        - outputs tflite_opencl.npy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from image import find_files, load_image_into_numpy_array

NUM_RESULTS = 1917
NUM_CLASSES = 1

X_SCALE = 10.0
Y_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0

def load_box_priors(filename):
  with open(filename) as f:
    count = 0
    for line in f:
      row = line.strip().split(' ')
      box_priors.append(row)
      #print(box_priors[count][0])
      count = count + 1
      if count == 4:
        return

def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels

def decode_center_size_boxes(locations):
  """calculate real sizes of boxes"""
  for i in range(0, NUM_RESULTS):
    ycenter = locations[i][0] / Y_SCALE * np.float(box_priors[2][i]) \
            + np.float(box_priors[0][i])
    xcenter = locations[i][1] / X_SCALE * np.float(box_priors[3][i]) \
            + np.float(box_priors[1][i])
    h = math.exp(locations[i][2] / H_SCALE) * np.float(box_priors[2][i])
    w = math.exp(locations[i][3] / W_SCALE) * np.float(box_priors[3][i])

    ymin = ycenter - h / 2.0
    xmin = xcenter - w / 2.0
    ymax = ycenter + h / 2.0
    xmax = xcenter + w / 2.0

    locations[i][0] = ymin
    locations[i][1] = xmin
    locations[i][2] = ymax
    locations[i][3] = xmax
  return locations

def iou(box_a, box_b):
  x_a = max(box_a[0], box_b[0])
  y_a = max(box_a[1], box_b[1])
  x_b = min(box_a[2], box_b[2])
  y_b = min(box_a[3], box_b[3])

  intersection_area = (x_b - x_a + 1) * (y_b - y_a + 1)

  box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
  box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

  iou = intersection_area / float(box_a_area + box_b_area - intersection_area)
  return iou

def nms(p, iou_threshold, max_boxes):
  sorted_p = sorted(p, reverse=True)
  selected_predictions = []
  for a in sorted_p:
    if len(selected_predictions) > max_boxes:
      break
    should_select = True
    for b in selected_predictions:
      if iou(a[3], b[3]) > iou_threshold:
        should_select = False
        break
    if should_select:
      selected_predictions.append(a)

  return selected_predictions

def load_test_image(test_image_dir, width=300, height=300,
        input_mean=127.5, input_std=127.5, index=-1):
    """ Load one test image """
    test_images = [os.path.join(d, f) for d, f in find_files(test_image_dir)]
    img = Image.open(test_images[index])
    img = img.resize((width, height))
    img = load_image_into_numpy_array(img)
    return img

if __name__ == "__main__":
  #filename = "tflite_manual.npy"
  #filename = "tflite_official.npy"
  filename = "tflite_opencl.npy"
  label_file = "labels.txt"
  box_prior_file = "box_priors.txt"
  input_mean = 127.5
  input_std = 127.5
  min_score = 0
  max_boxes = 10
  floating_model = True
  show_image = True
  alt_output_order = False

  print("Loading", filename)

  # NxHxWxC, H:1, W:2
  height = 300
  width = 300
  img = load_test_image("test_images")

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

  box_priors = []
  load_box_priors(box_prior_file)
  #labels = load_labels(label_file)
  labels = ['???', 'fryingpan']

  data = np.load(filename).item()
  #data = np.load("tflite_official.npy").item()
  #concat = data["concat"]
  #concat_1 = data["concat_1"]
  concat = data["Squeeze"]
  concat_1 = data["convert_scores"]

  print(concat)
  print(concat_1)

  predictions = np.squeeze(concat)
  output_classes = np.squeeze(concat_1)

  decode_center_size_boxes(predictions)

  pruned_predictions = [[],]
  for c in range(1, len(output_classes[0])):
    pruned_predictions.append([])
    for r in range(0, NUM_RESULTS):
      #score = 1. / (1. + math.exp(-output_classes[r][c]))
      score = output_classes[r][c]
      if score > 0.01:
        rect = (predictions[r][1] * width, predictions[r][0] * width, \
                predictions[r][3] * width, predictions[r][2] * width)

        pruned_predictions[c].append((output_classes[r][c], r, labels[c], rect))

  final_predictions = []
  for c in range(0, len(pruned_predictions)):
    predictions_for_class = pruned_predictions[c]
    suppressed_predictions = nms(predictions_for_class, 0.5, max_boxes)
    final_predictions = final_predictions +  suppressed_predictions

  #print(predictions)
  #print(pruned_predictions)
  #print(final_predictions)

  if show_image:
    fig, ax = plt.subplots(1)

  final_predictions = sorted(final_predictions, reverse=True)[:max_boxes]
  for e in final_predictions:
    score = 100. / (1. + math.exp(-e[0]))
    score_string = '{0:2.0f}%'.format(score)
    print(score_string, e[2], e[3])
    if score < min_score:
      break
    left, top, right, bottom = e[3]
    rect = patches.Rectangle((left, top), (right - left), (bottom - top), \
             linewidth=1, edgecolor='r', facecolor='none')

    if show_image:
      # Add the patch to the Axes
      ax.add_patch(rect)
      ax.text(left, top, e[2]+': '+score_string, fontsize=6,
              bbox=dict(facecolor='y', edgecolor='y', alpha=0.5))

  if show_image:
    ax.imshow(img)
    plt.show()
