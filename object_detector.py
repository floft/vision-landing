#!/usr/bin/env python3
"""
Object Detector

Modification of my object detector for the RAS project:
https://github.com/WSU-RAS/object_detection/blob/master/scripts/object_detector.py

Also, referenced for the RPi camera stuff:
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py

And for TF Lite stuff -- i.e. most of the auxiliary functions like iou,
non_max_suppression, etc. are almost copy-pasted from here:
https://github.com/freedomtan/tensorflow/blob/deeplab_tflite_python/tensorflow/contrib/lite/examples/python/object_detection.py
"""
import os
import math
import time
import numpy as np
import tensorflow as tf
from collections import deque

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper

try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
except ImportError:
    print("Warning: cannot import picamera, will only work in offline mode")


X_SCALE = 10.0
Y_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0

def load_image_into_numpy_array(image):
    """
    Helper function from: models/research/object_detection/
    """
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def find_files(folder, prefix="", extension=".jpg"):
    """
    Find all files recursively in specified folder with a particular
    prefix and extension

    (for running on set of test images)
    """
    files = []

    for dirname, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.startswith(prefix) and filename.endswith(extension):
                files += [(dirname, filename)]

    return files

def load_box_priors(filename):
    """
    Load the box_priors.txt file

    You can get this file from:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/examples/android/app/src/main/assets/box_priors.txt
    """
    box_priors = []

    with open(filename, 'r') as f:
        count = 0

        for line in f:
            row = line.strip().split(' ')
            box_priors.append(row)
            count += 1

            if count == 4:
                break

    return box_priors

def decode_center_size_boxes(predictions, box_priors):
    """ Scale and use the box priors to compute the bounding box positions in
    terms of the image pixels """
    results = []

    for i in range(0, len(predictions)):
        ycenter = predictions[i][0] / Y_SCALE * np.float(box_priors[2][i]) \
                + np.float(box_priors[0][i])
        xcenter = predictions[i][1] / X_SCALE * np.float(box_priors[3][i]) \
                + np.float(box_priors[1][i])
        h = math.exp(predictions[i][2] / H_SCALE) * np.float(box_priors[2][i])
        w = math.exp(predictions[i][3] / W_SCALE) * np.float(box_priors[3][i])

        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0

        results.append((ymin, xmin, ymax, xmax))

    return results

def iou(box_a, box_b):
    """ Calculate intersection over union """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    intersection_area = (x_b - x_a + 1) * (y_b - y_a + 1)

    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = intersection_area / float(box_a_area + box_b_area - intersection_area)

    return iou

def non_max_suppression(predictions, iou_threshold, max_boxes):
    """ Perform non-max suppression """
    sorted_p = sorted(predictions, reverse=True)
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

def score_pruning(predictions, width, detection_classes, labels, min_score):
    """ Prune predictions by score """
    pruned_predictions = []

    for c in range(0, len(detection_classes[0])):
        pruned_predictions.append([])

        for r in range(0, len(predictions)):
            score = 1. / (1. + math.exp(-detection_classes[r][c]))

            if score > min_score:
                rect = (predictions[r][1] * width, predictions[r][0] * width, \
                predictions[r][3] * width, predictions[r][2] * width)

                pruned_predictions[c].append((detection_classes[r][c], r, labels[c], rect))

    return pruned_predictions

def nms_pruning(predictions, iou_threshold, max_boxes):
    """ Prune via non-max suppression """
    pruned_predictions = []

    for c in range(0, len(predictions)):
        predictions_for_class = predictions[c]
        suppressed_predictions = non_max_suppression(
                predictions_for_class, iou_threshold, max_boxes)
        pruned_predictions += suppressed_predictions

    return pruned_predictions

def load_labels(filename):
    """
    Load labels from the label file

    Note: this is not the tf_label_map.pbtxt, instead just one label per line.
    Must have a ??? line at the beginning for unknown label.
    """
    labels = []

    with open(filename, 'r') as f:
        for l in f:
            labels.append(l.strip())

    return labels

class TFLiteObjectDetector:
    """
    Object Detection with TensorFlow via their models/research/object_detection

    Based on:
    https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb


    Usage:
        with TFLiteObjectDetector("path/to/model_file.tflite", "path/to/tf_label_map.pbtxt") as detector:
            boxes, scores, classes = detector.process(newImage)

    Or:
        detector = TFLiteObjectDetector("path/to/model_file.tflite", "path/to/tf_label_map.pbtxt")
        detector.open()
        boxes, scores, classes = detector.process(newImage)
        detector.close()
    """
    def __init__(self, model_file, labels_path, box_priors_file, min_score, iou_threshold, max_boxes):
        # How to prune
        self.min_score = min_score
        self.iou_threshold = iou_threshold
        self.max_boxes = max_boxes

        # TF Lite model
        self.interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(self.input_details)
        print(self.output_details)

        if self.input_details[0]['dtype'] == type(np.float32(1.0)):
            self.floating_model = True
        else:
            self.floating_model = False

        # NxHxWxC, H:1, W:2
        self.model_input_height = self.input_details[0]['shape'][1]
        self.model_input_width = self.input_details[0]['shape'][2]

        # Load label map
        self.labels = load_labels(labels_path)

        # Load box priors
        self.box_priors = load_box_priors(box_priors_file)

    def model_input_dims(self):
        """ Get desired model input dimensions """
        return (self.model_input_width, self.model_input_height)

    def process(self, image_np, input_mean=127.5, input_std=127.5):
        # Normalize if floating point (in contrast to quantized)
        if self.floating_model:
            image_np = (np.float32(image_np) - input_mean) / input_std

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Pass image to the network
        self.interpreter.set_tensor(self.input_details[0]['index'], image_np_expanded)

        # Run
        start_time = time.time() # TODO remove
        self.interpreter.invoke()
        finish_time = time.time() # TODO remove
        print("time spent:", ((finish_time - start_time) * 1000), "ms") # TODO remove

        # Get results
        detection_boxes = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        detection_classes = self.interpreter.get_tensor(
            self.output_details[1]['index'])
        detection_scores = self.interpreter.get_tensor(
            self.output_details[2]['index'])
        num_detections = self.interpreter.get_tensor(
            self.output_details[3]['index'])

        # TODO remove
        print("boxes:", detection_boxes)
        print("classes:", detection_classes)
        print("scores:", detection_scores)
        print("num:", num_detections)

        # "squeeze" out the first dimension (of size 1)
        detection_boxes = np.squeeze(detection_boxes)

        # squeeze the first dimension but add another dimension to the end
        # since we only have one class
        detection_classes = np.expand_dims(np.squeeze(detection_classes), axis=1)

        if not self.floating_model:
            box_scale, box_mean = self.output_details[0]['quantization']
            class_scale, class_mean = self.output_details[1]['quantization']

            detection_boxes = (detection_boxes - box_mean * 1.0) * box_scale
            detection_classes = (detection_classes - class_mean * 1.0) * class_scale

        # Convert from relative to in terms of pixels
        detection_boxes = decode_center_size_boxes(detection_boxes, self.box_priors)

        # Prune based on score and do non-max suppresion
        detection_boxes = score_pruning(
                detection_boxes, self.model_input_width, detection_classes,
                self.labels, self.min_score)
        detection_boxes = nms_pruning(detection_boxes, self.iou_threshold, self.max_boxes)

        # Get top-n
        detection_boxes = sorted(detection_boxes, reverse=True)[:self.max_boxes]

        # Make results readable
        detections = []

        for e in detection_boxes:
            score = 100. / (1. + math.exp(-e[0]))
            detections.append({
                "label": e[2],
                "score": score,
                "xmin": e[3][1],
                "ymin": e[3][0],
                "xmax": e[3][3],
                "ymax": e[3][2],
            })

        return detections

    def show(self, image_np, detections, debug_image_size=(12,8)):
        """
        For debugging, show the image with the bounding boxes
        """
        fig, ax = plt.subplots(1, figsize=debug_image_size)

        for r in detections: 
            print(r)

            topleft = (r["xmin"], r["ymin"])
            width = r["xmax"] - r["xmin"]
            height = r["ymax"] - r["ymin"]
            score_string = "{0:2.0f}%".format(r["score"])

            rect = patches.Rectangle(topleft, width, height, \
                linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.text(r["xmin"], r["ymin"], r["label"]+": "+score_string, fontsize=6,
                bbox=dict(facecolor="y", edgecolor="y", alpha=0.5))

        ax.imshow(image_np)
        plt.show()

class ObjectDetectorBase:
    """ Wrap detector to calculate FPS """
    def __init__(self, model_file, labels_path, box_priors_file,
            min_score=0.01, iou_threshold=0.5, max_boxes=10,
            average_fps_frames=30, debug=False):
        self.debug = debug
        self.detector = TFLiteObjectDetector(
            model_file, labels_path, box_priors_file,
            min_score, iou_threshold, max_boxes)
        self.fps = deque(maxlen=average_fps_frames) # compute average FPS over # of frames
        self.fps_start_time = 0

    def avg_fps(self):
        """ Return average FPS over last so many frames (specified in constructor) """
        return sum(list(self.fps))/len(self.fps)

    def process(self, image):
        if self.debug:
            # Start timer
            fps = time.time()

        detections = self.detector.process(image)

        if self.debug:
            # End timer
            fps = 1/(time.time() - fps)
            self.fps.append(fps)

            print("Object Detection FPS", "{:<5}".format("%.2f"%fps), \
                    "Average", "{:<5}".format("%.2f"%self.avg_fps()))

        return detections

    def run(self):
        raise NotImplemented("Must implement run() function")

class LiveObjectDetector(ObjectDetectorBase):
    """ Run object detection on live images """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, width, height, framerate):
        camera = PiCamera()
        camera.resolution = (width, height)
        camera.framerate = framerate
        rawCapture = PiRGBArray(camera, size=(width, height))

        for input_image in camera.capture_continuous(
                rawCapture, format="bgr", use_video_port=True,
                resize=self.detector.model_input_dims()):
            frame = input_image.array
            frame.setflags(write=1) # not sure what this does?

            detections = self.process(frame)

            if self.debug:
                for i, d in enumerate(detections):
                    print("Result "+str(i)+":", d)

            rawCapture.truncate(0) # Needed?

class OfflineObjectDetector(ObjectDetectorBase):
    """ Run object detection on already captured images """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, test_image_dir):
        test_images = [os.path.join(d, f) for d, f in find_files(test_image_dir)]

        for filename in test_images:
            img = Image.open(filename)
            img = img.resize(self.detector.model_input_dims())
            img = load_image_into_numpy_array(img)

            detections = self.process(img)

            if self.debug:
                self.detector.show(img, detections)

if __name__ == "__main__":
    debug = True
    live = False

    # What model and labels to use
    model_file = "detect.tflite"
    label_map = "labels.txt"
    box_priors_file = "box_priors.txt"

    # Live options
    width = 640
    height = 480
    framerate = 10

    # Non-live (offline) options -- the images to test
    offline_image_dir = "test_images"

    # Run detection
    if live:
        d = LiveObjectDetector(model_file, label_map, box_priors_file, debug=debug)
        d.run(width, height, framerate)
    else:
        d = OfflineObjectDetector(model_file, label_map, box_priors_file, debug=debug)
        d.run(offline_image_dir)
