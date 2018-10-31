#!/usr/bin/env python3
"""
Object Detector

Modification of my object detector for the RAS project:
https://github.com/WSU-RAS/object_detection/blob/master/scripts/object_detector.py

Also, referenced for the RPi camera stuff:
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py
"""
import os
import time
import numpy as np
import tensorflow as tf
from collections import deque

from picamera import PiCamera
from picamera.array import PiRGBArray

 # Visualization for debugging
from PIL import Image
from matplotlib import pyplot as plt

# Note: must have models/research/ and models/research/slim/ in PYTHONPATH
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    """
    Helper function from: models/research/object_detection/
    """
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def find_files(folder, prefix="rgb", extension=".png"):
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

class ObjectDetector:
    """
    Object Detection with TensorFlow via their models/research/object_detection

    Based on:
    https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb


    Usage:
        with ObjectDetectorTF("path/to/model_dir.pb", "path/to/tf_label_map.pbtxt") as detector:
            boxes, scores, classes = detector.process(newImage)

    Or:
        detector = ObjectDetectorTF("path/to/model_dir.pb", "path/to/tf_label_map.pbtxt")
        detector.open()
        boxes, scores, classes = detector.process(newImage)
        detector.close()
    """
    def __init__(self, graph_path, labels_path, threshold, memory):
        # Threshold for what to count as objects
        self.threshold = threshold

        # Max memory usage (0 - 1)
        self.memory = memory

        # Load frozen TensorFlow model into memory
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(os.path.join(graph_path, "frozen_inference_graph.pb"), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        # Load label map
        label_map = label_map_util.load_labelmap(labels_path)
        numClasses = len(label_map.item) # Use all classes
        categories = label_map_util.convert_label_map_to_categories(label_map,
                        max_num_classes=numClasses, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def open(self):
        # Config options: max GPU memory to use.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.memory)
        config = tf.ConfigProto(gpu_options=gpu_options)

        # Session
        self.session = tf.Session(graph=self.detection_graph, config=config)

        #
        # Inputs/outputs to network
        #
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def close(self):
        self.session.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def process(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Run detection
        (boxes, scores, classes, num) = self.session.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        # Class label needs to be int32 or else publishing gives error
        classes = classes.astype(np.int32)

        return boxes, scores, classes

    def show(self, image_np, boxes, classes, scores, debug_image_size=(12,8)):
        """
        For debugging, show the image with the bounding boxes
        """
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True, line_thickness=8)
        plt.figure(figsize=debug_image_size)
        plt.imshow(image_np)
        plt.show()

    def detected_results_dims(self, img_width, img_height, boxes, scores, classes):
        """
        Detection results specifying the top left (x,y) and the width and
        height from that point
        """
        results = []
        scores_above_threshold = np.where(scores > self.threshold)[1]

        for s in scores_above_threshold:
            bb = boxes[0,s,:]
            sc = scores[0,s]
            cl = classes[0,s]

            results.append({
                "label_str": self.category_index[int(cl)]['name'],
                "label_int": cl,
                "score": sc,
                "x": int((img_width-1) * bb[1]),
                "y": int((img_height-1) * bb[0]),
                "width": int((img_width-1) * (bb[3]-bb[1])),
                "height": int((img_height-1) * (bb[2]-bb[0])),
            })

        return results

    def detected_results_corners(self, img_width, img_height, boxes, scores, classes):
        """
        Detection results specifying the top left (xmin,ymin) and bottom right
        (xmax,ymax) values
        """
        results = []
        scores_above_threshold = np.where(scores > self.threshold)[1]

        for s in scores_above_threshold:
            bb = boxes[0,s,:]
            sc = scores[0,s]
            cl = classes[0,s]

            results.append({
                "label_str": self.category_index[int(cl)]['name'],
                "label_int": cl,
                "score": sc,
                "xmin": int((img_width-1) * bb[1]),
                "ymin": int((img_height-1) * bb[0]),
                "xmax": int((img_width-1) * bb[3]),
                "ymax": int((img_height-1) * bb[2]),
            )

        return results

class ObjectDetectorBase:
    """ Wrap detector to calculate FPS """
    def __init__(self, graph_path, labels_path, average_fps_frames=30, memory=0.9, threshold=0.5, debug=False):
        self.detector = ObjectDetector(graph_path, labels_path, threshold, memory)
        self.detector.open()
        self.fps = deque(maxlen=average_fps_frames) # compute average FPS over # of frames
        self.fps_start_time = 0

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.detector.close()

    def avg_fps(self):
        """ Return average FPS over last so many frames (specified in constructor) """
        return sum(list(self.fps))/len(self.fps)

    def process(self, image):
        if self.debug:
            # Start timer
            fps = time.time()

        boxes, scores, classes = self.detector.process(image)

        if self.debug:
            # End timer
            fps = 1/(time.time() - fps)
            self.fps.append(fps)

            print("Object Detection FPS", "{:<5}".format("%.2f"%fps), \
                    "Average", "{:<5}".format("%.2f"%self.avg_fps()))

        return boxes, scores, classes

    def run(self):
        raise NotImplemented("Must implement run() function")

class LiveObjectDetector(ObjectDetectorBase):
    """ Run object detection on live images """
    def __init__(self, *args, **kwargs):
        super(LiveObjectDetector, self).__init__(*args, **kwargs)

    def run(self, width, height, framerate):
        camera = PiCamera()
        camera.resolution = (width, height)
        camera.framerate = framerate
        rawCapture = PiRGBArray(camera, size=(width, height))
        rawCapture.truncate(0)

        for input_image in camera.capture_continuous(
                rawCapture, format="bgr", use_video_port=True):
            frame = input_image.array
            frame.setflags(write=1)
            boxes, scores, classes = self.process(frame)
            results = self.detector.detected_results_corners(width, height, boxes, scores, classes)

            if self.debug:
                for i, r in enumerate(results):
                    print("Result "+str(i)+":", r)

class OfflineObjectDetector(ObjectDetectorBase):
    """ Run object detection on already captured images """
    def __init__(self, test_image_dir, *args, **kwargs):
        self.test_image_dir = test_image_dir
        super(OfflineObjectDetector, self).__init__(*args, **kwargs)

    def run(self, graph_name, label_name, test_image_dir):
        test_images = [os.path.join(d, f) for d, f in find_files(self.test_image_dir)]

        for img in test_images:
            image_np = load_image_into_numpy_array(Image.open(img))
            boxes, scores, classes, number = self.process(image_np)

            if self.debug:
                self.detector.show(img, boxes, scores, classes)

if __name__ == "__main__":
    debug = True
    live = True

    # What model and labels to use
    graph_name = "detect.tflite"
    label_map = "tf_label_map.pbtxt"

    # Live options
    width = 640
    height = 480
    framerate = 10

    # Non-live (offline) options -- the images to test
    offline_image_dir = "test_images"

    # Run detection
    if live:
        with LiveObjectDetector(graph_name, label_map, debug=debug) as d:
            d.run(width, height, framerate)
    else:
        with OfflineObjectDetector(offline_image_dir, graph_name, label_map, debug=debug) as d:
            d.run()
