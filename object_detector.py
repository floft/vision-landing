#!/usr/bin/env python3
"""
Object Detector

Modification of my object detector for the RAS project:
https://github.com/WSU-RAS/object_detection/blob/master/scripts/object_detector.py

Also, referenced for the RPi camera stuff:
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py

And for some TF Lite stuff:
https://github.com/freedomtan/tensorflow/blob/deeplab_tflite_python/tensorflow/contrib/lite/examples/python/object_detection.py
"""
import os
import time
import zmq
import argparse
import numpy as np
import tensorflow as tf
from collections import deque

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper

from image import find_files, load_image_into_numpy_array

try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
except ImportError:
    print("Warning: cannot import picamera, live mode won't work")

def load_labels(filename):
    """
    Load labels from the label file

    Note: this is not the tf_label_map.pbtxt, instead just one label per line.
    """
    labels = []

    with open(filename, 'r') as f:
        for l in f:
            labels.append(l.strip())

    return labels

def detection_results(boxes, classes, scores, img_width, img_height,
        labels, min_score):
    """ Get readable results and apply min score threshold """
    detections = []
    scores_above_threshold = np.where(scores > min_score)[1]

    for s in scores_above_threshold:
        bb = boxes[0,s,:]
        sc = scores[0,s]
        cl = classes[0,s]

        detections.append({
            "label_str": labels[int(cl)],
            "label_int": cl,
            "score": sc,
            "xmin": int((img_width-1) * bb[1]),
            "ymin": int((img_height-1) * bb[0]),
            "xmax": int((img_width-1) * bb[3]),
            "ymax": int((img_height-1) * bb[2]),
        })

    return detections

def detection_show(image_np, detections, debug=True, show_image=True, debug_image_size=(12,8)):
    """ For debugging, show the image with the bounding boxes """
    if len(detections) == 0:
        return

    if show_image:
        plt.ion()
        fig, ax = plt.subplots(1, figsize=debug_image_size, num=1)

    for r in detections:
        if debug:
            print(r)

        if show_image:
            topleft = (r["xmin"], r["ymin"])
            width = r["xmax"] - r["xmin"]
            height = r["ymax"] - r["ymin"]

            rect = patches.Rectangle(topleft, width, height, \
                linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.text(r["xmin"], r["ymin"], r["label_str"]+": %.2f"%r["score"], fontsize=6,
                bbox=dict(facecolor="y", edgecolor="y", alpha=0.5))

    if show_image:
        ax.imshow(image_np)
        fig.canvas.flush_events()
        #plt.pause(0.05)

class TFObjectDetector:
    """
    Object Detection with TensorFlow model trained with
    models/research/object_detection (Non-TF Lite version)

    Based on:
    https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

    Usage:
        with TFObjectDetector("path/to/model_dir.pb", "path/to/labels.txt", 0.5)
            detections = d.process(newImage, orig_img_width, orig_img_height)
    """
    def __init__(self, graph_path, labels_path, min_score, memory=0.9, width=300, height=300):
        # Prune based on score
        self.min_score = min_score

        # Model dimensions
        self.model_input_height = height
        self.model_input_width = width

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

        # Load label map -- index starts with 1 for the non-TF Lite version
        self.labels = ["???"] + load_labels(labels_path)

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

    def model_input_dims(self):
        """ Get desired model input dimensions """
        return (self.model_input_width, self.model_input_height)

    def close(self):
        self.session.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def process(self, image_np, img_width, img_height):
        # Expand dimensions since the model expects images to have shape:
        #   [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Run detection
        (boxes, scores, classes, num) = self.session.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        # Make results readable
        return detection_results(boxes, classes, scores,
                img_width, img_height, self.labels, self.min_score)

class TFLiteObjectDetector:
    """
    Object Detection with TensorFlow model trained with
    models/research/object_detection (TF Lite version)

    Based on:
    https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

    Usage:
        d = TFLiteObjectDetector("path/to/model_file.tflite", "path/to/tf_label_map.pbtxt", 0.5)
        detections = d.process(newImage, orig_img_width, orig_img_height)
    """
    def __init__(self, model_file, labels_path, min_score):
        # Prune based on score
        self.min_score = min_score

        # TF Lite model
        self.interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        if self.input_details[0]['dtype'] == type(np.float32(1.0)):
            self.floating_model = True
        else:
            self.floating_model = False

        # NxHxWxC, H:1, W:2
        self.model_input_height = self.input_details[0]['shape'][1]
        self.model_input_width = self.input_details[0]['shape'][2]

        # Load label map
        self.labels = load_labels(labels_path)

    def model_input_dims(self):
        """ Get desired model input dimensions """
        return (self.model_input_width, self.model_input_height)

    def process(self, image_np, img_width, img_height, input_mean=127.5, input_std=127.5,
            output_numpy_concat=False):
        # Normalize if floating point (but not if quantized)
        if self.floating_model:
            image_np = (np.float32(image_np) - input_mean) / input_std

        # Expand dimensions since the model expects images to have shape:
        #   [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Pass image to the network
        self.interpreter.set_tensor(self.input_details[0]['index'], image_np_expanded)

        # Run
        self.interpreter.invoke()

        # Get results
        detection_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        detection_classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        detection_scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        num_detections = self.interpreter.get_tensor(self.output_details[3]['index'])

        num_detections = self.interpreter.get_tensor(self.output_details[3]['index'])

        if output_numpy_concat:
            np.save("tflite_official.npy", {
                self.interpreter._get_tensor_details(i)["name"]: self.interpreter.get_tensor(i) for i in range(176)
            })

        if not self.floating_model:
            box_scale, box_mean = self.output_details[0]['quantization']
            class_scale, class_mean = self.output_details[1]['quantization']

            # If these are zero, then we end up setting all our results to zero
            if box_scale != 0:
                detection_boxes = (detection_boxes - box_mean * 1.0) * box_scale
            if class_mean != 0:
                detection_classes = (detection_classes - class_mean * 1.0) * class_scale

        # Make results readable
        return detection_results(detection_boxes, detection_classes, detection_scores,
                img_width, img_height, self.labels, self.min_score)

class ObjectDetectorBase:
    """ Wrap detector to calculate FPS """
    def __init__(self, model_file, labels_path, min_score=0.5,
            average_fps_frames=30, debug=False, lite=True):
        self.debug = debug
        self.lite = lite

        if lite:
            self.detector = TFLiteObjectDetector(model_file, labels_path, min_score)
        else:
            self.detector = TFObjectDetector(model_file, labels_path, min_score)

        self.fps = deque(maxlen=average_fps_frames) # compute average FPS over # of frames
        self.fps_start_time = 0

    def open(self):
        if not self.lite:
            self.detector.open()

    def __enter__(self):
        self.open()
        return self

    def close(self):
        if not self.lite:
            self.detector.close()

    def __exit__(self, type, value, traceback):
        self.close()

    def avg_fps(self):
        """ Return average FPS over last so many frames (specified in constructor) """
        return sum(list(self.fps))/len(self.fps)

    def process(self, *args, **kwargs):
        if self.debug:
            # Start timer
            fps = time.time()

        detections = self.detector.process(*args, **kwargs)

        if self.debug:
            # End timer
            fps = 1/(time.time() - fps)
            self.fps.append(fps)

            print("Object Detection FPS", "{:<5}".format("%.2f"%fps), \
                    "Average", "{:<5}".format("%.2f"%self.avg_fps()))

        return detections

    def run(self):
        raise NotImplemented("Must implement run() function")

class RemoteObjectDetector(ObjectDetectorBase):
    """ Run object detection on images streamed from a remote camera """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, host, port, show_image=False):
        while True:
            try:
                context = zmq.Context()
                socket = context.socket(zmq.SUB)
                socket.setsockopt(zmq.SNDHWM, 1)
                socket.setsockopt(zmq.RCVHWM, 1)
                socket.setsockopt(zmq.CONFLATE, 1) # Only get last message
                socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
                socket.connect("tcp://"+host+":"+str(port))

                while socket:
                    frame = socket.recv_pyobj()
                    detections = self.process(frame, frame.shape[1], frame.shape[0])

                    if self.debug:
                        for i, d in enumerate(detections):
                            print("Result "+str(i)+":", d)

                    detection_show(frame, detections, self.debug, show_image)

                time.sleep(0.5)
            except KeyboardInterrupt:
                break

class LiveObjectDetector(ObjectDetectorBase):
    """ Run object detection on live images """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, capture_width, capture_height, framerate):
        camera = PiCamera()
        camera.resolution = (capture_width, capture_height)
        camera.framerate = framerate
        raw_capture = PiRGBArray(camera, size=self.detector.model_input_dims())

        for input_image in camera.capture_continuous(
                raw_capture, format="rgb", use_video_port=True,
                resize=self.detector.model_input_dims()):
            frame = input_image.array
            frame.setflags(write=1) # not sure what this does?

            detections = self.process(frame, frame.shape[1], frame.shape[0])
            raw_capture.truncate(0)

            if self.debug:
                for i, d in enumerate(detections):
                    print("Result "+str(i)+":", d)

class OfflineObjectDetector(ObjectDetectorBase):
    """ Run object detection on already captured images """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, test_image_dir, show_image=True):
        test_images = [os.path.join(d, f) for d, f in find_files(test_image_dir)]

        for i, filename in enumerate(test_images):
            orig_img = Image.open(filename)
            resize_img = orig_img.resize(self.detector.model_input_dims())

            orig_img = load_image_into_numpy_array(orig_img)
            resize_img = load_image_into_numpy_array(resize_img)

            detections = self.process(resize_img, orig_img.shape[1], orig_img.shape[0],
                    output_numpy_concat=(self.debug and i == 0))

            detection_show(orig_img, detections, self.debug, show_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="detect_quantized.tflite", type=str,
        help="Model file (if TF lite) or directory (if graph) (default detect_quantized.tflite)")
    parser.add_argument("--labels", default="labels.txt", type=str,
        help="Label file (one per line) (default labels.txt")

    parser.add_argument("--images", default="test_images", type=str,
        help="If offline, directory of test .jpg images (default test_images/)")
    parser.add_argument("--host", default="rpi-zero", type=str,
        help="Hostname to connect to if in remote mode (default rpi-zero)")
    parser.add_argument("--port", default=5555, type=int,
        help="Port to connect to if in remote mode (default 5555)")

    parser.add_argument("--remote", dest='remote', action='store_true',
        help="Run detection on remote streamed video")
    parser.add_argument("--no-remote", dest='remote', action='store_false',
        help="Do not run detection on remote streamed video (default)")

    parser.add_argument("--live", dest='live', action='store_true',
        help="Run detection on local live camera video")
    parser.add_argument("--no-live", dest='live', action='store_false',
        help="Do not run detection on local live camera video (default)")

    parser.add_argument("--offline", dest='offline', action='store_true',
        help="Run detection on --images directory of test images")
    parser.add_argument("--no-offline", dest='offline', action='store_false',
        help="Do not run detection on directory of test images (default)")

    parser.add_argument("--show", dest='show', action='store_true',
        help="Show image with detection results")
    parser.add_argument("--no-show", dest='show', action='store_false',
        help="Do not show image with detection results (default)")

    parser.add_argument("--lite", dest='lite', action='store_true',
        help="Use TF Lite (default)")
    parser.add_argument("--no-lite", dest='lite', action='store_false',
        help="Do not use TF Lite")

    parser.add_argument("--debug", dest='debug', action='store_true',
        help="Output debug information (fps and detection results) (default) ")
    parser.add_argument("--no-debug", dest='debug', action='store_false',
        help="Do not output debug information ")

    parser.set_defaults(
        remote=False, live=False, offline=False,
        lite=True, show=False, debug=False)
    args = parser.parse_args()

    assert args.remote + args.live + args.offline == 1, \
        "Must specify exactly one of --remote, --live, or --offline"

    # Run detection
    if args.remote:
        with RemoteObjectDetector(args.model, args.labels, debug=args.debug, lite=args.lite) as d:
            d.run(args.host, args.port, show_image=args.show)
    elif args.live:
        with LiveObjectDetector(args.model, args.labels, debug=args.debug, lite=args.lite) as d:
            d.run(640, 480, 15)
    elif args.offline:
        with OfflineObjectDetector(args.model, args.labels, debug=args.debug, lite=args.lite) as d:
            d.run(args.images, show_image=args.show)
