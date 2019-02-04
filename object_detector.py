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
import re
import time
import zmq
import pathlib
import argparse
import threading
import numpy as np
import tensorflow as tf
from collections import deque

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper

# Gstreamer
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
gi.require_version('GObject', '2.0')
from gi.repository import GLib, GObject, Gst

from filesystem import latest_index, get_record_dir
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

def detection_show(image_np, detections, show_image=True, debug_image_size=(12,8)):
    """ For debugging, show the image with the bounding boxes """
    if len(detections) == 0:
        return

    if show_image:
        plt.ion()
        fig, ax = plt.subplots(1, figsize=debug_image_size, num=1)

    for r in detections:
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

def low_level_detection_show(image_np, detections, color=[255,0,0], amt=1):
    """ Overwrite portions on input image with red to display via GStreamer rather than
    with matplotlib which is slow """
    for r in detections:
        # left edge
        image_np[r["ymin"]-amt:r["ymax"]+amt, r["xmin"]-amt:r["xmin"]+amt, :] = color
        # right edge
        image_np[r["ymin"]-amt:r["ymax"]+amt, r["xmax"]-amt:r["xmax"]+amt, :] = color
        # top edge
        image_np[r["ymin"]-amt:r["ymin"]+amt, r["xmin"]-amt:r["xmax"]+amt, :] = color
        # bottom edge
        image_np[r["ymax"]-amt:r["ymax"]+amt, r["xmin"]-amt:r["xmax"]+amt, :] = color

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

        # if output_numpy_concat:
        #     # For use in comparing with tflite_numpy.py
        #     #
        #     # For internals of Interpreter, see:
        #     # https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/lite/python/interpreter.py
        #     np.save("tflite_official.npy", {
        #         self.interpreter._get_tensor_details(i)["name"]: self.interpreter.get_tensor(i) for i in range(176)
        #         #self.interpreter._get_tensor_details(i)["name"]: np.copy(self.interpreter.tensor(i)()) for i in range(176)
        #     })

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
            average_fps_frames=30, debug=False, lite=True,
            gst=False, gst_width=300, gst_height=300, gst_framerate=15,
            gst_already_setup=False):
        self.debug = debug
        self.lite = lite
        self.gst = gst
        self.exiting = False

        if lite:
            self.detector = TFLiteObjectDetector(model_file, labels_path, min_score)
        else:
            self.detector = TFObjectDetector(model_file, labels_path, min_score)

        # compute average FPS over # of frames
        self.fps = deque(maxlen=average_fps_frames)

        # compute streaming FPS (how fast frames are arriving from camera
        # and we're able to process them, i.e. this is the actual FPS)
        self.stream_fps = deque(maxlen=average_fps_frames)
        self.process_end_last = 0

        # Run GStreamer in separate thread
        if self.gst:
            self.t_gst = threading.Thread(target=self.gst_run)

            if not gst_already_setup:
                Gst.init(None)

            self.pipe = Gst.Pipeline.new("object-detection")

            # appsrc -> videoconvert (since data is RGB) -> autovideosink
            self.src = Gst.ElementFactory.make("appsrc")
            convert = Gst.ElementFactory.make("videoconvert")
            sink = Gst.ElementFactory.make("autovideosink")

            # For now we'll just assume it's a fixed size and a fixed framerate,
            # though it'll probably be less than this frame rate. It'll just
            # keep showing the old image till a new one arrives.
            caps = Gst.Caps.from_string("video/x-raw,"
                +"format=(string)RGB,"
                +"width="+str(gst_width)+","
                +"height="+str(gst_height)+","
                +"framerate="+str(gst_framerate)+"/1")
            self.src.set_property("caps", caps)
            self.src.set_property("format", Gst.Format.TIME)

            self.pipe.add(self.src, convert, sink)
            self.src.link(convert)
            convert.link(sink)

            # Event loop
            self.loop = GLib.MainLoop()

            # Get error messages or end of stream on bus
            bus = self.pipe.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.gst_bus_call, self.loop)

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

    def avg_stream_fps(self):
        """ Return average streaming FPS over last so many frames (specified in constructor) """
        return sum(list(self.stream_fps))/len(self.stream_fps)

    def process(self, *args, **kwargs):
        if self.debug:
            # Start timer
            fps = time.time()

        detections = self.detector.process(*args, **kwargs)

        if self.debug:
            now = time.time()

            # End timer
            fps = 1/(now - fps)
            self.fps.append(fps)

            # Streaming FPS
            stream_fps = 1/(now - self.process_end_last)
            self.stream_fps.append(stream_fps)
            self.process_end_last = now

            print("Object Detection",
                "Process FPS", "{:<5}".format("%.2f"%self.avg_fps()),
                "Stream FPS", "{:<5}".format("%.2f"%self.avg_stream_fps()))

        return detections

    def run(self):
        raise NotImplementedError("Must implement run() function")

    def gst_bus_call(self, bus, message, loop):
        """ Print important messages """
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print("Error: %s: %s" % (err, debug))
            loop.quit()
        return True

    def gst_next_frame(self, frame):
        """ When we have a new numpy array RGB image, push it to GStreamer """
        data = frame.tobytes()
        buf = Gst.Buffer.new_wrapped(data)
        self.src.emit("push-buffer", buf)

    def gst_run(self):
        """ This is run in a separate thread. Start, loop, and cleanup. """
        self.pipe.set_state(Gst.State.PLAYING)

        try:
            self.loop.run()
        except:
            pass

        self.pipe.set_state(Gst.State.NULL)

    def gst_start(self):
        """ If using GStreamer, start that thread """
        if self.gst:
            self.t_gst.start()

    def gst_next_detection(self, frame, detections):
        """ Push new image to GStreamer """
        if self.gst:
            low_level_detection_show(frame, detections)
            self.gst_next_frame(frame)

    def gst_stop(self):
        """ If using GStreamer, tell it to exit and then wait """
        if self.gst:
            self.loop.quit()
            self.t_gst.join()

class RemoteObjectDetector(ObjectDetectorBase):
    """
    Run object detection on images streamed from a remote camera,
    also supports displaying live stream via GStreamer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, host, port, show_image=False, record=""):
        # Don't overwrite previous images if there are any
        if record != "":
            if not os.path.exists(record):
                os.makedirs(record)
                img_index = 1
            else:
                img_index = latest_index(record, "*.jpg")+1

        self.gst_start()

        while not self.exiting:
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

                    if record != "":
                        filename = os.path.join(record, "%05d.jpg"%img_index)
                        Image.fromarray(frame).save(filename)
                        img_index += 1
                        print("Saved", filename)

                    if self.debug:
                        for i, d in enumerate(detections):
                            print("Result "+str(i)+":", d)

                    detection_show(frame, detections, show_image)

                    # We do this last since low_level_detection_show modifies
                    # the image
                    self.gst_next_detection(frame, detections)

                time.sleep(0.5)
            except KeyboardInterrupt:
                self.exiting = True
                self.gst_stop()

def get_buffer_size(caps):
    """
    Returns width, height of buffer from caps
    Taken from: http://lifestyletransfer.com/how-to-get-buffer-width-height-from-gstreamer-caps/

    :param caps: https://lazka.github.io/pgi-docs/Gst-1.0/classes/Caps.html
    :type caps: Gst.Caps

    :rtype: bool, (int, int)
    """
    caps_struct = caps.get_structure(0)

    (success, width) = caps_struct.get_int('width')
    if not success:
        return False, (0, 0)

    (success, height) = caps_struct.get_int('height')
    if not success:
        return False, (0, 0)

    return True, (width, height)

class RemoteObjectDetectorUdp(ObjectDetectorBase):
    """
    Run object detection on images streamed from a remote camera over UDP,
    also supports displaying live stream via GStreamer
    """
    def __init__(self, host, port, record, *args, **kwargs):
        # Needed when processing frames in GStreamer
        self.record = record
        self.img_index = 1

        Gst.init(None)

        # uridecodebin -> appsink
        self.remote_pipe = Gst.parse_launch("uridecodebin " \
            + "uri=rtsp://"+host+":"+str(port)+"/unicast source::latency=0 " \
            + "! videoconvert ! video/x-raw,format=RGB " \
            + "! appsink name=appsink")
        remote_sink = self.remote_pipe.get_by_name("appsink")

        # Process new frames
        remote_sink.set_property("sync", False)
        remote_sink.set_property("drop", True)
        remote_sink.set_property("emit-signals", True)
        remote_sink.connect("new-sample", lambda x: self.remote_process_frame(x))

        # Event loop
        self.remote_loop = GLib.MainLoop()

        # Get error messages or end of stream on bus
        bus = self.remote_pipe.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.gst_bus_call, self.remote_loop)

        # Make sure we don't reinit GStreamer
        super().__init__(*args, **kwargs, gst_already_setup=True)

    def run(self):
        # Don't overwrite previous images if there are any
        if self.record != "":
            if not os.path.exists(self.record):
                os.makedirs(self.record)
                self.img_index = 1
            else:
                self.img_index = latest_index(self.record, "*.jpg")+1

        self.gst_start()

        try:
            self.remote_gst_run()
        except KeyboardInterrupt:
            self.exiting = True
            self.remote_loop.quit()
            self.gst_stop()

    def remote_process_frame(self, appsink):
        # Get frame
        sample = appsink.emit("pull-sample")
        got_caps, (width, height) = get_buffer_size(sample.get_caps())

        assert got_caps, \
            "Could not get width/height from buffer!"

        # See: https://github.com/TheImagingSource/tiscamera/blob/master/examples/python/opencv.py
        buf = sample.get_buffer()

        try:
            _, mapinfo = buf.map(Gst.MapFlags.READ)
            # Create a numpy array from the data
            frame = np.asarray(bytearray(mapinfo.data), dtype=np.uint8)
            # Give the array the correct dimensions of the video image
            # Note: 3 channels: R, G, B
            frame = frame.reshape((height, width, 3))

            detections = self.process(frame, width, height)

            if self.record != "":
                filename = os.path.join(self.record, "%05d.jpg"%self.img_index)
                Image.fromarray(frame).save(filename)
                self.img_index += 1
                print("Saved", filename)

            if self.debug:
                for i, d in enumerate(detections):
                    print("Result "+str(i)+":", d)

            # We do this last since low_level_detection_show modifies
            # the image
            self.gst_next_detection(frame, detections)
        finally:
            buf.unmap(mapinfo)

        return False

    def remote_gst_run(self):
        """ This is run in a separate thread. Start, loop, and cleanup. """
        self.remote_pipe.set_state(Gst.State.PLAYING)

        try:
            self.remote_loop.run()
        except:
            pass

        self.remote_pipe.set_state(Gst.State.NULL)

class LiveObjectDetector(ObjectDetectorBase):
    """ Run object detection on live images """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, capture_width, capture_height, framerate):
        camera = PiCamera()
        camera.resolution = (capture_width, capture_height)
        camera.framerate = framerate
        raw_capture = PiRGBArray(camera, size=self.detector.model_input_dims())

        self.gst_start()

        try:
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

                self.gst_next_detection(frame, detections)
        except KeyboardInterrupt:
            pass

        self.gst_stop()

class OfflineObjectDetector(ObjectDetectorBase):
    """ Run object detection on already captured images """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, test_image_dir, show_image=True):
        self.gst_start()

        test_images = [os.path.join(d, f) for d, f in find_files(test_image_dir)]

        try:
            for i, filename in enumerate(test_images):
                orig_img = Image.open(filename)

                if orig_img.size == self.detector.model_input_dims():
                    orig_img = load_image_into_numpy_array(orig_img)
                    resize_img = orig_img
                else:
                    resize_img = orig_img.resize(self.detector.model_input_dims())
                    orig_img = load_image_into_numpy_array(orig_img)
                    resize_img = load_image_into_numpy_array(resize_img)

                detections = self.process(resize_img, orig_img.shape[1], orig_img.shape[0])
                #        output_numpy_concat=(self.debug and i == len(test_images)-1))

                if self.debug:
                    for i, d in enumerate(detections):
                        print("Result "+str(i)+":", d)

                detection_show(orig_img, detections, show_image)
                self.gst_next_detection(orig_img, detections)
        except KeyboardInterrupt:
            pass

        self.gst_stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="detect_float_v2.tflite", type=str,
        help="Model file (if TF lite) or directory (if graph) (default detect_float_v2.tflite)")
    parser.add_argument("--labels", default="labels.txt", type=str,
        help="Label file (one per line) (default labels.txt")

    parser.add_argument("--images", default="test_images", type=str,
        help="If offline, directory of test .jpg images (default test_images/)")
    parser.add_argument("--host", default="192.168.4.1", type=str,
        help="Hostname to connect to if in remote mode (default 192.168.4.1)")
    parser.add_argument("--port", default=8555, type=int,
        help="Port to connect to if in remote mode (default 8555)")

    parser.add_argument("--remote-udp", dest='remote_udp', action='store_true',
        help="Run detection on remote streamed video over UDP (default)")
    parser.add_argument("--no-remote-udp", dest='remote_udp', action='store_false',
        help="Do not run detection on remote streamed video over UDP")

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

    parser.add_argument("--gst", dest='gst', action='store_true',
        help="Show streamed images with GStreamer (default)")
    parser.add_argument("--no-gst", dest='gst', action='store_false',
        help="Do not show streamed images with GStreamer")

    parser.add_argument("--lite", dest='lite', action='store_true',
        help="Use TF Lite (default)")
    parser.add_argument("--no-lite", dest='lite', action='store_false',
        help="Do not use TF Lite")

    parser.add_argument("--record", default="", type=str,
        help="Record the received remote frames in a specified directory (default disabled)")

    parser.add_argument("--debug", dest='debug', action='store_true',
        help="Output debug information (fps and detection results) (default) ")
    parser.add_argument("--no-debug", dest='debug', action='store_false',
        help="Do not output debug information ")

    parser.set_defaults(
        remote=False, remote_udp=True, live=False, offline=False,
        lite=True, show=False, gst=True, debug=True)
    args = parser.parse_args()

    assert args.remote + args.remote_udp + args.live + args.offline == 1, \
        "Must specify exactly one of --remote, --remote-udp --live, or --offline"

    record_dir = ""
    if args.record != "":
        record_dir = get_record_dir(args.record)
        print("Recording to:", record_dir)

    # Run detection
    if args.remote:
        with RemoteObjectDetector(args.model, args.labels,
                debug=args.debug, lite=args.lite, gst=args.gst) as d:
            d.run(args.host, args.port, show_image=args.show, record=record_dir)
    elif args.remote_udp:
        with RemoteObjectDetectorUdp(args.host, args.port, record_dir,
                args.model, args.labels,
                debug=args.debug, lite=args.lite, gst=args.gst) as d:
            d.run()
    elif args.live:
        with LiveObjectDetector(args.model, args.labels,
                debug=args.debug, lite=args.lite, gst=args.gst) as d:
            d.run(640, 480, 15)
    elif args.offline:
        with OfflineObjectDetector(args.model, args.labels,
                debug=args.debug, lite=args.lite, gst=args.gst) as d:
            d.run(args.images, show_image=args.show)
