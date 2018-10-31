#!/usr/bin/env python3
"""
Stream RPi video to another computer to do processing there
"""
import zmq
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray

class LiveStream:
    """ Publish images to port """
    def __init__(self, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1) # Only keep last message
        self.socket.bind("tcp://*:"+str(port))

    def run(self, capture_width, capture_height, stream_width, stream_height, framerate):
        """
        Capture at one size, resize to another, publish this on the socket

        Resizing on the Pi is advantageous since it does it on the GPU and then
        we can stream faster since we're not sending as much data.
        """
        camera = PiCamera()
        camera.resolution = (capture_width, capture_height)
        camera.framerate = framerate
        raw_capture = PiRGBArray(camera, size=(stream_width, stream_height))

        for input_image in camera.capture_continuous(
                raw_capture, format="rgb", use_video_port=True,
                resize=(stream_width, stream_height)):
            try:
                frame = np.ascontiguousarray(input_image.array)
                self.socket.send_pyobj(frame)
                raw_capture.truncate(0)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    s = LiveStream(5555)
    s.run(640, 480, 300, 300, 15)
