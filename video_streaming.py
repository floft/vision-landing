#!/usr/bin/env python3
"""
Stream RPi video to another computer to do processing there
"""
import zmq
import argparse
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

        self.exiting = False

    def run(self, capture_width, capture_height, stream_width, stream_height, framerate):
        """
        Capture at one size, resize to another, publish this on the socket

        Resizing on the Pi is advantageous since it does it on the GPU and then
        we can stream faster since we're not sending as much data.
        """
        with PiCamera() as camera:
            camera.resolution = (capture_width, capture_height)
            camera.framerate = framerate
            raw_capture = PiRGBArray(camera, size=(stream_width, stream_height))

            for input_image in camera.capture_continuous(
                    raw_capture, format="rgb", use_video_port=True,
                    resize=(stream_width, stream_height)):
                frame = np.ascontiguousarray(input_image.array)
                self.socket.send_pyobj(frame)
                raw_capture.truncate(0)

                if self.exiting:
                    break

    def exit(self):
        self.exiting = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555,
        help="What port to listen on for streaming (default 5555)")
    parser.add_argument("--capture-width", type=int, default=640,
        help="Width to capture video at (default 640)")
    parser.add_argument("--capture-height", type=int, default=480,
        help="Height to capture video at (default 480)")
    parser.add_argument("--stream-width", type=int, default=300,
        help="Width to stream video at (default 300)")
    parser.add_argument("--stream-height", type=int, default=300,
        help="Height to stream video at (default 300)")
    parser.add_argument("--framerate", type=int, default=15,
        help="Framerate to stream at in Hz (default 15)")
    args = parser.parse_args()

    s = LiveStream(args.port)

    try:
        s.run(args.capture_width, args.capture_height,
            args.stream_width, args.stream_height,
            args.framerate)
    except KeyboardInterrupt:
        pass
