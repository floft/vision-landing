#!/usr/bin/env python3
"""
Capture video
"""
import os
import time
import argparse
import subprocess
from picamera import PiCamera

class VideoCapture:
    """ Capture and/or stream video until told to exit """
    def __init__(self, path):
        # Put files in this directory
        self.path = path
        os.makedirs(path, exist_ok=True)

        # Don't overwrite the images we took last time we enabled photo capture
        self.increment = 0

        # Is it running? capturing?
        self.running = False
        self.capture = False

        # Start the cvlc process first so we've got something to output to. It'll
        # always run, but we won't always feed it data
        self.cvlc = subprocess.Popen([
            'cvlc',
            '--sout-transcode-hurry-up',
            '--no-drop-late-frames',
            '--no-skip-frames',
            'stream:///dev/stdin',
            '--sout', '#rtp{sdp=rtsp://:8555/unicast}',
            ':demux=h264',
            ], stdin=subprocess.PIPE)
        # self.cvlc = subprocess.Popen([
        #     "gst-launch-1.0",
        #     "filesrc", "location=/dev/stdin",
        #     "!", "udpsink", "host=0.0.0.0", "port=8555"
        # ])

    def start(self, width, height, fps, capture):
        """ See: https://raspberrypi.stackexchange.com/a/54735 """
        assert not self.running, "Cannot start video when already started"

        self.running = True
        self.capture = capture

        path = os.path.join(self.path, "video%05d.h264"%self.increment)
        self.increment += 1

        self.camera = PiCamera(resolution=(width, height), framerate=fps)

        # Stream with RTSP via VLC
        self.camera.start_recording(self.cvlc.stdin, 'h264', splitter_port=1)

        # Save to disk
        if self.capture:
            self.camera.start_recording(path, 'h264', splitter_port=2)

    def stop(self):
        if self.running:
            self.camera.stop_recording(splitter_port=1)
            if self.capture:
                self.camera.stop_recording(splitter_port=2)
            self.camera.close()
            self.running = False

    def exit(self):
        self.stop()
        self.cvlc.stdin.close()
        self.cvlc.terminate()
        self.cvlc.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="capture",
        help="Where to store the videos (default \"capture\")")
    parser.add_argument("--width", type=int, default=1280,
        help="Width to capture video at (default 1280)")
    parser.add_argument("--height", type=int, default=720,
        help="Height to capture video at (default 720)")
    parser.add_argument("--fps", type=int, default=30,
        help="Framerate to capture at in Hz (default 30)")
    parser.add_argument('--capture', dest='capture', action='store_true',
        help="Capture video to file")
    parser.add_argument('--no-capture', dest='capture', action='store_false',
        help="Do not capture video to file (default)")
    parser.set_defaults(capture=False)
    args = parser.parse_args()

    c = VideoCapture(args.path)

    try:
        c.start(args.width, args.height, args.fps, args.capture)

        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        c.exit()
