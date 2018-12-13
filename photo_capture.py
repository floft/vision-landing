#!/usr/bin/env python3
"""
Continuously capture photos
"""
import os
import time
import argparse
from picamera import PiCamera

class PhotoCapture:
    """ Capture images at max resolution as fast as possible until told to exit """
    def __init__(self, path):
        # Put files in this directory
        self.path = path
        os.makedirs(path, exist_ok=True)

        # Don't overwrite the images we took last time we enabled photo capture
        self.increment = 0

        # Stop capture when desired
        self.completed = True
        self.exiting = False

    def run(self, width=3280, height=2464):
        self.completed = False

        with PiCamera(resolution=(width, height)) as camera:
            path = os.path.join(self.path, "image%05d"%self.increment+"_{counter:05d}.jpg")
            self.increment += 1

            for filename in camera.capture_continuous(path):
                print("Captured", filename)

                if self.exiting:
                    break

        self.completed = True

    def exit(self):
        self.exiting = True

        # Not a good way of doing it.... but this will wait to return until
        # PiCamera has released it's resources
        while not self.completed:
            time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="capture",
        help="Where to store the photos (default \"capture\")")
    args = parser.parse_args()

    c = PhotoCapture(args.path)

    try:
        c.run()
    except KeyboardInterrupt:
        pass
