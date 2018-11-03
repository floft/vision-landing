#!/usr/bin/env python3
"""
Record images at 2 Hz as you hold the checkboard in front of the camera at
varying positions. Save these as .jpg files in a calibration/ folder.
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
from picamera import PiCamera
from picamera.array import PiRGBArray

def capture(output_dir, width, height, framerate,
        start_frame_number=1, show_image=True, figsize=(4,3)):
    if show_image:
        plt.ion()
        fig, ax = plt.subplots(1, figsize=figsize, num=1)

    camera = PiCamera()
    camera.resolution = (width, height)
    camera.framerate = framerate
    raw_capture = PiRGBArray(camera, size=(width, height))
    frame_number = start_frame_number

    for input_image in camera.capture_continuous(
            raw_capture, format="rgb", use_video_port=True):
        filename = os.path.join(output_dir, "%05d.jpg"%frame_number)
        print("Saving", filename)

        frame = Image.fromarray(input_image.array)
        frame.save(filename)

        if show_image:
            ax.imshow(frame)
            fig.canvas.flush_events()
            plt.pause(0.5)

        raw_capture.truncate(0)
        frame_number += 1

if __name__ == "__main__":
    output_dir = "calibration/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        capture(output_dir, 640, 480, 2)
    except KeyboardInterrupt:
        pass
