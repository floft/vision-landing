#!/usr/bin/env python3
"""
Photo/video control with a couple R/C controller channels

Channel 6 (three-way switch G -- channel_capture)
    low  - RTSP video stream
    med  - capture either videos or photos
    high - turn off
Channel 7 (two-way switch D -- channel_capture_mode)
    low  - set video capture
    high - set photo capture

Default - no capture, stream with v4l2rtspserver (separate process, 1280x720, 15fps)
Photos - capture with PiCamera (max resolution), turn off live stream
Videos - stream and save with v4l2rtspserver (separate process, -O option, 1280x720, 30fps)
"""
import argparse
import threading
from subprocess import call
from enum import Enum

from filesystem import get_record_dir
from photo_capture import PhotoCapture
from video_capture import VideoCapture
from v4l2rtspserver import V4L2RTSPServer
from autopilot_communication import AutopilotCommuncation

Mode = Enum("Mode", "PHOTO VIDEO")

class ControlCapture:
    def __init__(self, device, baudrate, source_system,
        channel_capture, channel_capture_mode,
        capture_width, capture_height,
        capture_fps, stream_fps,
        path, enabled_mode=1, exit_mode=2):
        # create separate 00001, etc. subdirectory for each time we start this
        # since the timestamps on the RPi will almost always be wrong
        path = get_record_dir(path)
        self.path = path

        # Object to capture photos
        self.p = PhotoCapture(path)

        # Object to record/stream videos
        self.v = V4L2RTSPServer(path)
        #self.v = VideoCapture(path)

        # Create objects to communicate with autopilot
        self.ap = AutopilotCommuncation(
            device, baudrate, source_system,
            [channel_capture, channel_capture_mode],
            on_mode=lambda c, m: self.on_mode(c, m))

        # Which mode we should record in
        self.enabled_mode = enabled_mode

        # Start off in photo mode
        self.capture_mode = Mode.PHOTO

        # What switch does what
        self.channel_capture = channel_capture
        self.channel_capture_mode = channel_capture_mode

        # Shutdown in this mode
        self.exit_mode = exit_mode

        # Params for video streaming/capturing -- boolean is whether to capture
        self.stream_params = (capture_width, capture_height, stream_fps, False)
        self.capture_params = (capture_width, capture_height, capture_fps, True)

        # Create threads
        self.t_ap = threading.Thread(target=self.ap.run)
        self.t_p = None

    def run(self):
        # Always start autopilot part, but not necessarily video yet
        self.t_ap.start()

    def start_photo(self):
        if self.t_p is not None:
            self.p.exit()
            del self.t_p

        self.t_p = threading.Thread(target=self.p.run)
        self.p.exiting = False # Reset
        self.t_p.start()

    def start_video(self, capture):
        """ No need for another thread since this opens a separate process """
        self.v.stop()

        if capture:
            self.v.start(*self.capture_params)
        else:
            self.v.start(*self.stream_params)

    def on_mode(self, channel, mode):
        # Only change what we'll capture next time
        if channel == self.channel_capture_mode:
            if mode == 0:
                print("Photo mode")
                self.capture_mode = Mode.PHOTO
            else:
                print("Video mode")
                self.capture_mode = Mode.VIDEO
        # Start/stop capturing, poweroff, etc. operations
        elif channel == self.channel_capture:
            if mode == self.enabled_mode:
                if self.capture_mode == Mode.PHOTO:
                    print("Starting photo capture")
                    self.v.stop() # Exit video
                    self.start_photo()
                elif self.capture_mode == Mode.VIDEO:
                    print("Starting video capture")
                    self.p.exit() # Exit photo
                    self.start_video(True)
            elif mode == self.exit_mode:
                print("Shutting down")
                self.ap.exit()
                self.v.exit()
                self.p.exit()
                self.shutdown()
            else:
                print("Stopping capture")
                # Stop video or photo capture if either is running
                self.v.stop()
                self.p.exit()

                # Then start streaming
                print("Starting streaming")
                self.start_video(False)

    def exit(self):
        self.ap.exit()
        self.v.exit()
        self.p.exit()
        if self.t_p is not None:
            self.t_p.join()
        self.t_ap.join()

    def shutdown(self):
        """ Shutdown the Pi """
        call("sudo poweroff", shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Autopilot
    parser.add_argument("--baudrate", type=int, default=115200,
        help="Master port baud rate (default 115200)")
    parser.add_argument("--device", type=str, default="/dev/ttyAMA0",
        help="Serial device (default /dev/ttyAMA0)")
    parser.add_argument("--source-system", type=int, default=255,
        help="MAVLink source system for this GCS (default 255)")
    parser.add_argument("--channel-capture", type=int, default=6,
        help="What aux channel to use for enabling capturing / poweroff (default 6)")
    parser.add_argument("--channel-mode", type=int, default=7,
        help="What aux channel to use for choosing video / photos (default 7)")

    # Where to store photos/videos
    parser.add_argument("--path", type=str, default="capture",
        help="Where to store the photos/videos (default \"capture\")")

    # Video streaming/capturing
    parser.add_argument("--capture-width", type=int, default=1280,
        help="Width to capture video at (default 1280)")
    parser.add_argument("--capture-height", type=int, default=720,
        help="Height to capture video at (default 720)")
    parser.add_argument("--capture-fps", type=int, default=30,
        help="Framerate to capture at in Hz (default 30)")
    parser.add_argument("--stream-fps", type=int, default=15,
        help="Framerate to stream at in Hz (default 15)")

    args = parser.parse_args()

    capture = ControlCapture(
        args.device, args.baudrate, args.source_system,
        args.channel_capture, args.channel_mode,
        args.capture_width, args.capture_height,
        args.capture_fps, args.stream_fps,
        args.path)

    try:
        capture.run()
    except KeyboardInterrupt:
        print("Exiting")
        capture.exit()
