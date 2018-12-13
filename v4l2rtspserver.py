#!/usr/bin/env python3
"""
Run the V4L2RTSPServer for video streaming and/or saving videos to files
"""
import os
import time
import argparse
import subprocess

class V4L2RTSPServer:
    """
    Run v4l2rtspserver either just streaming or also saving to a file
    """
    def __init__(self, path,
            prog="/usr/local/bin/v4l2rtspserver",
            copy_prog="/usr/bin/local/v4l2copy",
            port=8555, device="/dev/video0", device_copy="/dev/video1"):
        self.prog = prog
        self.copy_prog = copy_prog
        self.port = port
        self.device = device
        self.device_copy = device_copy

        # Put files in this directory
        self.path = path
        os.makedirs(path, exist_ok=True)

        # Don't overwrite the videos we took last time we enabled video capture
        self.increment = 0
        # Processes
        self.p_rtsp = None
        self.p_capture = None
        self.p_copy = None

    def start(self, width, height, fps, capture):
        if capture:
            #cmd = [self.copy_prog, self.device, self.device_copy]
            #print("Executing", cmd)
            #self.p_copy = subprocess.Popen(cmd)

            path = os.path.join(self.path, "video%05d.mp4"%self.increment)
            self.increment += 1

            cmd = "ffmpeg -f v4l2 " \
                + "-video_size %dx%d -r %d "%(width, height, fps) \
                + "-input_format h264 " \
                + "-analyzeduration 0 " \
                + "-i %s -c:v copy -f mp4 "%self.device
            cmd = cmd.split() + [path]
            print("Executing", cmd)
            self.p_capture = subprocess.Popen(cmd)
        else:
            cmd = [self.prog, "-W%d"%width, "-H%d"%height, "-F%d"%fps,
                "-P%d"%self.port, self.device]
            print("Executing", cmd)
            self.p_rtsp = subprocess.Popen(cmd)

    def stop(self):
        # Tell to stop
        if self.p_rtsp is not None:
            self.p_rtsp.terminate()
        if self.p_capture is not None:
            self.p_capture.terminate()
        if self.p_copy is not None:
            self.p_copy.terminate()

        # Wait to complete
        if self.p_rtsp is not None:
            self.p_rtsp.wait()
            self.p_rtsp = None
        if self.p_capture is not None:
            self.p_capture.wait()
            self.p_capture = None
        if self.p_copy is not None:
            self.p_copy.wait()
            self.p_copy = None

    def exit(self):
        self.stop()

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

    c = V4L2RTSPServer(args.path)

    try:
        c.start(args.width, args.height, args.fps, args.capture)

        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        c.exit()
