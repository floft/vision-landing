#!/usr/bin/env python3
"""
Run the video streaming when enabled via the R/C controller
Shutdown via the 3rd position in the switch
"""
import argparse
import threading
from subprocess import call

from video_streaming import LiveStream
from autopilot_communication import AutopilotCommuncation

class ControlStreaming:
    def __init__(self, device, baudrate, source_system, aux_channel, port,
            capture_width, capture_height, stream_width, stream_height,
            framerate, enabled_mode=1, exit_mode=2):
        # Create objects to communicate with autopilot and stream video
        self.ap = AutopilotCommuncation(
            device, baudrate, source_system,
            [aux_channel], on_mode=lambda c, m: self.on_mode(c, m))
        self.s = LiveStream(port)

        # Which mode we should record in
        self.enabled_mode = enabled_mode

        # Shutdown in this mode
        self.exit_mode = exit_mode

        # Params for video
        self.vid_params = (
            capture_width, capture_height,
            stream_width, stream_height,
            framerate)

        # Create threads
        self.t_ap = threading.Thread(target=self.ap.run)
        self.t_vid = None

    def run(self):
        # Always start autopilot part, but not necessarily video yet
        self.t_ap.start()

    def start_streaming(self):
        if self.t_vid is not None:
            del self.t_vid

        self.t_vid = threading.Thread(target=self.s.run, args=self.vid_params)
        self.s.exiting = False # Reset
        self.t_vid.start()

    def on_mode(self, channel, mode):
        if mode == self.enabled_mode:
            # Only start streaming if not already running
            if self.t_vid is None or not self.t_vid.is_alive():
                print("Starting streaming")
                self.start_streaming()
            else:
                print("Streaming already started")
        elif mode == self.exit_mode:
            print("Shutdown mode")
            self.ap.exit()
            self.s.exit()
            self.shutdown()
        else:
            print("Stopping streaming")
            # Stop the streaming
            self.s.exit()

    def exit(self):
        self.ap.exit()
        self.s.exit()
        self.t_vid.join()
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
    parser.add_argument("--aux-channel", type=int, default=6,
        help="What aux channel to use for enabling (default 6)")

    # Video streaming
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

    streaming = ControlStreaming(
        args.device, args.baudrate, args.source_system, args.aux_channel, args.port,
        args.capture_width, args.capture_height, args.stream_width, args.stream_height,
        args.framerate)

    try:
        streaming.run()
    except KeyboardInterrupt:
        print("Exiting")
        streaming.exit()
