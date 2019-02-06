#!/usr/bin/env python3
"""
Run the video streaming when enabled via the R/C controller (2nd position)
Recieve bounding boxes and send to the Pixhawk
Shutdown via the 3rd position in the switch
"""
import zmq
import time
import argparse
import threading
import numpy as np

from subprocess import call
from collections import deque

from v4l2rtspserver import V4L2RTSPServer
from autopilot_communication import AutopilotCommuncation, \
    AutopilotCommuncationSend

class BufferManager:
    """
    Add items to a deque and provide a function to wait for new items, which
    will be run in a separate thread.

    Based on:
    https://github.com/ThermalSoaring/thermal-soaring/blob/master/soaring.py
    """
    def __init__(self, max_length=25):
        self.cond = threading.Condition()
        self.buffer = deque(maxlen=max_length)
        self.exiting = False

    def add_data(self, data):
        """ Add new data """
        with self.cond:
            self.buffer.append(data)
            self.cond.notify()

    def get_data(self):
        """ Get data now, if not available return None """
        with self.cond:
            data = self.buffer.copy()

            if data:
                d = data[0]
                self.buffer.popleft()
                return d

            return None

    def get_wait(self):
        """ Wait till we have new data """
        with self.cond:
            while not self.exiting:
                data = self.get_data()

                if data:
                    return data

                self.cond.wait()

    def exit(self):
        """ Make sure anything waiting on this dies """
        with self.cond:
            self.exiting = True
            self.cond.notify()

class ControlStreaming:
    def __init__(self, device, baudrate, source_system, aux_channel,
            stream_port, receive_port, width, height,
            framerate, enabled_mode=1, exit_mode=2,
            threshold=0.0):
        self.threshold = threshold

        # Create objects to communicate with autopilot and stream video
        self.ap = AutopilotCommuncation(
            device, baudrate, source_system,
            [aux_channel], on_mode=lambda c, m: self.on_mode(c, m))

        # Totally separate connection (i.e. we need network connection not
        # serial, since you can't connect twice over serial)
        #
        # TODO ideally this would use just one connection but that may not
        # be possible due to the GIL and if we block on receive...
        self.buffer_manager = BufferManager()
        self.ap_send = AutopilotCommuncationSend(self.buffer_manager,
            device, baudrate, source_system)

        # Video streaming
        self.v = V4L2RTSPServer(None) # path isn't used
        self.v_running = False # keep track of if running

        # Which mode we should record in
        self.enabled_mode = enabled_mode

        # Shutdown in this mode
        self.exiting = False
        self.exit_mode = exit_mode

        # Params for video
        self.stream_params = (width, height, framerate, False)

        # Create threads
        self.t_ap = threading.Thread(target=self.ap.run)
        self.t_ap_send = threading.Thread(target=self.ap_send.run)

        # Receive back bounding boxes
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1) # Only get last message
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.bind("tcp://*:"+str(receive_port))

    def run(self):
        # Always start autopilot part, but not necessarily video yet
        self.t_ap.start()
        self.t_ap_send.start()

        # Get bounding boxes back
        while self.socket:
            detection = self.socket.recv_json()

            # If no bounding box, it sends None
            if detection is not None:
                try:
                    detection["score"] = float(detection["score"])
                except:
                    detection["score"] = 0.0

                # If high enough, add to the queue to send to the autopilot
                if detection["score"] > self.threshold:
                    self.buffer_manager.add_data(detection)

    def start_streaming(self):
        self.v_running = True
        self.v.start(*self.stream_params)

    def stop_streaming(self):
        self.v_running = False
        self.v.stop()

    def stop_ap_send(self):
        self.ap_send.exit()
        self.buffer_manager.exit()
        self.t_ap_send.join() # wait for send thread to exit

    def on_mode(self, channel, mode):
        if mode == self.enabled_mode:
            # Only start streaming if not already running
            if not self.v_running:
                print("Starting streaming")
                self.start_streaming()
            else:
                print("Streaming already started")
        elif mode == self.exit_mode:
            print("Shutdown mode")
            self.exiting = True
            self.ap.exit()
            self.stop_streaming()
            self.stop_ap_send()
            self.shutdown()
        else:
            print("Stopping streaming")
            # Stop the streaming
            self.stop_streaming()

    def exit(self):
        self.exiting = True
        self.ap.exit()
        self.stop_streaming()
        self.stop_ap_send()
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
    parser.add_argument("--stream-port", type=int, default=8555,
        help="What port to stream video on (default 8555)")
    parser.add_argument("--receive-port", type=int, default=5555,
        help="What port to receive bounding boxes on (default 5555)")
    parser.add_argument("--width", type=int, default=300,
        help="Width to stream video at (default 300)")
    parser.add_argument("--height", type=int, default=300,
        help="Height to stream video at (default 300)")
    parser.add_argument("--framerate", type=int, default=15,
        help="Framerate to stream at in Hz (default 15)")

    args = parser.parse_args()

    streaming = ControlStreaming(
        args.device, args.baudrate, args.source_system, args.aux_channel,
        args.stream_port, args.receive_port, args.width, args.height,
        args.framerate)

    try:
        streaming.run()
    except KeyboardInterrupt:
        print("Exiting")
        streaming.exit()
