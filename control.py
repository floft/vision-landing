#!/usr/bin/env python3
"""
Run the video streaming when enabled via the R/C controller (2nd position)
Recieve bounding boxes and send to the Pixhawk
Shutdown via the 3rd position in the switch
"""
import zmq
import time
import argparse
import numpy as np
import threading
import multiprocessing

from subprocess import call
from collections import deque

# Allow working with a deque between threads
# http://stackoverflow.com/a/27345949
from multiprocessing.managers import SyncManager
SyncManager.register('deque', deque)

from v4l2rtspserver import V4L2RTSPServer
from autopilot_communication_processes import Buffer, AutopilotConnection, \
    AutopilotCommuncationReceive, AutopilotCommuncationSend

class ControlStreaming:
    def __init__(self, sync_manager, device, baudrate, source_system, aux_channel,
            stream_port, receive_port, width, height,
            framerate, enabled_mode=1, exit_mode=2,
            threshold=0.0):
        self.sync_manager = sync_manager
        self.threshold = threshold

        # Get connection to autpilot
        #
        # Note: apparently if we have one and block on receive, then it won't
        # send while being blocked? One fix: have two.
        #
        # Also note: source_system defaults to 1 now to indicate this code is
        # actually on the drone itself (source_system=1 is the autopilot), but
        # with different source_components since this isn't the autopilto but
        # a companion computer.
        self.ap_connection1 = AutopilotConnection(device, baudrate, source_system)
        self.ap_connection2 = AutopilotConnection(device, baudrate, source_system)

        # Receive messages from the autopilot - the enable switch
        self.receive_buffer = Buffer(sync_manager)
        self.ap_receive = AutopilotCommuncationReceive(
            self.ap_connection1, [aux_channel],
            on_mode=lambda c, m: self.on_mode_set(c, m))

        # Since receive is in a separate process, we'll add its messages to
        # a buffer in on_mode_set and then receive those in this process with
        # on_mode_run and send them to the on_mode() function to actually handle
        # them. Thus, we need to run on_mode_run() in a separate thread
        self.t_mode = threading.Thread(target=self.on_mode_run)

        # Send data to auto pilot through a buffer
        self.send_buffer = Buffer(sync_manager)
        self.ap_send = AutopilotCommuncationSend(self.ap_connection2, self.send_buffer)

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

        # Receive back bounding boxes
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1) # Only get last message
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.bind("tcp://*:"+str(receive_port))

    def run(self):
        """ Start everything and wait """
        # Always start autopilot part, but not necessarily video yet
        self.t_mode.start()
        self.ap_connection1.connect()
        self.ap_connection2.connect()
        self.ap_receive.start()
        self.ap_send.start()

        # We could either run send_run() or on_mode_run() in a separate thread
        # and the other in this (the main) thread. I chose to run on_mode_run()
        # in a separate thread, so here we'll run send_run().
        self.send_run()

    def send_run(self):
        """
        In this thread, we'll receive detection bounding boxes and
        send them to the autopilot process buffer to send to the autopilot.
        """
        # Get bounding boxes back
        while self.socket and not self.exiting:
            detection = self.socket.recv_json()

            # If no bounding box, it sends None
            if detection is not None:
                try:
                    detection["score"] = float(detection["score"])
                except:
                    detection["score"] = 0.0

                # If high enough, add to the queue to send to the autopilot
                if detection["score"] > self.threshold:
                    self.send_buffer.add_data(detection)

    def on_mode_set(self, channel, mode):
        """ This is executed in the AutopilotCommunicationReceive process, so
        we add the data to a buffer that'll move it to the other process """
        self.receive_buffer.add_data((channel, mode))

    def on_mode_run(self):
        """ Wait for new modes to be added to the receive_buffer and pass to
        on_mode() when added """
        while not self.exiting:
            values = self.receive_buffer.get_wait()

            if values:
                self.on_mode(values[0], values[1])

    def on_mode(self, channel, mode):
        """ Function to handle mode changes """
        if mode == self.enabled_mode:
            # Only start streaming if not already running
            if not self.v_running:
                print("Starting streaming")
                self.start_streaming()
            else:
                print("Streaming already started")
        elif mode == self.exit_mode:
            print("Shutdown mode")
            self.exit(from_mode_thread=True) # this is running in on_mode_run() thread
            self.shutdown()
        else:
            print("Stopping streaming")
            # Stop the streaming
            self.stop_streaming()

    def start_streaming(self):
        self.v_running = True
        self.v.start(*self.stream_params)

    def stop_streaming(self):
        self.v_running = False
        self.v.stop()

    def stop_ap_receive(self):
        self.receive_buffer.exit()
        self.ap_receive.exit()
        self.ap_receive.join()

    def stop_ap_send(self):
        self.send_buffer.exit()
        self.ap_send.exit()
        self.ap_send.join()

    def exit(self, from_mode_thread=False):
        self.exiting = True
        self.stop_streaming()
        self.stop_ap_receive()
        self.stop_ap_send()

        # If we call exit() from the mode thread, we cannot join it to its
        # own thread.
        if not from_mode_thread:
            self.t_mode.join()

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
    parser.add_argument("--source-system", type=int, default=1,
        help="MAVLink source system (default 1, i.e. companion computer on the drone)")
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

    with SyncManager() as sync_manager:
        streaming = ControlStreaming(
            sync_manager,
            args.device, args.baudrate, args.source_system, args.aux_channel,
            args.stream_port, args.receive_port, args.width, args.height,
            args.framerate)

        try:
            streaming.run()
        except KeyboardInterrupt:
            print("Exiting")
            streaming.exit()
