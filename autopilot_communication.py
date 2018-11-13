#!/usr/bin/env python3
"""
Enable some functionality upon a switch on the R/C controller

Examples: https://github.com/ArduPilot/pymavlink/blob/master/examples/
"""
import time
import argparse
import pymavlink
from pymavlink import mavutil, mavwp, mavparm

class AutopilotCommuncation:
    def __init__(self, device="/dev/ttyAMA0", baudrate=115200, source_system=255,
            channel=6, autoreconnect=True, cutoffs=[1282, 1716], rate=2,
            on_mode=None):
        """
        Device: e.g. /dev/ttyAMA0
        Baudrate: e.g. 115200
        Source_system: probably 255
        Channel: e.g. 6 for aux 6 (set via your RC controller)
        Autoreconnect: passed to MAVLink connection
        Cutoffs: my switch 6 goes between 1065, 1499, and 1933, and the defaults
            are between those
        Rate: the Hz you want to be parsing these messages
        On_mode: a function (or None) to evaluate on mode change
        """
        print("Connecting to", device, "at", baudrate, "set to id", source_system)
        self.master = mavutil.mavlink_connection(device, baudrate, source_system,
            autoreconnect=autoreconnect)
        self.channel = channel
        self.cutoffs = cutoffs
        self.exiting = False
        self.mode = None
        self.rate = rate
        self.on_mode = on_mode

    def run(self):
        # Wait for a heartbeat so we know the target system IDs
        print("Waiting for heartbeat")
        self.master.wait_heartbeat()

        # Set that we want to receive data
        print("Requesting data")
        self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, self.rate, 1)

        while not self.exiting:
            msg = self.master.recv_match(blocking=True)
            msg_type = msg.get_type()
            msg_data = msg.to_dict()

            if msg_type == "RC_CHANNELS":
                val = msg_data["chan"+str(self.channel)+"_raw"]

                if val < self.cutoffs[0]:
                    self.set_mode(0)
                elif val < self.cutoffs[1]:
                    self.set_mode(1)
                else:
                    self.set_mode(2)

    def set_mode(self, mode):
        if self.mode != mode:
            print("Changing to mode", mode)
            self.mode = mode

            if self.on_mode is not None:
                self.on_mode(mode)

    def exit(self):
        self.exiting = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baudrate", type=int, default=115200,
        help="Master port baud rate (default 115200)")
    parser.add_argument("--device", type=str, default="/dev/ttyAMA0",
        help="Serial device (default /dev/ttyAMA0)")
    parser.add_argument("--source-system", type=int, default=255,
        help="MAVLink source system for this GCS (default 255)")
    parser.add_argument("--aux-channel", type=int, default=6,
        help="What aux channel to use for enabling (default 6)")
    args = parser.parse_args()

    ap = AutopilotCommuncation(args.device, args.baudrate, args.source_system,
        args.aux_channel)

    try:
        ap.run()
    except KeyboardInterrupt:
        ap.exit()
