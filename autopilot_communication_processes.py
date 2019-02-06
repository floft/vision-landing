#!/usr/bin/env python3
"""
Enable some functionality upon a switch on the R/C controller

(version where theses are processes rather than threads)

Examples: https://github.com/ArduPilot/pymavlink/blob/master/examples/
https://github.com/ThermalSoaring/thermal-soaring/blob/master/networking_mavlink.py
"""
import math
import time
import datetime
import argparse
import pymavlink
import multiprocessing

from collections import deque
from pymavlink import mavutil, mavwp, mavparm

def us_since_epoch():
    """ Microseconds since 1970-01-01 00:00:00 """
    ep = datetime.datetime(1970,1,1,0,0,0)
    diff = datetime.datetime.utcnow() - ep
    diff_us = diff.total_seconds()*1e6 + diff.microseconds
    return int(diff_us)

def rotate(angle_x, angle_y):
    """
    Actual calcuation:
        sin_yaw = math.sin(rotation/180*math.pi)
        #cos_yaw = math.cos(rotation/180*math.pi)
        cos_yaw = 0 # Python math doesn't actually make it zero though

        rot = np.array([[cos_yaw, -sin_yaw, 0],[sin_yaw,cos_yaw,0],[0,0,1]])
        vec = np.array([1,2,3]) # y, x, z
        rot.dot(vec) # returns: array([ 2., -1.,  3.]) # y, x, z = x, -y, z

        i.e. we get: x, y = -y, x

    Reference frame of ArduCopter -- x forward, y right
    https://discuss.ardupilot.org/t/copter-x-y-z-which-is-which/6823/3
    https://docs.px4.io/en/config/flight_controller_orientation.html

    Basically, try it till it works. My RPi is mounted so the top of the image
    is actually the left side of my drone and the right of the image is the front.

    Actually, I think that +y in the image is the bottom of the image. Thus,
    by mounting the RPi in that way, I think it's actually outputting (x,y)
    coordinates in the same frame as the drone.
    See: https://github.com/tensorflow/tensorflow/issues/9142

    Or.... from this Youtube comment
    https://www.youtube.com/watch?v=5AVb2hA2EUs&lc=UggMS88TyPsEt3gCoAEC.8I2Q1bOkhXh8I3ZMGRRKTY
    saying "negative X = target to the left, negative Y = target is forward", then
    we want:
    """
    # Other tries:
    #return angle_x, -angle_y # if x is forward, y is right
    #return -angle_y, angle_x # if y is forward, x is right
    #return angle_x, angle_y # if x is forward, y is right
    return angle_y, -angle_x # from Youtube comment and TF origin in top left

class Buffer:
    """
    Add items to a deque and provide a function to wait for new items, which
    will be run in a separate thread.

    You'll need this too:
        # Allow working with a deque between threads
        # http://stackoverflow.com/a/27345949
        from multiprocessing.managers import SyncManager
        SyncManager.register('deque', deque)

    Based on:
    https://github.com/ThermalSoaring/thermal-soaring/blob/master/soaring.py
    """
    def __init__(self, sync_manager=None, max_length=25):
        self.cond = multiprocessing.Condition()

        if sync_manager is not None:
            self.buffer = sync_manager.deque(maxlen=max_length)
        else:
            self.buffer = deque(maxlen=max_length)

        self.exiting = False

    def add_data(self, data):
        """ Add new data """
        with self.cond:
            if not self.exiting:
                self.buffer.append(data)
                self.cond.notify()

    def get_data(self):
        """ Get data now, if not available return None """
        with self.cond:
            if not self.exiting:
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

            return None

    def exit(self):
        """ Make sure anything waiting on this dies """
        with self.cond:
            self.exiting = True
            self.cond.notify()

class AutopilotConnection:
    def __init__(self, device="/dev/ttyAMA0", baudrate=115200,
            source_system=255, autoreconnect=True):
        print("Connecting to", device, "at", baudrate, "set to id", source_system)
        self.master = mavutil.mavlink_connection(device, baudrate, source_system,
            autoreconnect=autoreconnect)

    def connect(self):
        # Wait for a heartbeat so we know the target system IDs
        print("Waiting for heartbeat")
        self.master.wait_heartbeat()

class AutopilotCommuncationReceive(multiprocessing.Process):
    def __init__(self, connection, channels=[6], cutoffs=[1282, 1716], rate=2,
            on_mode=None):
        """
        Device: e.g. /dev/ttyAMA0
        Baudrate: e.g. 115200
        Source_system: probably 255
        Channels: e.g. [6] for aux 6 (set via your RC controller)
        Autoreconnect: passed to MAVLink connection
        Cutoffs: my switch 6 goes between 1065, 1499, and 1933, and the defaults
            are between those
        Rate: the Hz you want to be parsing these messages
        On_mode: a function (or None) to evaluate on mode change
        """
        # Process able to exit
        # https://stackoverflow.com/a/1231648/2698494
        super().__init__()
        self.exiting = multiprocessing.Event()
        self.master = connection.master
        self.channels = channels
        self.cutoffs = cutoffs
        self.modes = {c: None for c in self.channels}
        self.rate = rate
        self.on_mode = on_mode

    def run(self):
        # Set that we want to receive data
        print("Requesting data")
        self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, self.rate, 1)
        # For everything: MAV_DATA_STREAM_ALL
        # See https://github.com/PX4/Firmware/blob/master/Tools/mavlink_px4.py#L338

        while not self.exiting.is_set():
            msg = self.master.recv_match(blocking=True)
            msg_type = msg.get_type()
            msg_data = msg.to_dict()

            if msg_type == "RC_CHANNELS_RAW":
                for c in self.channels:
                    val = msg_data["chan"+str(c)+"_raw"]

                    if val < self.cutoffs[0]:
                        self.set_mode(c, 0)
                    elif val < self.cutoffs[1]:
                        self.set_mode(c, 1)
                    else:
                        self.set_mode(c, 2)

    def set_mode(self, channel, mode):
        if self.modes[channel] != mode:
            print("Changing channel", channel, "to mode", mode)
            self.modes[channel] = mode

            if self.on_mode is not None:
                self.on_mode(channel, mode)

    def exit(self):
        self.exiting.set()

class AutopilotCommuncationSend(multiprocessing.Process):
    def __init__(self, connection, buffer_manager):
        # Process able to exit
        # https://stackoverflow.com/a/1231648/2698494
        super().__init__()
        self.exiting = multiprocessing.Event()
        self.buffer_manager = buffer_manager
        self.master = connection.master

    def run(self):
        while not self.exiting.is_set():
            detection = self.buffer_manager.get_wait()

            if detection:
                self.send_detection(detection)

    def send_detection(self, detection,
            horizontal_resolution=300, vertical_resolution=300,
            horizontal_fov=62.2, vertical_fov=48.8,
            target_size=0.26, # diameter in meters
            debug=True):
        x = (detection["xmin"] + detection["xmax"]) / 2
        y = (detection["ymin"] + detection["ymax"]) / 2

        # Convert fov to radians
        # Given defaults above taken from:
        # https://elinux.org/Rpi_Camera_Module#Technical_Parameters_.28v.2_board.29
        # Alternative: https://stackoverflow.com/a/41137160/2698494
        #    from camera_calibration_compute import load_camera_calibration
        #    m = load_camera_calibration("camera_calibration.npy")
        #    fx = 2*math.atan(640/2/a[0][0,0]) * 180/math.pi
        #    fy = 2*math.atan(480/2/a[0][1,1]) * 180/math.pi
        #    Gives: 83.80538074799328 and 68.51413217813369, which are wrong?
        # Says it's the partial FOV of the camera at 640x480 for v2, so it's not
        # quite the default values?
        #    https://picamera.readthedocs.io/en/release-1.13/fov.html
        horizontal_fov = horizontal_fov / 180 * math.pi
        vertical_fov = vertical_fov / 180 * math.pi

        angle_x = (x - horizontal_resolution / 2) / horizontal_resolution * horizontal_fov
        angle_y = (y - vertical_resolution / 2) / vertical_resolution * vertical_fov

        width = detection["xmax"] - detection["xmin"]
        height = detection["ymax"] - detection["ymin"]
        size_x = width / horizontal_resolution * horizontal_fov
        size_y = height / vertical_resolution * vertical_fov

        # Above are all for camera orientation, but it's actually rotated on
        # the frame
        angle_x, angle_y = rotate(angle_x, angle_y)

        # Calculate height of drone above the target based on whichever measurement
        # is larger. This is because if it's on the side of an image, it'll only
        # detect part of the frying pan, but one side of it (the larger detected
        # side) is probably all the way in the image, so use that for the size.
        # This is easy since the frying pan is circular (not rectangular).
        larger_size = size_x if width > height else size_y
        height = (target_size/2) / math.tan(larger_size/2)

        # See:
        # https://github.com/ArduPilot/ardupilot/blob/master/Tools/autotest/arducopter.py
        # https://github.com/squilter/target-land/blob/master/target_land.py
        self.master.mav.landing_target_send(
            0,       # time_boot_ms (not used)
            0,       # target num (not used)
            0,       # frame (not used)
            angle_x, # angle_x
            angle_y, # angle_y
            height, # height above target (m)
            0, # size_x (rad) -- size of target (not used)
            0) # size_y (rad) -- size of target (not used)

        if debug:
            print("sent x", angle_x, "y", angle_y, "z", height*3.28084, "ft")

    def exit(self):
        self.exiting.set()
