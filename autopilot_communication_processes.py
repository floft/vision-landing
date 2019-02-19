#!/usr/bin/env python3
"""
Enable some functionality upon a switch on the R/C controller

(version where theses are processes rather than threads)

Examples: https://github.com/ArduPilot/pymavlink/blob/master/examples/
https://github.com/ThermalSoaring/thermal-soaring/blob/master/networking_mavlink.py
"""
import os
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
    we want: angle_y, -angle_x
    """
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
    """
    Connection with MAVLink to the autopilot

    By default use serial, but device can also be set to something like:
        tcp:127.0.0.1:5760
    if using mavlink-routerd for instance.

    By default also uses MAVLink 2 since some data we're interested in like
    if the target is acquired is only provided in the v2 messages.

    source_system=1, source_component=2, because: "Other MAVLink capable
    device on the vehicle (i.e. companion computer, gimbal) should use
    the same System ID as the flight controller but use a different
    Component ID" -- http://ardupilot.org/dev/docs/mavlink-basics.html
    """
    def __init__(self, device="/dev/ttyAMA0", baudrate=115200,
            source_system=1, source_component=2, autoreconnect=True,
            mavlink2=True):
        # See: https://mavlink.io/en/mavgen_python/#dialect_file
        if mavlink2:
            os.environ["MAVLINK20"] = "1"
            dialect = "common" # must be set to care about above line
        else:
            dialect = None

        print("Connecting to", device, "at", baudrate, "set to ",
            "system", source_system, "component", source_component)
        self.master = mavutil.mavlink_connection(device, baudrate, source_system,
            source_component, autoreconnect=autoreconnect, dialect=dialect)
        self.connected = False

    def connect(self):
        # Wait for a heartbeat so we know the target system IDs
        print("Waiting for heartbeat")
        self.master.wait_heartbeat()
        self.connected = True

        print("Connecting to target", self.master.target_system,
            "component", self.master.target_component)

    def get_home(self, wait_condition=True):
        """
        Get the home position -- wait till armed though since set on arm

        We'll wait while wait_condition, which by default is forever, but you
        could alternatively set to something like (lambda: not exiting) for some
        exiting variable.
        """
        if not self.connected:
            return None

        # Home point is created when the motors are armed, so it won't be
        # correct before then
        self.master.motors_armed_wait()

        # Request the home point
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_GET_HOME_POSITION,
            0, 0, 0, 0, 0, 0, 0, 0)

        # Wait to receive it
        while wait_condition:
            msg = self.master.recv_match(type=["HOME_POSITION"], blocking=True)
            msg_type = msg.get_type()
            msg_data = msg.to_dict()

            if msg_type == "HOME_POSITION":
                print("Got home position:", msg_data)
                return msg_data

        # Didn't get it, i.e. wait_condition evaluated to False at some point
        return None

    def get_yaw(self, wait_condition=True):
        """
        Get the next yaw from an attitude packet we can

        Warning: make sure you have run request_all() before doing this.
        Otherwise we'll never receive the attitude message.
        """
        if not self.connected:
            return None

        # Wait to receive it
        while wait_condition:
            msg = self.master.recv_match(type=["ATTITUDE"], blocking=True)
            msg_type = msg.get_type()
            msg_data = msg.to_dict()

            if msg_type == "ATTITUDE":
                print("Got attitude:", msg_data)
                return msg_data["yaw"] # -pi..pi

        # Didn't get it, i.e. wait_condition evaluated to False at some point
        return None

    def send_waypoints(self, waypoints, radius=0.5, land_at_end=True,
            origin=None, takeoff_at_beginning=True):
        """
        Send list of waypoints as a new flight plan

        Radius is in meters -- how close to get to waypoint before heading to
        the next one, but only applies if hold time is > 1 second

        If origin==None: Format of waypoints is [(hold1, lat1, lon1, alt1), ...]

        If origin == (orig_lat, orig_lon, orig_alt), then the format is:
            [(hold1, north1, east1, alt1), ...] where the NEU (m) is relative to
            the origin lat/lon/alt. Since apparently ArduPilot does not support
            local reference frames, we do an approximate conversion to global
            coordinates but calculated relative to the specified origin. Note
            that the altitude is already relative to the home position though,
            so you might just set orig_alt=0 and specify the altitudes relative
            to the home position.

        If land_at_end==True, then after the last waypoint, a final landing
        waypoint is set at the same location as the last waypoint.

        From: https://www.colorado.edu/recuv/2015/05/25/mavlink-protocol-waypoints
        See: https://mavlink.io/en/services/mission.html
        """
        wp = mavwp.MAVWPLoader()
        frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
        autocontinue = 1

        # TODO maybe it skips takeoff command if this is not 0? try seq=0
        # but send the HOME position as the seq=0? See:
        # https://discuss.ardupilot.org/t/planning-missions-via-mavlink/16489/8
        seq = 1

        for i, (hold, lat, lon, alt) in enumerate(waypoints):
            # If relative to some origin, convert to global coordinates
            if origin is not None:
                orig_lat, orig_lon, orig_alt = origin
                lat, lon = self.get_location_meters(orig_lat, orig_lon, lat, lon)
                alt = orig_alt+alt

            # Optionally (probably) take off at beginning
            # Maybe see:
            # https://gist.github.com/donghee/8d8377ba51aa11721dcaa7c811644169
            if takeoff_at_beginning and i == 0:
                wp.add(mavutil.mavlink.MAVLink_mission_item_message(
                    self.master.target_system, self.master.target_component,
                    seq, frame, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, autocontinue,
                    0, 0, 0, math.nan,
                    lat, lon, alt))
                print("takeoff at", lat, lon, alt)
                seq += 1

            # Add waypoint
            wp.add(mavutil.mavlink.MAVLink_mission_item_message(
                self.master.target_system, self.master.target_component,
                seq, frame, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, autocontinue,
                hold, radius, 0, math.nan,
                lat, lon, alt))
            print("waypoint at", lat, lon, alt)
            seq += 1

            # Optionally land at the last waypoint
            if land_at_end and i == len(waypoints)-1:
                # 1 == opportunistic precision land, i.e. use it if it's tracking
                wp.add(mavutil.mavlink.MAVLink_mission_item_message(
                    self.master.target_system, self.master.target_component,
                    seq, frame, mavutil.mavlink.MAV_CMD_NAV_LAND, 0, autocontinue,
                    0, 1, 0, math.nan,
                    lat, lon, alt))
                print("land at", lat, lon, alt)
                seq += 1

        self.master.waypoint_clear_all_send()
        self.master.waypoint_count_send(wp.count())

        for _ in range(wp.count()):
            msg = self.master.recv_match(type=["MISSION_REQUEST"], blocking=True)
            self.master.mav.send(wp.wp(msg.seq))
            print("Sending waypoint {0}".format(msg.seq))

        return wp

    def get_location_meters(self, orig_lat, orig_lon, north_dist, east_dist):
        """
        Compute (lat, lon) north_dist meters North and east_dist meters East
        of the (orig_lat, orig_lon) starting point. Assumes (lat,lon) was given
        in degrees and returns result in degrees.

        The function is useful when you want to move the vehicle around
        specifying locations relative to the current vehicle position.

        The algorithm is relatively accurate over small distances
        (10m within 1km) except close to the poles.

        For more information see:
        http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters

        Based on get_location_metres given on:
        http://python.dronekit.io/examples/guided-set-speed-yaw-demo.html#guided-example-source-code
        https://github.com/dronekit/dronekit-python/blob/master/examples/guided_set_speed_yaw/guided_set_speed_yaw.py#L165
        """
        # Radius of "spherical" earth
        earth_radius = 6378137.0

        # Coordinate offsets in radians
        dLat = north_dist / earth_radius
        dLon = east_dist / (earth_radius * math.cos(orig_lat * math.pi/180))

        # New position in decimal degrees
        new_lat = orig_lat + (dLat * 180/math.pi)
        new_lon = orig_lon + (dLon * 180/math.pi)

        return new_lat, new_lon

    def get_yaw_from_quaternion(self, q, degrees=False):
        """
        Extract the yaw angle from a quaternion

        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Source_Code_2
        """
        w, x, y, z = q
        siny_cosp = +2.0 * (w * z + x * y)
        cosy_cosp = +1.0 - 2.0*(y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        if degrees:
            return yaw * 180/math.pi

        return yaw

    #
    # Request different data
    #
    def request_all(self, rate):
        """
        Request all data
        https://github.com/PX4/Firmware/blob/master/Tools/mavlink_px4.py#L338 """
        if self.connected:
            self.master.mav.request_data_stream_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL, rate, 1)

    def request_rc(self, rate):
        """
        Request just RC channels
        https://github.com/PX4/Firmware/blob/master/Tools/mavlink_px4.py#L338
        """
        if self.connected:
            self.master.mav.request_data_stream_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, rate, 1)

    #
    # Easy access to some commonly-used modes
    #
    # List, see mode_mapping_acm
    # https://github.com/ArduPilot/pymavlink/blob/master/mavutil.py#L1840
    def set_mode_poshold(self):
        if self.connected:
            self.master.set_mode("POSHOLD")

    def set_mode_auto(self):
        if self.connected:
            self.master.set_mode_auto()

    def set_mode_land(self):
        if self.connected:
            self.master.set_mode("LAND")

    def set_mode_rtl(self):
        if self.connected:
            self.master.set_mode_rtl()

class AutopilotCommuncationReceive(multiprocessing.Process):
    """
    The process that receives information from the autopilot

    This will manage enabling and disabling. If a certain channel is set high,
    then we'll start streaming video to the laptop. If it's set higher, then we'll
    initiate shutdown (though the exact functionality is in callbacks implemented
    in control.py). This also prints out debug information about if we have
    acquired the landing target.
    """
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
        self.connection = connection
        self.channels = channels
        self.cutoffs = cutoffs
        self.modes = {c: None for c in self.channels}
        self.rate = rate
        self.on_mode = on_mode

    def run(self, debug=True):
        # Set that we want to receive all data
        self.connection.request_all(self.rate)

        while not self.exiting.is_set():
            msg = self.connection.master.recv_match(
                type=["RC_CHANNELS", "RC_CHANNELS_RAW", "LANDING_TARGET"],
                blocking=True)
            msg_type = msg.get_type()
            msg_data = msg.to_dict()

            # If SERIALx_PROTOCOL = Mavlink 1 -- RC_CHANNELS_RAW
            # if Mavlink 2 -- RC_CHANNELS
            if msg_type == "RC_CHANNELS_RAW" or msg_type == "RC_CHANNELS":
                for c in self.channels:
                    val = msg_data["chan"+str(c)+"_raw"]

                    if val < self.cutoffs[0]:
                        self.set_mode(c, 0)
                    elif val < self.cutoffs[1]:
                        self.set_mode(c, 1)
                    else:
                        self.set_mode(c, 2)
            elif msg_type == "LANDING_TARGET":
                if "position_valid" in msg_data:
                    target_acquired = msg_data["position_valid"]
                    if debug:
                        print("Target acquired:", target_acquired)
                else:
                    if debug:
                        print("Landing target:", msg_data)

    def set_mode(self, channel, mode):
        if self.modes[channel] != mode:
            print("Changing channel", channel, "to mode", mode)
            self.modes[channel] = mode

            if self.on_mode is not None:
                self.on_mode(channel, mode)

    def exit(self):
        self.exiting.set()

class AutopilotCommuncationSend(multiprocessing.Process):
    """
    The process that sends information to the autopilot

    This will both get the home point, calculate the flight path, and send that
    to the autopilot. It will also send landing target messages whenever we
    detect the target in an image (see control.py that receives these bounding
    boxes and puts them into the buffer manager).
    """
    def __init__(self, connection, buffer_manager):
        # Process able to exit
        # https://stackoverflow.com/a/1231648/2698494
        super().__init__()
        self.exiting = multiprocessing.Event()
        self.buffer_manager = buffer_manager
        self.connection = connection

    def run(self, enable_auto=True, min_num_detections=10):
        # If we will fly autonomously, then we first need to know where it was
        # armed and from that we'll create the flight plan. Request all data
        # at a low frequency.
        if enable_auto:
            print("Waiting for home position (and for motors to be armed)")
            home = self.connection.get_home(lambda: not self.exiting.is_set())

            # HOME_POSITION always sets the quaternion to (1,0,0,0) regardless
            # of orientation, so get it from an ATTITUDE packet now since we're
            # armed. Note: don't need to request_all() in this process since
            # we did it in the receive process and these are both over the same
            # link via mavlink-router.
            yaw = self.connection.get_yaw()

            if home and yaw:
                # See: https://mavlink.io/en/messages/common.html#HOME_POSITION
                home_lat = home["latitude"]*1e-7 # deg
                home_lon = home["longitude"]*1e-7 # deg
                #home_alt = home["altitude"]*1e-3 # m
                # q is atitude quaternion: w, x, y, z order -- zero-rotation is {1, 0, 0, 0}
                #yaw = self.connection.get_yaw_from_quaternion(home["q"]) # from North

                # Calculate distances North and East to be a certain distance
                # forward from the home position/heading
                #
                # Note: I'm in the northern hemisphere
                forward_6m = (6*math.cos(yaw), 6*math.sin(yaw))
                forward_11m = (11.5*math.cos(yaw), 11.5*math.sin(yaw))

                # Calculate new waypoints for flight plan
                # Note: set WPNAV_SPEED=180 or so to slow down navigation
                """ Flight plan for actual egg drop
                up = (0, 0, 0, 3) # up 3 meters above home position
                forward = (0, forward_6m[0], forward_6m[1], 3) # forward 6 meters
                down = (0, forward_6m[0], forward_6m[1], -3) # down 6 meters
                forward2 = (1, forward_11m[0], forward_11m[1], -3) # foward 5.5 meters, hold 1s
                """

                # Debugging flight plan
                up = (0, 0, 0, 6) # up 6 meters above home position
                forward = (0, forward_6m[0], forward_6m[1], 6) # forward 6 meters
                down = (0, forward_6m[0], forward_6m[1], 4.5) # down 1.5 meters
                forward2 = (1, forward_11m[0], forward_11m[1], 4.5) # foward 5.5 meters, hold 1s

                self.connection.send_waypoints(
                    [up, forward, down, forward2],
                    origin=(home_lat, home_lon, 0))

                print("Done creating flight plan")
            else:
                if not home:
                    print("Warning: could not get home position")
                if not yaw:
                    print("Warning: could not get yaw")

        # If we are flying autonomously, don't force landing until we've seen
        # the target a number of times. Otherwise, maybe it's a false positive.
        num_detections = 0

        # Also, only set to land once. Otherwise, it's going to be hard to
        # get it out of this mode if we want to manually take control.
        already_set = False

        while not self.exiting.is_set():
            # Wait for control.py to receive bounding boxes and add them to the
            # deque passed to this process
            detection = self.buffer_manager.get_wait()

            # Send detection to the autopilot
            if detection:
                self.send_detection(detection)
                num_detections += 1

            # If flying autonomously and we've detected the target for long
            # enough, then land immediately
            if enable_auto and num_detections > min_num_detections and not already_set:
                print("Setting to LAND mode")
                self.connection.set_mode_land()
                already_set = True

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
        self.connection.master.mav.landing_target_send(
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
