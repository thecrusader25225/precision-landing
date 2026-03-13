import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

import argparse
import time
import math
import logging

from gi.repository import Gst
import numpy as np
import cv2

from pymavlink import mavutil
from opencv.lib_aruco_pose import ArucoSingleTracker

# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("PREC_LAND")

last_debug_log = 0
DEBUG_LOG_PERIOD = 0.4


def debug_log(msg):
    global last_debug_log
    now = time.time()
    if now - last_debug_log > DEBUG_LOG_PERIOD:
        log.info(msg)
        last_debug_log = now


# ==========================================================
# MAVLINK VEHICLE ABSTRACTION
# ==========================================================
class MAVVehicle:
    def __init__(self, conn_str):
        self.master = mavutil.mavlink_connection(conn_str)
        self.master.wait_heartbeat()
        log.info("Heartbeat received")

        self.mode = None
        self.lat = None
        self.lon = None
        self.alt = None
        self.yaw = None

    def update(self):
        while True:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break

            t = msg.get_type()

            if t == "GLOBAL_POSITION_INT":
                self.lat = msg.lat / 1e7
                self.lon = msg.lon / 1e7
                self.alt = msg.relative_alt / 1000.0

            elif t == "ATTITUDE":
                self.yaw = msg.yaw

            elif t == "HEARTBEAT":
                self.mode = mavutil.mode_string_v10(msg)

    def set_mode(self, mode):
        mapping = self.master.mode_mapping()
        if mode not in mapping:
            log.error(f"Mode {mode} not supported")
            return
        self.master.set_mode(mapping[mode])
        log.info(f"Mode → {mode}")

    def simple_goto(self, lat, lon, alt):
        self.master.mav.set_position_target_global_int_send(
            int(time.time() * 1e6),
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b110111111000,
            int(lat * 1e7),
            int(lon * 1e7),
            alt,
            0, 0, 0,
            0, 0, 0,
            0, 0
        )

    def send_landing_target(self, angle_x, angle_y, dist_m):
        self.master.mav.landing_target_send(
            int(time.time() * 1e6),
            0,
            mavutil.mavlink.MAV_FRAME_BODY_FRD,
            angle_x,
            angle_y,
            dist_m,
            0, 0
        )


# ==========================================================
# GSTREAMER PIPELINE
# ==========================================================
Gst.init(None)

pipeline = Gst.parse_launch(
    "videotestsrc is-live=true "
    "! video/x-raw,width=640,height=480,framerate=30/1 "
    "! videoconvert "
    "! video/x-raw,format=BGR "
    "! appsink name=appsink emit-signals=true max-buffers=1 drop=true"
)

appsink = pipeline.get_by_name("appsink")
pipeline.set_state(Gst.State.PLAYING)

# ==========================================================
# CONNECT VEHICLE
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--connect", required=True)
args = parser.parse_args()

log.info("Connecting to vehicle…")
vehicle = MAVVehicle(args.connect)
log.info("Connected")

# ==========================================================
# CAMERA CALIBRATION
# ==========================================================
camera_matrix = np.loadtxt("opencv/cameraMatrix.txt", delimiter=",")
camera_distortion = np.loadtxt("opencv/cameraDistortion.txt", delimiter=",")

aruco = ArucoSingleTracker(
    id_to_find=72,
    marker_size=11.3,
    camera_matrix=camera_matrix,
    camera_distortion=camera_distortion
)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
rad_2_deg = 180.0 / math.pi
deg_2_rad = 1.0 / rad_2_deg


def get_location_metres(lat, lon, dNorth, dEast):
    R = 6378137.0
    dLat = dNorth / R
    dLon = dEast / (R * math.cos(math.pi * lat / 180))
    return (
        lat + dLat * 180 / math.pi,
        lon + dLon * 180 / math.pi
    )


def marker_position_to_angle(x, y, z):
    return math.atan2(x, z), math.atan2(y, z)


def camera_to_uav(x_cam, y_cam):
    return -y_cam, x_cam


def uav_to_ne(x_uav, y_uav, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return x_uav * c - y_uav * s, x_uav * s + y_uav * c


def check_angle_descend(ax, ay, limit):
    return math.sqrt(ax ** 2 + ay ** 2) <= limit


# ==========================================================
# PARAMETERS
# ==========================================================
STABLE_FRAMES = 15
LOST_FRAMES_LIMIT = 20

LANDING_Z_THRESHOLD = 30.0
angle_descend = 20 * deg_2_rad
land_speed_cms = 30.0

SEARCH_RADIUS = 1.5
SEARCH_ANG_VEL = 0.3
SEARCH_TIMEOUT = 30.0

RTL_FINAL_ALT_M = 2.0
RTL_ALT_TOL = 0.3

# ==========================================================
# STATE
# ==========================================================
marker_seen_count = 0
marker_lost_count = 0

landing_active = False
search_active = False
search_start_time = None
search_theta = 0.0

rtl_complete = True

# ==========================================================
# MAIN LOOP
# ==========================================================
log.info("System ready")

while True:
    vehicle.update()

    sample = appsink.emit("pull-sample")
    if sample is None:
        continue

    buf = sample.get_buffer()
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        continue

    frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((480, 640, 3))
    buf.unmap(mapinfo)

    found, x, y, z = aruco.track(frame, verbose=False)

    if found:
        marker_seen_count += 1
        marker_lost_count = 0
        debug_log(f"ARUCO → x={x:.2f} y={y:.2f} z={z:.2f}")
    else:
        marker_lost_count += 1

    # RTL completion
    if (
        vehicle.mode == "RTL"
        and not rtl_complete
        and vehicle.alt is not None
        and abs(vehicle.alt - RTL_FINAL_ALT_M) <= RTL_ALT_TOL
    ):
        rtl_complete = True
        log.info("RTL complete")

    # RTL handoff
    if rtl_complete and not landing_active and not search_active:
        vehicle.set_mode("GUIDED")
        if found:
            landing_active = True
        else:
            search_active = True
            search_start_time = time.time()
            search_theta = 0.0

    # Start landing
    if marker_seen_count >= STABLE_FRAMES and not landing_active and not rtl_complete:
        vehicle.set_mode("GUIDED")
        landing_active = True

    # Start search
    if (landing_active or rtl_complete) and marker_lost_count > LOST_FRAMES_LIMIT and not search_active:
        search_active = True
        search_start_time = time.time()
        search_theta = 0.0

    # Stop search
    if found and search_active:
        search_active = False
        landing_active = True

    # LANDING CONTROL
    if landing_active and found and vehicle.alt is not None:
        x_uav, y_uav = camera_to_uav(x, y)

        if vehicle.alt >= 5.0:
            z = vehicle.alt * 100.0

        angle_x, angle_y = marker_position_to_angle(x_uav, y_uav, z)
        north, east = uav_to_ne(x_uav, y_uav, vehicle.yaw)

        tgt_lat, tgt_lon = get_location_metres(
            vehicle.lat,
            vehicle.lon,
            north * 0.01,
            east * 0.01
        )

        tgt_alt = vehicle.alt - (land_speed_cms * 0.01) \
            if check_angle_descend(angle_x, angle_y, angle_descend) \
            else vehicle.alt

        vehicle.simple_goto(tgt_lat, tgt_lon, tgt_alt)
        vehicle.send_landing_target(angle_x, angle_y, z / 100.0)

        if z <= LANDING_Z_THRESHOLD:
            log.warning("Landing threshold → LAND")
            vehicle.set_mode("LAND")

    # SEARCH
    if search_active and not found:
        dN = SEARCH_RADIUS * math.cos(search_theta)
        dE = SEARCH_RADIUS * math.sin(search_theta)

        tgt_lat, tgt_lon = get_location_metres(
            vehicle.lat,
            vehicle.lon,
            dN,
            dE
        )

        vehicle.simple_goto(tgt_lat, tgt_lon, vehicle.alt)
        search_theta += SEARCH_ANG_VEL * 0.03

        if time.time() - search_start_time > SEARCH_TIMEOUT:
            vehicle.set_mode("LOITER")
            search_active = False
            landing_active = False

    time.sleep(0.03)
