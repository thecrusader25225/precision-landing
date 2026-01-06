import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import Gst
import numpy as np
import cv2
import time
import math
import logging

from dronekit import connect, VehicleMode, LocationGlobalRelative
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
DEBUG_LOG_PERIOD = 0.4  # seconds


def debug_log(msg):
    global last_debug_log
    now = time.time()
    if now - last_debug_log > DEBUG_LOG_PERIOD:
        log.info(msg)
        last_debug_log = now


# ==========================================================
# GSTREAMER PIPELINE
# ==========================================================
Gst.init(None)

pipeline = Gst.parse_launch(
    "libcamerasrc "
    "! video/x-raw,format=NV12,width=640,height=480,framerate=30/1 "
    "! tee name=t "
    "t. ! queue ! videoconvert ! video/x-raw,format=I420 "
    "! x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast "
    "! h264parse ! flvmux streamable=true "
    "! rtmpsink location=rtmp://100.78.97.114:1935/stream "
    "t. ! queue ! videoconvert ! video/x-raw,format=BGR "
    "! appsink name=appsink emit-signals=true max-buffers=1 drop=true"
)

appsink = pipeline.get_by_name("appsink")
pipeline.set_state(Gst.State.PLAYING)


# ==========================================================
# CONNECT VEHICLE
# ==========================================================
log.info("Connecting to vehicle...")
vehicle = connect("/dev/ttyACM0", baud=115200, wait_ready=True)
log.info("Connected")


# ==========================================================
# CAMERA CALIBRATION
# ==========================================================
camera_matrix = np.loadtxt("opencv/cameraMatrix.txt", delimiter=",")
camera_distortion = np.loadtxt("opencv/cameraDistortion.txt", delimiter=",")

aruco = ArucoSingleTracker(
    id_to_find=72,
    marker_size=11.3,  # cm
    camera_matrix=camera_matrix,
    camera_distortion=camera_distortion
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
rad_2_deg = 180.0 / math.pi
deg_2_rad = 1.0 / rad_2_deg


def get_location_metres(original_location, dNorth, dEast):
    earth_radius = 6378137.0
    dLat = dNorth / earth_radius
    dLon = dEast / (earth_radius * math.cos(math.pi * original_location.lat / 180))
    newlat = original_location.lat + (dLat * 180 / math.pi)
    newlon = original_location.lon + (dLon * 180 / math.pi)
    return newlat, newlon


def marker_position_to_angle(x, y, z):
    return math.atan2(x, z), math.atan2(y, z)


def camera_to_uav(x_cam, y_cam):
    return -y_cam, x_cam


def uav_to_ne(x_uav, y_uav, yaw_rad):
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return (x_uav * c - y_uav * s), (x_uav * s + y_uav * c)


def check_angle_descend(ax, ay, limit):
    return math.sqrt(ax**2 + ay**2) <= limit


# ==========================================================
# CONTROL PARAMETERS
# ==========================================================
STABLE_FRAMES = 15
LOST_FRAMES_LIMIT = 20

LANDING_Z_THRESHOLD = 30.0  # cm
angle_descend = 20 * deg_2_rad
land_speed_cms = 30.0

SEARCH_RADIUS = 1.5
SEARCH_ANG_VEL = 0.3
SEARCH_TIMEOUT = 30.0

# -------- RTL HANDOFF PARAMETERS --------
RTL_FINAL_ALT_M = 1.0   # must match RTL_ALT_FINAL
RTL_ALT_TOL = 0.3       # meters


# ==========================================================
# STATE VARIABLES
# ==========================================================
marker_seen_count = 0
marker_lost_count = 0

landing_active = False
search_active = False

search_start_time = None
search_theta = 0.0

rtl_complete = False


# ==========================================================
# MAIN LOOP
# ==========================================================
log.info("System ready")

while True:

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
        debug_log(f"ARUCO → x={x:.2f} y={y:.2f} z={z:.2f}")

    # ---------------- DETECTION STATE ----------------
    if found:
        marker_seen_count += 1
        marker_lost_count = 0
    else:
        marker_lost_count += 1

    # ---------------- RTL COMPLETION DETECTION ----------------
    if (
        vehicle.mode.name == "RTL"
        and not rtl_complete
        and abs(vehicle.location.global_relative_frame.alt - RTL_FINAL_ALT_M) <= RTL_ALT_TOL
    ):
        rtl_complete = True
        log.info("RTL complete at final altitude")

    # ---------------- RTL HANDOFF ----------------
    if rtl_complete and not landing_active and not search_active:
        log.info("Taking control after RTL")
        vehicle.mode = VehicleMode("GUIDED")

        if found:
            log.info("Tag visible after RTL → precision landing")
            landing_active = True
        else:
            log.info("Tag NOT visible after RTL → starting search")
            search_active = True
            search_start_time = time.time()
            search_theta = 0.0

    # ---------------- ACTIVATE LANDING (NON-RTL PATH) ----------------
    if marker_seen_count >= STABLE_FRAMES and not landing_active and not rtl_complete:
        log.info("Marker stable → GUIDED")
        vehicle.mode = VehicleMode("GUIDED")
        landing_active = True

    # ---------------- START SEARCH (LOSS DURING LANDING) ----------------
    if (
        (landing_active or rtl_complete)
        and marker_lost_count > LOST_FRAMES_LIMIT
        and not search_active
    ):
        log.warning("Marker lost → starting local search")
        search_active = True
        search_start_time = time.time()
        search_theta = 0.0

    # ---------------- STOP SEARCH ----------------
    if found and search_active:
        log.info("Marker reacquired → resume landing")
        search_active = False
        landing_active = True

    # ---------------- LANDING CONTROL ----------------
    if landing_active and found:

        x_uav, y_uav = camera_to_uav(x, y)
        uav_location = vehicle.location.global_relative_frame

        if uav_location.alt >= 5.0:
            z = uav_location.alt * 100.0

        angle_x, angle_y = marker_position_to_angle(x_uav, y_uav, z)
        north, east = uav_to_ne(x_uav, y_uav, vehicle.attitude.yaw)

        tgt_lat, tgt_lon = get_location_metres(
            uav_location,
            north * 0.01,
            east * 0.01
        )

        if check_angle_descend(angle_x, angle_y, angle_descend):
            tgt_alt = uav_location.alt - (land_speed_cms * 0.01)
        else:
            tgt_alt = uav_location.alt

        vehicle.simple_goto(LocationGlobalRelative(tgt_lat, tgt_lon, tgt_alt))

        debug_log(
            f"MOVING TO | x={x:.2f} y={y:.2f} z={z:.2f} "
            f"alt={uav_location.alt:.2f}"
        )

        if z <= LANDING_Z_THRESHOLD:
            log.warning("Landing threshold reached → LAND")
            vehicle.mode = VehicleMode("LAND")

    # ---------------- LOCAL CIRCULAR SEARCH ----------------
    if search_active and not found:

        debug_log(
            f"SEARCH | radius={SEARCH_RADIUS:.1f} "
            f"theta={math.degrees(search_theta):.1f}°"
        )

        uav_location = vehicle.location.global_relative_frame
        dN = SEARCH_RADIUS * math.cos(search_theta)
        dE = SEARCH_RADIUS * math.sin(search_theta)

        tgt_lat, tgt_lon = get_location_metres(uav_location, dN, dE)
        vehicle.simple_goto(LocationGlobalRelative(tgt_lat, tgt_lon, uav_location.alt))

        search_theta += SEARCH_ANG_VEL * 0.03

        if time.time() - search_start_time > SEARCH_TIMEOUT:
            log.error("Search timeout → LOITER")
            vehicle.mode = VehicleMode("LOITER")
            search_active = False
            landing_active = False

    time.sleep(0.03)
