import argparse
import time
import math
import logging
import numpy as np
import cv2

from dronekit import connect, VehicleMode, LocationGlobalRelative
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

from opencv.lib_aruco_pose import ArucoSingleTracker


# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("PREC_LAND_SIM")


# ==========================================================
# CONNECT VEHICLE
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--connect", required=True)
args = parser.parse_args()

log.info("Connecting to vehicle...")
vehicle = connect(args.connect, wait_ready=False)
log.info("Connected")


# ==========================================================
# CAMERA CALIBRATION
# ==========================================================
camera_matrix = np.loadtxt("opencv/cameraMatrix.txt", delimiter=",")
camera_distortion = np.loadtxt("opencv/cameraDistortion.txt", delimiter=",")

aruco = ArucoSingleTracker(
    id_to_find=72,
    marker_size=0.1,   # IMPORTANT: meters (SIM marker is 0.1m)
    camera_matrix=camera_matrix,
    camera_distortion=camera_distortion
)


# ==========================================================
# HELPER FUNCTIONS (same as your original)
# ==========================================================
rad_2_deg = 180.0 / math.pi
deg_2_rad = 1.0 / rad_2_deg


def marker_position_to_angle(x, y, z):
    return math.atan2(x, z), math.atan2(y, z)


def camera_to_uav(x_cam, y_cam):
    return -y_cam, x_cam


def uav_to_ne(x_uav, y_uav, yaw_rad):
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return (x_uav * c - y_uav * s), (x_uav * s + y_uav * c)


# ==========================================================
# STATE VARIABLES
# ==========================================================
landing_active = False


# ==========================================================
# GAZEBO IMAGE SUBSCRIBER
# ==========================================================
node = Node()

topic = "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image"


def image_callback(msg):

    global landing_active

    width = msg.width
    height = msg.height

    img = np.frombuffer(msg.data, dtype=np.uint8)
    img = img.reshape((height, width, 3))

    # Gazebo gives RGB → convert to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    found, x, y, z = aruco.track(img)

    if found:
        log.info(f"ARUCO x={x:.3f} y={y:.3f} z={z:.3f}")

        # ===============================
        # SIMPLE TEST CONTROL (FOR NOW)
        # ===============================
        if vehicle.mode.name != "POSCTL":
            log.info("Switching to POSCTL")
            vehicle.mode = VehicleMode("POSCTL")
            landing_active = True

        if landing_active:

            x_uav, y_uav = camera_to_uav(x, y)

            angle_x, angle_y = marker_position_to_angle(x_uav, y_uav, z)

            log.info(f"Angles: {math.degrees(angle_x):.2f}, {math.degrees(angle_y):.2f}")

            # VERY SIMPLE proportional move
            Kp = 0.5

            north, east = uav_to_ne(
                Kp * x_uav,
                Kp * y_uav,
                vehicle.attitude.yaw
            )

            current = vehicle.location.global_relative_frame

            target = LocationGlobalRelative(
                current.lat + north * 1e-5,
                current.lon + east * 1e-5,
                current.alt
            )

            vehicle.simple_goto(target)

            # Descend when centered
            if abs(x) < 0.05 and abs(y) < 0.05:
                vehicle.simple_goto(
                    LocationGlobalRelative(
                        current.lat,
                        current.lon,
                        current.alt - 0.1
                    )
                )

            if z < 0.2:
                log.warning("LAND")
                vehicle.mode = VehicleMode("LAND")


node.subscribe(Image, topic, image_callback)

log.info("Subscribed to Gazebo camera. Waiting for frames...")

while True:
    time.sleep(1)
