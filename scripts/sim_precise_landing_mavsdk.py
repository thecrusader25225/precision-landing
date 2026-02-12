import asyncio
import math
import numpy as np
import cv2
import logging
import time

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

from opencv.lib_aruco_pose import ArucoSingleTracker


# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PREC_LAND_MAVSDK")


# ==========================================================
# GLOBAL FRAME STORAGE
# ==========================================================
latest_frame = None


# ==========================================================
# GAZEBO IMAGE CALLBACK
# ==========================================================
def image_callback(msg):
    global latest_frame

    width = msg.width
    height = msg.height

    img = np.frombuffer(msg.data, dtype=np.uint8)
    img = img.reshape((height, width, 3))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    latest_frame = img


# ==========================================================
# MAIN PRECISION LANDING LOOP
# ==========================================================
async def run():

    global latest_frame

    # Connect to PX4
    drone = System()
    await drone.connect(system_address="udp://:14540")

    log.info("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            log.info("Drone connected")
            break

    # Wait for position estimate
    log.info("Waiting for global position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            log.info("Global position OK")
            break

    # Arm and takeoff
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(5)

    # Prepare Offboard
    await drone.offboard.set_velocity_ned(
        VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
    )

    try:
        await drone.offboard.start()
        log.info("OFFBOARD started")
    except OffboardError as e:
        log.error(f"Failed to start offboard: {e}")
        return

    # ======================================================
    # ARUCO SETUP
    # ======================================================
    camera_matrix = np.loadtxt("opencv/cameraMatrix.txt", delimiter=",")
    camera_distortion = np.loadtxt("opencv/cameraDistortion.txt", delimiter=",")

    aruco = ArucoSingleTracker(
        id_to_find=72,
        marker_size=0.1,  # meters
        camera_matrix=camera_matrix,
        camera_distortion=camera_distortion
    )

    Kp_xy = 0.8
    descend_rate = 0.3  # m/s
    max_xy_vel = 1.0
    landing_threshold = 0.2  # meters

    log.info("Precision landing active")

    while True:

        if latest_frame is None:
            await asyncio.sleep(0.02)
            continue

        found, x, y, z = aruco.track(latest_frame)

        if found:

            log.info(f"x={x:.3f} y={y:.3f} z={z:.3f}")

            # OpenCV camera frame:
            # X right, Y down, Z forward

            # For downward-facing camera:
            x_uav = -y
            y_uav = x

            # Proportional velocity control
            vx = Kp_xy * x_uav
            vy = Kp_xy * y_uav

            # Clamp
            vx = max(min(vx, max_xy_vel), -max_xy_vel)
            vy = max(min(vy, max_xy_vel), -max_xy_vel)

            # Descend when centered
            if abs(x) < 0.05 and abs(y) < 0.05:
                vz = descend_rate
            else:
                vz = 0.0

            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(
                    vx,      # North
                    vy,      # East
                    vz,      # Down (positive down!)
                    0.0
                )
            )

            if z < landing_threshold:
                log.warning("Landing")
                await drone.offboard.stop()
                await drone.action.land()
                break

        else:
            # No marker → hover
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            )

        await asyncio.sleep(0.03)


# ==========================================================
# ENTRY
# ==========================================================
if __name__ == "__main__":

    # Start Gazebo subscriber
    node = Node()
    topic = "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
    node.subscribe(Image, topic, image_callback)

    asyncio.run(run())
