import asyncio
import socket
import struct
import math
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError


# -----------------------------
# UDP VISION INPUT
# -----------------------------
UDP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", UDP_PORT))
sock.setblocking(False)


# -----------------------------
# PARAMETERS (same idea as DroneKit script)
# -----------------------------
ANGLE_DESCEND = math.radians(20)
LAND_HEIGHT = 0.3
DESCENT_RATE = 0.25
KP_MOVE = 0.6
MAX_SPEED = 0.7


# -----------------------------
# MAIN LOOP
# -----------------------------
async def run():

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")


    print("Starting OFFBOARD")

    await drone.offboard.set_velocity_ned(
        VelocityNedYaw(0,0,0,0)
    )
    await asyncio.sleep(0.2)

    try:
        await drone.offboard.start()
    except OffboardError as e:
        print("Offboard failed:", e)
        return


    while True:

        # -----------------------------
        # READ VISION
        # -----------------------------
        try:
            data,_ = sock.recvfrom(1024)
            found,x_cam,y_cam,z_cam = struct.unpack("ffff",data)
            x_cam /= 100.0
            y_cam /= 100.0
            z_cam /= 100.0

        except BlockingIOError:
            await asyncio.sleep(0.02)
            continue

        if found < 0.5:
            print("Tag lost → hover")
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0,0,0,0)
            )
            await asyncio.sleep(0.05)
            continue


        # -----------------------------
        # CAMERA → UAV FRAME
        # (same as DroneKit camera_to_uav)
        # -----------------------------
        x_uav = -y_cam     # forward
        y_uav =  x_cam     # right


        # -----------------------------
        # UAV → NED using yaw
        # (same as uav_to_ne)
        # -----------------------------
        async for att in drone.telemetry.attitude_euler():
            yaw = math.radians(att.yaw_deg)
            break

        c = math.cos(yaw)
        s = math.sin(yaw)

        north = x_uav*c - y_uav*s
        east  = x_uav*s + y_uav*c


        # -----------------------------
        # ANGLE COMPUTATION
        # (same as marker_position_to_angle)
        # -----------------------------
        angle_x = math.atan2(x_uav, z_cam)
        angle_y = math.atan2(y_uav, z_cam)

        angle_total = math.sqrt(angle_x**2 + angle_y**2)


        # -----------------------------
        # HORIZONTAL MOVEMENT
        # (DroneKit simple_goto equivalent)
        # -----------------------------
        vx = KP_MOVE * north
        vy = KP_MOVE * east

        vx = max(min(vx, MAX_SPEED), -MAX_SPEED)
        vy = max(min(vy, MAX_SPEED), -MAX_SPEED)


        # -----------------------------
        # DESCENT CONDITION
        # (same logic as DroneKit)
        # -----------------------------
        if angle_total <= ANGLE_DESCEND:
            vz = DESCENT_RATE
            print("Low error → descending")
        else:
            vz = 0.0
            print("Correcting position")


        # -----------------------------
        # LAND TRIGGER
        # -----------------------------
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        if altitude < LAND_HEIGHT:
            print("Switching to LAND")
            await drone.offboard.stop()
            await drone.action.land()
            return


        # -----------------------------
        # DEBUG
        # -----------------------------
        print(
            f"x={x_cam:.2f} y={y_cam:.2f} z={z_cam:.2f} "
            f"| N={north:.2f} E={east:.2f} "
            f"| angle={math.degrees(angle_total):.1f}"
        )


        # -----------------------------
        # SEND COMMAND
        # -----------------------------
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(vx, vy, vz, 0)
        )
        print("DESCENT CHECK:",
        math.degrees(angle_total),
        "<", math.degrees(ANGLE_DESCEND))


        await asyncio.sleep(0.03)


asyncio.run(run())
