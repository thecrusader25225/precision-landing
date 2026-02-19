import asyncio
import socket
import struct
import math
import time
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
# PARAMETERS
# -----------------------------
UPDATE_PERIOD = 0.05      # controller rate (fast = stable)
GAIN = 0.8                # horizontal correction gain
MAX_SPEED = 0.6           # horizontal velocity limit
LAND_HEIGHT = 0.35        # stop offboard here


# -----------------------------
# MAIN
# -----------------------------
async def run():

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break

    print("Connected")

    async for armed in drone.telemetry.armed():
        if armed:
            break

    async for in_air in drone.telemetry.in_air():
        if in_air:
            break

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

    last_update = 0

    # -----------------------------
    # LOOP
    # -----------------------------
    while True:

        # -----------------------------
        # RECEIVE VISION
        # -----------------------------
        try:
            data, _ = sock.recvfrom(1024)
            found, x_cam, y_cam, z_cam = struct.unpack("ffff", data)
        except BlockingIOError:
            await asyncio.sleep(0.01)
            continue

        if found < 0.5:
            print("Tag lost → holding")
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0,0,0,0)
            )
            await asyncio.sleep(0.05)
            continue

        # -----------------------------
        # RATE LIMIT
        # -----------------------------
        now = time.time()
        if now - last_update < UPDATE_PERIOD:
            await asyncio.sleep(0.005)
            continue
        last_update = now

        # -----------------------------
        # CAMERA FRAME → BODY ERRORS
        #
        # ArUco:
        #   x = right
        #   y = down
        #   z = forward
        #
        # UAV body:
        #   +X forward
        #   +Y right
        #
        # Therefore:
        #   forward error = y
        #   right error   = x
        # -----------------------------
        err_forward = y_cam
        err_right   = x_cam

        # -----------------------------
        # HORIZONTAL CONTROL
        # -----------------------------
        vx_body = -GAIN * err_forward
        vy_body = -GAIN * err_right

        vx_body = max(min(vx_body, MAX_SPEED), -MAX_SPEED)
        vy_body = max(min(vy_body, MAX_SPEED), -MAX_SPEED)

        # -----------------------------
        # GET YAW → rotate BODY to NED
        # -----------------------------
        async for att in drone.telemetry.attitude_euler():
            yaw = math.radians(att.yaw_deg)
            break

        c = math.cos(yaw)
        s = math.sin(yaw)

        north = vx_body*c - vy_body*s
        east  = vx_body*s + vy_body*c

        # -----------------------------
        # DESCENT CONDITION
        # descend only when centred
        # -----------------------------
        if abs(err_forward) < 0.10 and abs(err_right) < 0.10:

            if z_cam > 1.5:
                vz = 0.5
            elif z_cam > 0.8:
                vz = 0.35
            elif z_cam > 0.5:
                vz = 0.20
            else:
                vz = 0.10

            print("DESCENDING")

        else:
            vz = 0.0
            print("CORRECTING")

        # -----------------------------
        # LAND TRIGGER
        # -----------------------------
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        if altitude < LAND_HEIGHT:
            print("Landing height reached → stopping offboard")
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0,0,0,0)
            )
            await asyncio.sleep(0.3)
            await drone.offboard.stop()
            return

        # -----------------------------
        # DEBUG
        # -----------------------------
        print(
            f"x={x_cam:.2f} y={y_cam:.2f} z={z_cam:.2f} "
            f"| err_f={err_forward:.2f} err_r={err_right:.2f}"
        )

        # -----------------------------
        # SEND COMMAND
        # -----------------------------
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(north, east, vz, 0.0)
        )

        await asyncio.sleep(0.01)


asyncio.run(run())
