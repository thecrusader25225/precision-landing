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
# PARAMETERS (mirror dronekit script behaviour)
# -----------------------------
UPDATE_PERIOD = 0.05            # seconds between control updates
MOVE_GAIN = 0.6                # how strongly to move toward tag
MAX_SPEED = 0.7                # horizontal speed limit
DESCENT_SPEED = 0.25           # downward speed when aligned
ANGLE_DESCEND = math.radians(20)   # allowed angle before descent
LAND_HEIGHT = 0.35             # stop offboard below this height


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

    # wait until armed and in air
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

    while True:

        # -----------------------------
        # READ VISION DATA
        # -----------------------------
        try:
            data, _ = sock.recvfrom(1024)
            found, x, y, z = struct.unpack("ffff", data)
            # ---- CAMERA → CG OFFSET (meters) ----
            CAMERA_FORWARD_OFFSET = 0.12   # camera ahead of CG
            CAMERA_RIGHT_OFFSET   = -0.05  # camera left of CG

            x -= CAMERA_RIGHT_OFFSET
            y -= CAMERA_FORWARD_OFFSET
            print(f"RAW AFTER CG FIX → x={x:.2f} y={y:.2f}")
            # --- camera mounting offset correction ---
            # X_OFFSET = -0.10
            # Y_OFFSET = -0.04

            # x -= X_OFFSET
            # y -= Y_OFFSET
        except BlockingIOError:
            await asyncio.sleep(0.02)
            continue

        # ignore lost tag
        if found < 0.5:
            print("Tag lost → holding")
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0,0,0,0)
            )
            await asyncio.sleep(0.05)
            continue

        # -----------------------------
        # CONTROL UPDATE RATE LIMIT
        # -----------------------------
        now = time.time()
        if now - last_update < UPDATE_PERIOD:
            await asyncio.sleep(0.01)
            continue
        last_update = now

        # -----------------------------
        # CAMERA → UAV BODY FRAME
        # -----------------------------
        # ArUco gives camera frame:
        # x = right, y = down, z = forward

        x_cam = x
        y_cam = y
        z_cam = z

        # convert to UAV frame (same as DroneKit code)
        x_uav = -y_cam      # forward
        y_uav =  x_cam      # right


        # -----------------------------
        # ANGLE ERROR (CRITICAL FIX)
        # -----------------------------
        # angle_x = math.atan2(x_uav, z_cam)
        # angle_y = math.atan2(y_uav, z_cam)
        angle_x = math.atan2(x_uav, z_cam)
        angle_y = math.atan2(y_uav, z_cam)


        angle_total = math.sqrt(angle_x**2 + angle_y**2)
        horizontal_dist = math.sqrt(x_uav**2 + y_uav**2)

        # STRICT gate before descent
        if horizontal_dist < 0.08:   # 8 cm
            # descend
            if z > 1.5:
                vz = 0.6
            elif z > 0.8:
                vz = 0.4
            elif z > 0.5:
                vz = 0.25
            else:
                vz = 0.12
            print("DESCENDING (locked)")

        else:
            vz = 0.0
            print("HOLD ALTITUDE — correcting")


        # -----------------------------
        # GET YAW
        # -----------------------------
        async for att in drone.telemetry.attitude_euler():
            yaw = math.radians(att.yaw_deg)
            break


        # -----------------------------
        # BODY → NED FRAME
        # -----------------------------
        c = math.cos(yaw)
        s = math.sin(yaw)

        north = x_uav*c - y_uav*s
        east  = x_uav*s + y_uav*c


        # -----------------------------
        # HORIZONTAL CONTROL (ANGLE BASED)
        # -----------------------------
        ANGLE_GAIN = 1.2
        MAX_SPEED = 0.6

        vx = ANGLE_GAIN * angle_x
        vy = ANGLE_GAIN * angle_y
        # vx = MOVE_GAIN * north
        # vy = MOVE_GAIN * east

        vx = max(min(vx, MAX_SPEED), -MAX_SPEED)
        vy = max(min(vy, MAX_SPEED), -MAX_SPEED)


        # -----------------------------
        # DESCENT LOGIC (DroneKit style)
        # # -----------------------------
        # ANGLE_DESCEND = math.radians(12)

        # if angle_total < ANGLE_DESCEND:
        #     # descend faster when aligned
        #     if z_cam > 1.5:
        #         vz = 0.35
        #     elif z_cam > 0.8:
        #         vz = 0.25
        #     elif z_cam > 0.5:
        #         vz = 0.15
        #     else:
        #         vz = 0.08

        #     print(f"Aligned → descending vz={vz:.2f}")

        # else:
        #     vz = 0.0
        #     print("Correcting position")


        # -----------------------------
        # LAND TRIGGER
        # -----------------------------
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        if altitude < 0.35:
            print("Landing height reached → stopping offboard")
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0,0,0,0)
            )
            await asyncio.sleep(0.3)
            await drone.offboard.stop()
            return


        # -----------------------------
        # DEBUG PRINT
        # -----------------------------
        print(
            f"x={x:.2f} y={y:.2f} z={z:.2f} "
            f"| angle={math.degrees(angle_total):.1f}"
        )


        # -----------------------------
        # SEND COMMAND
        # -----------------------------
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(vx, vy, vz, 0.0)
        )
        # -----------------------------
        # DEBUG TEST — raw axis motion
        # -----------------------------
        # vx = 0.0
        # vy = 0.0
        # vz = 0.0

        # # move north/south based on x
        # if abs(x) > 0.05:
        #     vx = 0.3 if x > 0 else -0.3

        # # move east/west based on y
        # if abs(y) > 0.05:
        #     vy = 0.3 if y > 0 else -0.3

        # print(f"TEST MOVE → x={x:.2f} y={y:.2f} | vx={vx:.2f} vy={vy:.2f}")

        # await drone.offboard.set_velocity_ned(
        #     VelocityNedYaw(vx, vy, vz, 0.0)
        # )


        await asyncio.sleep(0.02)


asyncio.run(run())