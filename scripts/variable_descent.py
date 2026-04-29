import asyncio
import socket
import struct
import math
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
import time
# -----------------------------
# UDP INPUT (from vision script)
# -----------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# -----------------------------
# CONTROL PARAMETERS
# -----------------------------
KP_MOVE = 0.15              # proportional gain
MAX_SPEED = 0.25             # m/s clamp
LAND_HEIGHT = 0.5           # meters
DEADBAND = 0.05             # 5 cm deadband

#----adaptive angle thresholds for descent gating-----
# ANGLE_DESCEND = 0.174533    #10 deg
ANGLE_LOW  = math.radians(5.0)   # at ≤ 1m
ANGLE_HIGH = math.radians(15.0)  # at ≥ 2m

#---adaptive descent-----
DESCENT_RATE = 0.15         # m/s downward
DESCENT_MIN = 0.07
DESCENT_MAX = 0.20

async def yaw_align(drone):
    print("Starting yaw alignment...")

    ALIGN_THRESH = math.radians(5)
    STABLE_TIME = 1.0  # seconds to hold alignment

    aligned_since = None

    while True:
        latest = None

        # get latest packet
        while True:
            try:
                data, _ = sock.recvfrom(1024)
                latest = data
            except BlockingIOError:
                break

        if latest is None:
            await asyncio.sleep(0.02)
            continue

        packet_id, f72, x72, y72, z72, fX, xX, yX, zX = struct.unpack("Iffffffff", latest)
        print(f"id: ",packet_id,"Tag72: ", f72, x72, y72, z72,"TagX: ",fX, xX, yX, zX )
        # need both tags
        if not (f72 > 0.5 and fX > 0.5):
            print("Waiting for both tags...")
            await asyncio.sleep(0.05)
            continue

        dx = xX - x72
        dy = yX - y72

        yaw_error = math.atan2(dx, -dy)

        KP_YAW = 1.5
        yaw_rate = math.degrees(KP_YAW * yaw_error)
        yaw_rate = max(min(yaw_rate, 30.0), -30.0)

        print(f"Yaw error: {math.degrees(yaw_error):.2f}")

        # send yaw-only command
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, yaw_rate)
        )

        # alignment check
        if abs(yaw_error) < ALIGN_THRESH:
            if aligned_since is None:
                aligned_since = time.time()
            elif time.time() - aligned_since > STABLE_TIME:
                print("Yaw aligned and stable")
                break
        else:
            aligned_since = None

        await asyncio.sleep(0.02)

# CIRCULAR SEARCH
async def circular_search(drone):
    print("Starting spiral search")

    start_time = time.time()
    seen_count = 0

    BASE_VX = 0.15
    GROWTH_RATE = 0.1
    MAX_VX = 0.4
    YAW_RATE = 20.0

    while True:
        latest = None

        # get latest packet
        while True:
            try:
                data, _ = sock.recvfrom(1024)
                latest = data
            except BlockingIOError:
                break

        if latest is not None:
            packet_id, f72, x72, y72, z72, *_ = struct.unpack("Iffffffff", latest)

            if f72 >= 0.5:
                seen_count += 1
            else:
                seen_count = 0

            if seen_count >= 1:
                print("TAG FOUND → stopping search")

                # HARD STOP
                for _ in range(5):
                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                    )
                    await asyncio.sleep(0.1)

                return   # 🔑 exit search

        # circular motion
        t = time.time() - start_time

        # LOG SPIRAL (exponential growth)
        BASE_VX = 0.15
        GROWTH_FACTOR = 0.1   # critical parameter
        MAX_VX = 0.4

        vx = BASE_VX * math.exp(GROWTH_FACTOR * t)
        vx = min(vx, MAX_VX)

        vy = 0.0
        vz = 0.03

        # yaw must scale stronger here
        yaw_rate = 10.0 + 40.0 * vx
        yaw_rate = min(yaw_rate, 25.0)

        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(vx, vy, vz, yaw_rate)
        )

        print(f"[SEARCH] vx:{vx:.2f} yaw:{yaw_rate}")

        await asyncio.sleep(0.02)


# -----------------------------
# PRECISION LANDING LOOP
# -----------------------------
async def precision_land(drone):
    stable_since = None
    last_seen_time = time.time()
    last_x, last_y, last_z = 0.0, 0.0, 0.0
    print("Starting precision landing")

    while True:
        latest = None
        vz = 0
        while True:
            try:
                data, _ = sock.recvfrom(1024)
                latest = data
            except BlockingIOError:
                break

        if latest is None:
            await asyncio.sleep(0.02)
            continue

        packet_id, f72, x72, y72, z72,fX, xX, yX, zX  = struct.unpack("Iffffffff", latest)
        
        
        print(f"TAG72: ",packet_id, f72, x72, y72, z72)
        x_cam, y_cam, z_cam = 0.0, 0.0, 0.0
        # -----------------------------
        # Choose landing tag
        # -----------------------------
        if f72 > 0.5:
            x_cam, y_cam, z_cam = x72, y72, z72
        

        # convert cm → meters
        x_cam /= 100.0
        y_cam /= 100.0
        z_cam /= 100.0
        if z_cam < 0.4 and z_cam >0.1:
            print("Switching to LAND mode")
            await drone.action.land()
            return
        # -----------------------------
        # OpenCV camera → UAV BODY frame
        # Assumes:
        # - Downward-facing camera
        # - Image top aligned with drone nose
        # -----------------------------
        x_body = -y_cam - 0.035  # forward/back
        y_body = x_cam   # right/left

        # Noise filter
        if abs(x_body) > 1.5 or abs(y_body) > 1.5 or z_cam > 3:
            print("OUTLIER → ignored")
            continue
        current_time = time.time()

        if f72 >= 0.5:
            last_seen_time = current_time
            last_x = x_body
            last_y = y_body
            last_z = z_cam

        time_since_seen = current_time - last_seen_time
        if f72 < 0.5:
            x_body = last_x
            y_body = last_y
            # z_cam = last_z   # optional but useful
            print("SHORT LOSS → continuing")
        # -----------------------------
        # Deadband
        # -----------------------------
        if abs(x_body) < DEADBAND:
            x_body = 0.0
        if abs(y_body) < DEADBAND:
            y_body = 0.0
        # -----------------------------
        # Angle check for descent gating
        # -----------------------------
        angle_x = math.atan2(x_body, z_cam)
        angle_y = math.atan2(y_body, z_cam)
        angle_total = math.sqrt(angle_x**2 + angle_y**2)
        print(f"Angle: {math.degrees(angle_total):.1f}")
        # -----------------------------
        # Proportional velocity control
        # -----------------------------
        # dynamic gain
        kp_dynamic = KP_MOVE * min(1.0, z_cam / 2.0)

        vx = kp_dynamic * x_body
        vy = kp_dynamic * y_body

        # error magnitude
        error_mag = math.sqrt(x_body**2 + y_body**2)

        # scale for large errors
        if error_mag > 0.3:
            scale = min(2.0, error_mag / 0.3)
            vx *= scale
            vy *= scale

        # adaptive clamp
        base_vel = 0.05 + 0.5 * z_cam
        boost = min(0.3, error_mag)

        max_vel = min(MAX_SPEED + boost, base_vel + boost)

        # clamp
        vx = max(min(vx, max_vel), -max_vel)
        vy = max(min(vy, max_vel), -max_vel)

        #RATE LIMITER

        MAX_DELTA = 0.02   # m/s per step (tune 0.015–0.03)

        if not hasattr(precision_land, "prev_vx"):
            precision_land.prev_vx = 0.0
            precision_land.prev_vy = 0.0

        vx = max(min(vx, precision_land.prev_vx + MAX_DELTA),
                precision_land.prev_vx - MAX_DELTA)

        vy = max(min(vy, precision_land.prev_vy + MAX_DELTA),
                precision_land.prev_vy - MAX_DELTA)

        precision_land.prev_vx = vx
        precision_land.prev_vy = vy

        XY_THRESH = 0.05      # 5 cm
        STABLE_TIME = 0.4     # seconds
        # allow_angle = angle_total < ANGLE_DESCEND

 

        if z_cam <= 1.0:
            angle_thresh = ANGLE_LOW
        elif z_cam >= 2.0:
            angle_thresh = ANGLE_HIGH
        else:
            # linear interpolation between 1m and 2m
            t = (z_cam - 1.0) / (2.0 - 1.0)   # 0 → 1
            angle_thresh = ANGLE_LOW + t * (ANGLE_HIGH - ANGLE_LOW)

        allow_angle = angle_total < angle_thresh

        # allow_xy    = abs(x_body) < XY_THRESH and abs(y_body) < XY_THRESH
        if f72 < 0.5:
            vz = 0
        elif allow_angle:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since > STABLE_TIME:
                vz = DESCENT_RATE
                #z_norm = max(0.0, min(1.0, z_cam / 2.0))
                #vz = DESCENT_MIN + (DESCENT_MAX - DESCENT_MIN) * z_norm
                print("Stable angle → descending")
            else:
                vz = 0.0
        else:
            stable_since = None
            vz = 0.0
        print(f"vx: {vx:.2f}  vy: {vy:.2f}  vz: {vz:.2f}")

        # -----------------------------
        # Check altitude
        # -----------------------------
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        

        
        # -----------------------------
        # Send BODY velocity
        # -----------------------------
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(vx, vy, vz, 0)
        )

        await asyncio.sleep(0.02)

# -----------------------------
# MAIN
# -----------------------------
async def run():

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14550")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    print("Connected")

    # Allow telemetry to stabilize (important for ArduPilot)
    await asyncio.sleep(1)

    # Wait 2 seconds for tag
    print("Checking for tag before search...")
    start = time.time()

    tag_seen = False

    while time.time() - start < 2.0:
        try:
            data, _ = sock.recvfrom(1024)
            packet_id, f72, *_ = struct.unpack("Iffffffff", data)
            if f72 >= 0.5:
                tag_seen = True
                break
        except BlockingIOError:
            pass

        await asyncio.sleep(0.05)


    # Send initial neutral setpoint (required before offboard.start())
    #await drone.offboard.set_velocity_body(
    #    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    #)
    for _ in range(10):   # ~1 second
        await drone.offboard.set_velocity_body(
           VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        )

    await asyncio.sleep(0.1)

    # Start offboard mode
    await drone.offboard.start()
    print("Offboard mode started")
        # If not found → search
    print("Stabilizing...")

    for _ in range(20):   # ~2 seconds
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, -0.001, 0.0)
        )
        await asyncio.sleep(0.1)
        
    if not tag_seen:
        await circular_search(drone)

    await asyncio.sleep(1)
    print("Running yaw_align")
    #Run yaw alignment
    await yaw_align(drone)
    print("Run precision_land")
    # Run landing
    await precision_land(drone)


if __name__ == "__main__":
    asyncio.run(run())
