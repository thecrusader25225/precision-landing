import asyncio
import socket
import struct
import math
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
from mavsdk.mission import MissionProgress

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
MAX_SPEED = 0.15             # m/s clamp
DESCENT_RATE = 0.15         # m/s downward
ANGLE_DESCEND = 0.349066    # 20 deg
LAND_HEIGHT = 0.5           # meters
DEADBAND = 0.02             # 2 cm deadband


# DISTANCE FUNCTION
def distance_m(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

# MISSION FINISH CHECK
async def wait_for_mission_and_rtl(drone):

    print("Waiting for mission to finish...")

    async for progress in drone.mission.mission_progress():
        print(f"Mission progress: {progress.current}/{progress.total}")

        if progress.current >= progress.total - 1 and progress.total != 0:
            print("Mission finished. Switching to RTL")
            await drone.action.return_to_launch()
            break

# WAIT UNTIL DRONE REACHES HOME
async def wait_until_home(drone, home_lat, home_lon):

    print("Waiting until drone reaches home...")

    while True:
        async for pos in drone.telemetry.position():
            current_lat = pos.latitude_deg
            current_lon = pos.longitude_deg
            altitude = pos.relative_altitude_m
            break

        async for vel in drone.telemetry.velocity_ned():
            speed_xy = math.sqrt(vel.north_m_s**2 + vel.east_m_s**2)
            break

        dist = distance_m(current_lat, current_lon, home_lat, home_lon)

        print(f"Dist: {dist:.2f} | Alt: {altitude:.2f} | Speed: {speed_xy:.2f}")

        if dist < 2.0 and altitude < 7 and speed_xy < 0.3:
            print("Stable near home. start precision landing")
            break

        await asyncio.sleep(0.2)
# -----------------------------
# PRECISION LANDING LOOP
# -----------------------------
async def precision_land(drone):

    print("Starting precision landing")

    while True:
        latest = None

        while True:
            try:
                data, _ = sock.recvfrom(1024)
                latest = data
            except BlockingIOError:
                break

            if latest is None:
                await asyncio.sleep(0.02)
                continue

            packet_id, found, x_cam, y_cam, z_cam = struct.unpack("Iffff", latest)
        
        
        print(f"TAG: ",packet_id, found, x_cam, y_cam, z_cam)
        # convert cm → meters
        x_cam /= 100.0
        y_cam /= 100.0
        z_cam /= 100.0

        # -----------------------------
        # If tag not found → hover
        # -----------------------------
        if found < 0.5:
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
            print("NO TAG")
            await asyncio.sleep(0.1)
            continue
        
        
        # -----------------------------
        # OpenCV camera → UAV BODY frame
        # Assumes:
        # - Downward-facing camera
        # - Image top aligned with drone nose
        # -----------------------------
        x_body = -y_cam   # forward/back
        y_body =  x_cam   # right/left

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
        max_vel = min(MAX_SPEED, 0.05 + 0.5 * z_cam)

        vx = KP_MOVE * x_body
        vy = KP_MOVE * y_body

        vx = max(min(vx, max_vel), -max_vel)
        vy = max(min(vy, max_vel), -max_vel)

        if angle_total <= ANGLE_DESCEND:
            vz = DESCENT_RATE
        else:
            vz = 0.0   # no descent while centering
        print(f"vx: {vx:.2f}  vy: {vy:.2f}  vz: {vz:.2f}")

        # -----------------------------
        # Check altitude
        # -----------------------------
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        if altitude < LAND_HEIGHT:
            print("Switching to LAND mode")
            await drone.action.land()
            return

        # -----------------------------
        # Send BODY velocity
        # -----------------------------
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(vx, vy, vz, 0.0)
        )

        await asyncio.sleep(0.1)


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
    print("Connected. Hold 4 seconds for stabilization...")

    await asyncio.sleep(4)
    
    # get home once
    async for home in drone.telemetry.home():
        home_lat = home.latitude_deg
        home_lon = home.longitude_deg
        print(f"Home Coords: {home_lat}, {home_lon}")
        break

    # Wait for mission completion
    # while True:
    #     is_finished = await drone.mission.is_mission_finished()
    #     if is_finished:
    #         print("Mission finished → triggering RTL")
    #         try:
    #             await drone.action.return_to_launch()
    #         except Exception as e:
    #             print(f"RTL may already be active: {e}")
    #         break
    #     await asyncio.sleep(1)  # prevent CPU spam

    # Wait until drone returns home
    await wait_until_home(drone, home_lat, home_lon)

    # Send initial neutral setpoint
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    )

    await asyncio.sleep(0.2)

    # Start offboard
    await drone.offboard.start()
    print("Offboard mode started")

    await asyncio.sleep(1)

    # Run precision landing
    await precision_land(drone)

if __name__ == "__main__":
    asyncio.run(run())
