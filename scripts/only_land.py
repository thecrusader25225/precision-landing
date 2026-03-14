import asyncio
import socket
import struct
import math
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw
#from mavsdk.action import FlightMode

# -----------------------------
# UDP VISION INPUT
# -----------------------------
UDP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", UDP_PORT))
sock.setblocking(False)


# -----------------------------
# PARAMETERS (unchanged)
# -----------------------------
ANGLE_DESCEND = math.radians(20)
LAND_HEIGHT = 0.3
DESCENT_RATE = 0.25
KP_MOVE = 0.6
MAX_SPEED = 0.7

HOME_RADIUS = 1.0
RTL_CAPTURE_HEIGHT = 8.0   # slightly higher for ArduPilot


# -----------------------------
# DISTANCE FUNCTION
# -----------------------------
def distance_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dLat = math.radians(lat2-lat1)
    dLon = math.radians(lon2-lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dLon/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))


# -----------------------------
# SIMPLE CONNECTION WAIT
# -----------------------------
async def wait_for_connection(drone):
    async for s in drone.core.connection_state():
        if s.is_connected:
            break


# -----------------------------
# WAIT FOR MISSION END
# Works for PX4 + ArduPilot
# -----------------------------
async def wait_for_mission_complete(drone):
    print("Waiting for mission...")
    seen_auto = False

    async for mode in drone.telemetry.flight_mode():

        name = mode.name

        if name in ["MISSION", "AUTO"]:
            seen_auto = True

        if seen_auto and name not in ["MISSION", "AUTO"]:
            print("Mission finished")
            return


# -----------------------------
# LANDING LOOP (UNCHANGED MATH)
# -----------------------------
async def precision_land(drone):

    print("Starting precision landing")

    while True:

        try:
            data,_ = sock.recvfrom(1024)
            found,x_cam,y_cam,z_cam = struct.unpack("ffff",data)
            x_cam /= 100.0
            y_cam /= 100.0
            z_cam /= 100.0
        except BlockingIOError:
            await asyncio.sleep(0.05)
            continue

        if found < 0.5:
            await drone.offboard.set_velocity_ned(
                VelocityNedYawspeed(0,0,0,0)
            )
            await asyncio.sleep(0.1)
            continue

        # camera -> UAV frame
        x_uav = -y_cam
        y_uav =  x_cam

        # get yaw
#        async for att in drone.telemetry.attitude_euler():
#            yaw = math.radians(att.yaw_deg)
#            break

        c = math.cos(yaw)
        s = math.sin(yaw)

        north = x_uav*c - y_uav*s
        east  = x_uav*s + y_uav*c

        angle_x = math.atan2(x_uav, z_cam)
        angle_y = math.atan2(y_uav, z_cam)
        angle_total = math.sqrt(angle_x**2 + angle_y**2)

        vx = KP_MOVE * north
        vy = KP_MOVE * east
        vx = max(min(vx, MAX_SPEED), -MAX_SPEED)
        vy = max(min(vy, MAX_SPEED), -MAX_SPEED)

        vz = DESCENT_RATE if angle_total <= ANGLE_DESCEND else 0.0
        print("vx :", vx,"vy: ", vy, "vz: ", vz)
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        if altitude < LAND_HEIGHT:
            print("Switching to LAND mode")
            await drone.action.land()
            return

        await drone.offboard.set_velocity_ned(
            VelocityNedYawspeed(vx, vy, vz, 0)
        )

        await asyncio.sleep(0.1)


# -----------------------------
# MAIN
# -----------------------------
async def run():

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14550")

    print("Waiting for connection...")
    await wait_for_connection(drone)
    print("Connected")

    # give telemetry time to start (important for ArduPilot)
    await asyncio.sleep(4)
    # switch to GUIDED for external control
    #await drone.action.set_mode("GUIDED")
    print("Sending initial setpoint...")
    await drone.offboard.set_velocity_ned(
    VelocityNedYawspeed(0.0, 0.0, 0.0, 0.0)
    )

    await asyncio.sleep(0.2) 
    await drone.offboard.start()
    await asyncio.sleep(2)

    print("External velocity control active")

    # small neutral command before landing loop
    await drone.offboard.set_velocity_ned(
        VelocityNedYawspeed(0,0,0,0)
    )
    await asyncio.sleep(1)

    # run landing loop
    await precision_land(drone)


asyncio.run(run())
