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
# LANDING PARAMETERS (UNCHANGED)
# -----------------------------
ANGLE_DESCEND = math.radians(20)
LAND_HEIGHT = 0.3
DESCENT_RATE = 0.25
KP_MOVE = 0.6
MAX_SPEED = 0.7

RTL_CAPTURE_HEIGHT = 5.0
TAG_TIMEOUT = 2.0


# -----------------------------
# HELPERS
# -----------------------------
async def wait_for_connection(drone):
    async for s in drone.core.connection_state():
        if s.is_connected:
            break


async def wait_for_global_position(drone):
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            break


async def wait_for_mission_complete(drone):
    print("Waiting for mission...")
    seen = False
    async for mode in drone.telemetry.flight_mode():
        if mode.name == "MISSION":
            seen = True
        if seen and mode.name != "MISSION":
            print("Mission finished")
            return

def distance_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dLat = math.radians(lat2-lat1)
    dLon = math.radians(lon2-lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dLon/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

async def wait_until_still(drone):
    print("Waiting for drone to stabilize...")

    stable_time = None

    async for vel in drone.telemetry.velocity_ned():
        speed = math.sqrt(vel.north_m_s**2 + vel.east_m_s**2)

        if speed < 0.15:
            if stable_time is None:
                stable_time = asyncio.get_event_loop().time()

            if asyncio.get_event_loop().time() - stable_time > 2.0:
                print("Drone stabilized")
                return
        else:
            stable_time = None

        await asyncio.sleep(0.1)


async def enter_offboard(drone):

    print("Priming OFFBOARD...")

    for _ in range(15):
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0,0,0,0)
        )
        await asyncio.sleep(0.05)

    try:
        await drone.offboard.start()
        print("OFFBOARD started")
    except OffboardError as e:
        print("OFFBOARD failed:", e)
        return False

    return True


# -----------------------------
# OLD WORKING LANDING LOOP
# -----------------------------
async def precision_land_old(drone):

    print("Starting OLD landing logic")

    while True:

        # read vision
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
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0,0,0,0)
            )
            await asyncio.sleep(0.05)
            continue

        # camera → UAV
        x_uav = -y_cam
        y_uav =  x_cam

        # yaw → NED
        async for att in drone.telemetry.attitude_euler():
            yaw = math.radians(att.yaw_deg)
            break

        c = math.cos(yaw)
        s = math.sin(yaw)

        north = x_uav*c - y_uav*s
        east  = x_uav*s + y_uav*c

        # angle
        angle_x = math.atan2(x_uav, z_cam)
        angle_y = math.atan2(y_uav, z_cam)
        angle_total = math.sqrt(angle_x**2 + angle_y**2)

        # horizontal
        vx = KP_MOVE * north
        vy = KP_MOVE * east

        vx = max(min(vx, MAX_SPEED), -MAX_SPEED)
        vy = max(min(vy, MAX_SPEED), -MAX_SPEED)

        # descent
        if angle_total <= ANGLE_DESCEND:
            vz = DESCENT_RATE
        else:
            vz = 0.0

        # altitude
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        if altitude < LAND_HEIGHT:
            print("Landing")
            await drone.offboard.stop()
            await drone.action.land()
            return

        # send
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(vx, vy, vz, 0)
        )

        await asyncio.sleep(0.03)


# -----------------------------
# MAIN
# -----------------------------
async def run():

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Connected")
    await wait_for_connection(drone)
    await wait_for_global_position(drone)

    print("Saving HOME position...")
    async for home in drone.telemetry.home():
        home_lat = home.latitude_deg
        home_lon = home.longitude_deg
        print("Home saved:", home_lat, home_lon)
        break

    # wait mission
    await wait_for_mission_complete(drone)

    # RTL
    print("Trigger RTL")
    await drone.action.return_to_launch()

    tag_seen_time = None

    while True:

        # altitude check
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        # tag visible?
        try:
            data,_ = sock.recvfrom(1024)
            found,_,_,_ = struct.unpack("ffff",data)
        except BlockingIOError:
            found = 0.0

        # get current GPS
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            lat = pos.latitude_deg
            lon = pos.longitude_deg
            break

        dist_home = distance_m(lat, lon, home_lat, home_lon)

        # takeover only when REALLY at home
        if found and altitude < RTL_CAPTURE_HEIGHT and dist_home < 1.5:

            if tag_seen_time is None:
                tag_seen_time = asyncio.get_event_loop().time()

            if asyncio.get_event_loop().time() - tag_seen_time > TAG_TIMEOUT:

                print("TAKEOVER")

                await drone.action.hold()

                await wait_until_still(drone)

                ok = await enter_offboard(drone)
                if not ok:
                    return

                await precision_land_old(drone)
                return

        else:
            tag_seen_time = None

        await asyncio.sleep(0.2)


asyncio.run(run())