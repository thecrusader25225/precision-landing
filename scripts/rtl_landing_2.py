import asyncio
import socket
import struct
import math
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError


UDP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", UDP_PORT))
sock.setblocking(False)


ANGLE_DESCEND = math.radians(20)
LAND_HEIGHT = 0.3
DESCENT_RATE = 0.25
KP_MOVE = 0.6
MAX_SPEED = 0.7

RTL_CAPTURE_HEIGHT = 15.0
TAG_TIMEOUT = 2.0
HOME_RADIUS = 1.0   # meters


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


async def wait_until_slow(drone):
    print("Waiting for drone to slow down...")
    stable_time = None

    async for vel in drone.telemetry.velocity_ned():
        speed = math.sqrt(vel.north_m_s**2 + vel.east_m_s**2)

        if speed < 0.2:
            if stable_time is None:
                stable_time = asyncio.get_event_loop().time()

            if asyncio.get_event_loop().time() - stable_time > 1.5:
                print("Drone slow enough")
                return
        else:
            stable_time = None

        await asyncio.sleep(0.1)

async def wait_until_px4_ready(drone):

    print("Waiting for PX4 to fully settle after RTL...")

    stable_time = None

    while True:

        # velocity
        async for vel in drone.telemetry.velocity_ned():
            horiz = math.sqrt(vel.north_m_s**2 + vel.east_m_s**2)
            vz = vel.down_m_s
            break

        # yaw rate
        async for att in drone.telemetry.attitude_angular_velocity_body():
            yaw_rate = abs(att.yaw_rad_s)
            break

        # must be:
        # not moving
        # not descending
        # not rotating
        if horiz < 0.2 and abs(vz) < 0.1 and yaw_rate < 0.05:

            if stable_time is None:
                stable_time = asyncio.get_event_loop().time()

            if asyncio.get_event_loop().time() - stable_time > 3.0:
                print("PX4 fully settled")
                return

        else:
            stable_time = None

        await asyncio.sleep(0.2)
        
async def enter_offboard(drone):

    print("Preparing OFFBOARD stream...")

    # Step 1: wait extra for PX4 commander to fully settle
    await asyncio.sleep(2.5)

    # Step 2: start streaming neutral setpoints
    # PX4 requires this BEFORE offboard.start()
    for _ in range(40):
        try:
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            )
        except Exception as e:
            print("Waiting for MAVSDK recovery...", e)
            await asyncio.sleep(0.2)
            continue

        await asyncio.sleep(0.05)

    # Step 3: now start offboard
    try:
        await drone.offboard.start()
        print("OFFBOARD started successfully")
        return True

    except OffboardError as e:
        print("OFFBOARD failed:", e)
        return False

async def reset_and_enter_offboard(drone):

    print("Forcing HOLD mode...")
    await drone.action.hold()

    # --- wait until PX4 actually switches mode ---
    async for mode in drone.telemetry.flight_mode():
        if mode.name == "HOLD":
            break
        await asyncio.sleep(0.2)

    print("HOLD confirmed")

    # --- wait until drone truly stops ---
    print("Waiting for drone to stop...")

    stable_time = None

    async for vel in drone.telemetry.velocity_ned():

        speed = math.sqrt(vel.north_m_s**2 + vel.east_m_s**2)

        if speed < 0.15:
            if stable_time is None:
                stable_time = asyncio.get_event_loop().time()

            if asyncio.get_event_loop().time() - stable_time > 2.0:
                break
        else:
            stable_time = None

        await asyncio.sleep(0.1)

    print("Drone motion stabilized")

    # --- give PX4 internal controllers time to release ---
    print("Giving PX4 time to release RTL controllers...")
    await asyncio.sleep(2.5)

    # --- now start sending neutral setpoints BEFORE starting offboard ---
    print("Priming OFFBOARD stream...")

    for _ in range(25):
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0,0,0,0)
        )
        await asyncio.sleep(0.05)

    # --- start offboard ---
    try:
        await drone.offboard.start()
        print("OFFBOARD started cleanly")
    except OffboardError as e:
        print("OFFBOARD failed:", e)
        return False

    # --- hold neutral commands briefly so PX4 locks in ---
    start = asyncio.get_event_loop().time()

    while asyncio.get_event_loop().time() - start < 1.5:
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0,0,0,0)
        )
        await asyncio.sleep(0.05)

    print("OFFBOARD stable")
    return True

async def clean_takeover_to_offboard(drone):

    print("---- TAKEOVER START ----")

    # 1️⃣ Stop navigator (this cancels RTL safely)
    await drone.action.hold()

    # wait until mode actually changes
    async for mode in drone.telemetry.flight_mode():
        if mode.name == "HOLD":
            break
        await asyncio.sleep(0.1)

    print("HOLD confirmed")

    # 2️⃣ Freeze altitude by waiting for vertical speed ≈ 0
    stable = None
    async for vel in drone.telemetry.velocity_ned():
        if abs(vel.down_m_s) < 0.1:
            if stable is None:
                stable = asyncio.get_event_loop().time()
            if asyncio.get_event_loop().time() - stable > 1.2:
                break
        else:
            stable = None
        await asyncio.sleep(0.1)

    print("Vertical motion stabilized")

    # 3️⃣ Send neutral setpoints BEFORE starting offboard
    print("Priming OFFBOARD stream")

    for _ in range(20):
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0,0,0,0)
        )
        await asyncio.sleep(0.05)

    # 4️⃣ Start OFFBOARD
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print("OFFBOARD failed:", e)
        return False

    print("OFFBOARD engaged")

    # 5️⃣ Continue sending neutral commands briefly
    t0 = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - t0 < 1.0:
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0,0,0,0)
        )
        await asyncio.sleep(0.05)

    print("OFFBOARD stable")
    return True

# -----------------------------
# OLD LANDING LOOP (UNCHANGED)
# -----------------------------
async def precision_land_old(drone):

    print("Starting precision landing")

    while True:

        try:
            data, _ = sock.recvfrom(1024)
            found, x_cam, y_cam, z_cam = struct.unpack("ffff", data)
            x_cam /= 100.0
            y_cam /= 100.0
            z_cam /= 100.0
        except BlockingIOError:
            await asyncio.sleep(0.02)
            continue

        if found < 0.5:
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0, 0, 0, 0)
            )
            await asyncio.sleep(0.05)
            continue

        x_uav = -y_cam
        y_uav =  x_cam

        async for att in drone.telemetry.attitude_euler():
            yaw = math.radians(att.yaw_deg)
            break

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

        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        if altitude < LAND_HEIGHT:
            print("Landing")
            await drone.offboard.stop()
            await drone.action.land()
            return

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
    # drone = System(mavsdk_server_address="127.0.0.1", port=50051)
    # await drone.connect(system_address="udp://:14580")

    print("Connected")
    await wait_for_connection(drone)
    await wait_for_global_position(drone)

    # 🔴 SAVE HOME POSITION
    async for home in drone.telemetry.home():
        home_lat = home.latitude_deg
        home_lon = home.longitude_deg
        print("HOME SAVED:", home_lat, home_lon)
        break

    await wait_for_mission_complete(drone)

    print("Trigger RTL")
    await drone.action.return_to_launch()

    tag_seen_time = None
    takeover_started = False
    while True:

        # read position
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            lat = pos.latitude_deg
            lon = pos.longitude_deg
            break

        dist_home = distance_m(lat, lon, home_lat, home_lon)

        # read tag
        try:
            data, _ = sock.recvfrom(1024)
            found, _, _, _ = struct.unpack("ffff", data)
        except BlockingIOError:
            found = 0.0

        # 🔴 TAKEOVER ONLY WHEN ACTUALLY HOME
        if (
            not takeover_started
            and found
            and altitude < RTL_CAPTURE_HEIGHT
            and dist_home < HOME_RADIUS
        ):

            takeover_started = True

            print("Tag detected at altitude", altitude)
            print("Stopping RTL BEFORE descent...")

            print("Entering OFFBOARD takeover")

            # send ONE neutral setpoint immediately
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0,0,0,0)
            )

            await asyncio.sleep(0.2)

            try:
                await drone.offboard.start()
                print("OFFBOARD started")
            except OffboardError as e:
                print("OFFBOARD failed:", e)
                return

            # go straight to landing loop
            await precision_land_old(drone)
            return

        else:
            tag_seen_time = None

        await asyncio.sleep(0.2)


asyncio.run(run())