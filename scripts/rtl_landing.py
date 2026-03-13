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
# PARAMETERS
# -----------------------------
ANGLE_DESCEND = math.radians(18)
LAND_HEIGHT = 0.45
DESCENT_RATE = 0.25
KP_MOVE = 0.6
MAX_SPEED = 0.7

RTL_CAPTURE_HEIGHT = 8.0   # take over only when below this
TAG_TIMEOUT = 2.0          # seconds tag must be stable


# -----------------------------
# HELPERS
# -----------------------------
async def wait_for_connection(drone):
    async for state in drone.core.connection_state():
        if state.is_connected:
            break


async def wait_for_global_position(drone):
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            break


async def wait_for_mission_complete(drone):

    print("Waiting for mission mode...")

    mission_seen = False

    async for mode in drone.telemetry.flight_mode():

        # Mission started
        if mode.name == "MISSION":
            mission_seen = True
            print("Mission ongoing")

        # Mission finished when it leaves MISSION mode
        if mission_seen and mode.name != "MISSION":
            print("Mission completed (mode changed to)", mode.name)
            return


# -----------------------------
# PRECISION LANDING LOOP
# -----------------------------
async def precision_land(drone):

    print("Preparing OFFBOARD precision landing")

    # PX4 requires a stream of setpoints before OFFBOARD
    for _ in range(15):
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0,0,0,0)
        )
        await asyncio.sleep(0.05)

    try:
        await drone.offboard.start()
        print("OFFBOARD started successfully")
    except OffboardError as e:
        print("Offboard failed:", e)
        return

    print("Running precision landing loop")

    while True:

        # ---- read vision ----
        try:
            data,_ = sock.recvfrom(1024)
            found,x_cam,y_cam,z_cam = struct.unpack("ffff",data)
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

        # ---- camera → UAV frame ----
        x_uav = -y_cam
        y_uav =  x_cam

        # ---- yaw for body → NED ----
        async for att in drone.telemetry.attitude_euler():
            yaw = math.radians(att.yaw_deg)
            break

        c = math.cos(yaw)
        s = math.sin(yaw)

        north = x_uav*c - y_uav*s
        east  = x_uav*s + y_uav*c

        # ---- angle error ----
        angle_x = math.atan2(x_uav, z_cam)
        angle_y = math.atan2(y_uav, z_cam)
        angle_total = math.sqrt(angle_x**2 + angle_y**2)

        # ---- horizontal control ----
        vx = KP_MOVE * north
        vy = KP_MOVE * east

        vx = max(min(vx, MAX_SPEED), -MAX_SPEED)
        vy = max(min(vy, MAX_SPEED), -MAX_SPEED)

        # ---- descent condition ----
        if angle_total <= ANGLE_DESCEND:
            vz = DESCENT_RATE
            print("Aligned → descending")
        else:
            vz = 0.0
            print("Correcting position")

        # ---- altitude check ----
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        if altitude < LAND_HEIGHT:
            print("Switching to LAND")
            await drone.offboard.stop()
            await drone.action.land()
            return

        # ---- send velocity ----
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(vx, vy, vz, 0)
        )

        await asyncio.sleep(0.03)

async def precision_land_exact(drone):
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


# -----------------------------
# MAIN CONTROLLER
# -----------------------------
async def run():

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Connected")
    await wait_for_connection(drone)
    await wait_for_global_position(drone)

    # -----------------------------
    # WAIT FOR MISSION COMPLETE
    # -----------------------------
    print("Waiting for mission to finish...")
    await wait_for_mission_complete(drone)

    # -----------------------------
    # TRIGGER RTL
    # -----------------------------
    print("Mission done → RTL")
    await drone.action.return_to_launch()

    # -----------------------------
    # MONITOR RTL
    # -----------------------------
    tag_seen_time = None

    while True:

        # check altitude
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        # read tag
        try:
            data,_ = sock.recvfrom(1024)
            found,_,_,_ = struct.unpack("ffff",data)
        except BlockingIOError:
            found = 0.0

        # tag visible?
        if found > 0.5 and altitude < RTL_CAPTURE_HEIGHT:

            if tag_seen_time is None:
                tag_seen_time = asyncio.get_event_loop().time()

            # stable detection?
            if asyncio.get_event_loop().time() - tag_seen_time > TAG_TIMEOUT:
                print("Tag confirmed → TAKEOVER")

                # cancel RTL safely
                print("Cancelling AUTO safely")

                await drone.action.hold()

                # 🔴 CRITICAL: wait until PX4 mode really changes
                async for mode in drone.telemetry.flight_mode():
                    print("Current mode:", mode)

                    if mode != "Mission" and mode != "ReturnToLaunch":
                        break

                    await asyncio.sleep(0.2)

                print("PX4 exited AUTO stack")

                # 🔴 NOW send priming setpoint
                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
                )

                await asyncio.sleep(0.4)

                # 🔴 start offboard only after PX4 ready
                try:
                    await drone.offboard.start()
                    print("OFFBOARD started successfully")
                except OffboardError as e:
                    print("OFFBOARD failed:", e)
                    return

                # 🔴 now run landing loop
                await precision_land_exact(drone)
                return


        else:
            tag_seen_time = None

        await asyncio.sleep(0.2)


# -----------------------------
# ENTRY
# -----------------------------
asyncio.run(run())
