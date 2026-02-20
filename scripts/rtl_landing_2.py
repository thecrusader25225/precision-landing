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

prev_pose = None
pose_valid = False

# tuning values
MAX_Z_JUMP = 1.0      # meters per frame
MAX_XY_JUMP = 0.8     # meters per frame
MAX_RATE = 2.0        # m/s allowed apparent tag motion

SMOOTH = 0.35         # exponential smoothing factor

BOX_LIMIT = 0.5   # 1m box
BOX_GAIN  = 0.6   # wall push strength
BOX_MAX   = 0.35  # cap boundary speed

ANGLE_DESCEND = math.radians(18)
LAND_HEIGHT = 0.45
DESCENT_RATE = 0.25
KP_MOVE = 0.2
MAX_SPEED = 0.4

RTL_CAPTURE_HEIGHT = 8.0   # take over only when below this
TAG_TIMEOUT = 2.0          # seconds tag must be stable

home_lat = None
home_lon = None

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
    async for state in drone.core.connection_state():
        if state.is_connected:
            break

async def safe_set_velocity(drone, vx, vy, vz, yaw=0.0):
    try:
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(vx, vy, vz, yaw)
        )
        return True
    except Exception as e:
        print("⚠ MAVSDK connection lost:", e)
        return False

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

        if not found:
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

# async def precision_land_exact(drone):
#     while True:

#         # -----------------------------
#         # READ VISION
#         # -----------------------------
#         try:
#             data,_ = sock.recvfrom(1024)
#             found,x_cam,y_cam,z_cam = struct.unpack("ffff",data)
#             x_cam /= 100.0
#             y_cam /= 100.0
#             z_cam /= 100.0

#         except BlockingIOError:
#             await asyncio.sleep(0.02)
#             continue

#         if not found:
#             print("Tag lost → hover")
#             await drone.offboard.set_velocity_ned(
#                 VelocityNedYaw(0,0,0,0)
#             )
#             await asyncio.sleep(0.05)
#             continue


#         # -----------------------------
#         # CAMERA → UAV FRAME
#         # (same as DroneKit camera_to_uav)
#         # -----------------------------
#         x_uav = -y_cam     # forward
#         y_uav =  x_cam     # right


#         # -----------------------------
#         # UAV → NED using yaw
#         # (same as uav_to_ne)
#         # -----------------------------
#         async for att in drone.telemetry.attitude_euler():
#             yaw = math.radians(att.yaw_deg)
#             break

#         c = math.cos(yaw)
#         s = math.sin(yaw)

#         north = x_uav*c - y_uav*s
#         east  = x_uav*s + y_uav*c


#         # -----------------------------
#         # ANGLE COMPUTATION
#         # (same as marker_position_to_angle)
#         # -----------------------------
#         angle_x = math.atan2(x_uav, z_cam)
#         angle_y = math.atan2(y_uav, z_cam)

#         angle_total = math.sqrt(angle_x**2 + angle_y**2)


#         # -----------------------------
#         # HORIZONTAL MOVEMENT
#         # (DroneKit simple_goto equivalent)
#         # -----------------------------
#         vx = KP_MOVE * north
#         vy = KP_MOVE * east

#         vx = max(min(vx, MAX_SPEED), -MAX_SPEED)
#         vy = max(min(vy, MAX_SPEED), -MAX_SPEED)


#         # -----------------------------
#         # DESCENT CONDITION
#         # (same logic as DroneKit)
#         # -----------------------------
#         if angle_total <= ANGLE_DESCEND:
#             vz = DESCENT_RATE
#             print("Low error → descending")
#         else:
#             vz = 0.0
#             print("Correcting position")


#         # -----------------------------
#         # LAND TRIGGER
#         # -----------------------------
#         async for pos in drone.telemetry.position():
#             altitude = pos.relative_altitude_m
#             break

#         if altitude < LAND_HEIGHT:
#             print("Switching to LAND")
#             await drone.offboard.stop()
#             await drone.action.land()
#             return


#         # -----------------------------
#         # DEBUG
#         # -----------------------------
#         print(
#             f"x={x_cam:.2f} y={y_cam:.2f} z={z_cam:.2f} "
#             f"| N={north:.2f} E={east:.2f} "
#             f"| angle={math.degrees(angle_total):.1f}"
#         )


#         # -----------------------------
#         # SEND COMMAND
#         # -----------------------------
#         await drone.offboard.set_velocity_ned(
#             VelocityNedYaw(vx, vy, vz, 0)
#         )
#         print("DESCENT CHECK:",
#         math.degrees(angle_total),
#         "<", math.degrees(ANGLE_DESCEND))


#         await asyncio.sleep(0.03)

# async def precision_land_exact(drone):

#     global prev_pose

#     while True:

#         # -----------------------------
#         # READ VISION
#         # -----------------------------
#         try:
#             data,_ = sock.recvfrom(1024)
#             found,x_cam,y_cam,z_cam = struct.unpack("ffff",data)
#             x_cam /= 100.0
#             y_cam /= 100.0
#             z_cam /= 100.0

#         except BlockingIOError:
#             await asyncio.sleep(0.02)
#             continue

#         # -----------------------------
#         # TAG LOST
#         # -----------------------------
#         if not found:
#             prev_pose = None
#             print("Tag lost → hover")
#             await drone.offboard.set_velocity_ned(
#                 VelocityNedYaw(0,0,0,0)
#             )
#             await asyncio.sleep(0.05)
#             continue


#         # -----------------------------
#         # 🔥 POSE STABILIZER (NEW)
#         # -----------------------------
#         x = x_cam
#         y = y_cam
#         z = z_cam

#         now = asyncio.get_event_loop().time()

#         if prev_pose is not None:

#             px, py, pz, pt = prev_pose
#             dt = max(0.001, now - pt)

#             # reject impossible jumps
#             if abs(z - pz) > MAX_Z_JUMP:
#                 print("REJECT z jump")
#                 continue

#             if abs(x - px) > MAX_XY_JUMP or abs(y - py) > MAX_XY_JUMP:
#                 print("REJECT xy jump")
#                 continue

#             # reject impossible speed
#             vx_rate = abs(x - px)/dt
#             vy_rate = abs(y - py)/dt
#             vz_rate = abs(z - pz)/dt

#             if vx_rate > MAX_RATE or vy_rate > MAX_RATE or vz_rate > MAX_RATE:
#                 print("REJECT speed spike")
#                 continue

#             # smoothing
#             x = px + SMOOTH*(x - px)
#             y = py + SMOOTH*(y - py)
#             z = pz + SMOOTH*(z - pz)

#         # store pose
#         prev_pose = (x, y, z, now)

#         # replace values
#         x_cam, y_cam, z_cam = x, y, z


#         # -----------------------------
#         # CAMERA → UAV FRAME
#         # -----------------------------
#         x_uav = -y_cam
#         y_uav =  x_cam


#         # -----------------------------
#         # UAV → NED using yaw
#         # -----------------------------
#         async for att in drone.telemetry.attitude_euler():
#             yaw = math.radians(att.yaw_deg)
#             break

#         c = math.cos(yaw)
#         s = math.sin(yaw)

#         north = x_uav*c - y_uav*s
#         east  = x_uav*s + y_uav*c


#         # -----------------------------
#         # ANGLE COMPUTATION
#         # -----------------------------
#         angle_x = math.atan2(x_uav, z_cam)
#         angle_y = math.atan2(y_uav, z_cam)
#         angle_total = math.sqrt(angle_x**2 + angle_y**2)


#         # -----------------------------
#         # HORIZONTAL MOVEMENT
#         # -----------------------------
#         vx = KP_MOVE * north
#         vy = KP_MOVE * east

#         vx = max(min(vx, MAX_SPEED), -MAX_SPEED)
#         vy = max(min(vy, MAX_SPEED), -MAX_SPEED)


#         # -----------------------------
#         # DESCENT CONDITION
#         # -----------------------------
#         if angle_total <= ANGLE_DESCEND:
#             vz = DESCENT_RATE
#             print("Low error → descending")
#         else:
#             vz = 0.0
#             print("Correcting position")


#         # -----------------------------
#         # LAND TRIGGER
#         # -----------------------------
#         async for pos in drone.telemetry.position():
#             altitude = pos.relative_altitude_m
#             break

#         if altitude < LAND_HEIGHT:
#             print("Switching to LAND")
#             await drone.offboard.stop()
#             await drone.action.land()
#             return


#         # -----------------------------
#         # DEBUG
#         # -----------------------------
#         print(
#             f"x={x_cam:.2f} y={y_cam:.2f} z={z_cam:.2f} "
#             f"| N={north:.2f} E={east:.2f} "
#             f"| angle={math.degrees(angle_total):.1f}"
#         )


#         # -----------------------------
#         # SEND COMMAND
#         # -----------------------------
#         await drone.offboard.set_velocity_ned(
#             VelocityNedYaw(vx, vy, vz, 0)
#         )

#         print(
#             "DESCENT CHECK:",
#             math.degrees(angle_total),
#             "<",
#             math.degrees(ANGLE_DESCEND)
#         )

#         await asyncio.sleep(0.03)

async def precision_land_exact(drone):

    global prev_pose

    landing_origin = None   # 🔴 center of virtual box

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
            horizontal_error = math.sqrt(x_cam**2 + y_cam**2)
            print(f"RAW camera: x={x_cam:.2f} y={y_cam:.2f}")
        except BlockingIOError:
            await asyncio.sleep(0.02)
            continue

        # -----------------------------
        # TAG LOST
        # -----------------------------
        if not found:
            prev_pose = None
            print("Tag lost → hover")
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(0,0,0,0)
            )
            await asyncio.sleep(0.05)
            continue


        # -----------------------------
        # 🔥 POSE STABILIZER
        # -----------------------------
        x = x_cam
        y = y_cam
        z = z_cam

        now = asyncio.get_event_loop().time()

        if prev_pose is not None:

            px, py, pz, pt = prev_pose
            dt = max(0.001, now - pt)

            if abs(z - pz) > z_jump_limit:
                print("REJECT z jump")
                continue

            if abs(x - px) > xy_jump_limit or abs(y - py) > xy_jump_limit:
                print("REJECT xy jump")
                continue

            vx_rate = abs(x - px)/dt
            vy_rate = abs(y - py)/dt
            vz_rate = abs(z - pz)/dt

            if vx_rate > rate_limit or vy_rate > rate_limit or vz_rate > rate_limit:
                print("REJECT speed spike")
                continue

            # smoothing
            x = px + SMOOTH*(x - px)
            y = py + SMOOTH*(y - py)
            z = pz + SMOOTH*(z - pz)

        prev_pose = (x, y, z, now)
        x_cam, y_cam, z_cam = x, y, z


        # -----------------------------
        # LOCK LANDING BOX CENTER
        # -----------------------------
        if landing_origin is None:
            landing_origin = (x_cam, y_cam)
            print("Landing box center locked:", landing_origin)

        ox, oy = landing_origin
        dx = x_cam - ox
        dy = y_cam - oy


        # -----------------------------
        # CAMERA → UAV FRAME
        # -----------------------------
        x_uav = y_cam
        y_uav = -x_cam


        # -----------------------------
        # UAV → NED using yaw
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
        # -----------------------------
        angle_x = math.atan2(x_uav, z_cam)
        angle_y = math.atan2(y_uav, z_cam)
        angle_total = math.sqrt(angle_x**2 + angle_y**2)


        # -----------------------------
        # VISION VELOCITY
        # -----------------------------
        vx_vis = -KP_MOVE * north
        vy_vis = -KP_MOVE * east


        # -----------------------------
        # 🧱 VIRTUAL LANDING BOX
        # -----------------------------
        vx_box = 0.0
        vy_box = 0.0

        if dx > BOX_LIMIT:
            vx_box = -BOX_GAIN * (dx - BOX_LIMIT)
        elif dx < -BOX_LIMIT:
            vx_box = -BOX_GAIN * (dx + BOX_LIMIT)

        if dy > BOX_LIMIT:
            vy_box = -BOX_GAIN * (dy - BOX_LIMIT)
        elif dy < -BOX_LIMIT:
            vy_box = -BOX_GAIN * (dy + BOX_LIMIT)

        vx_box = max(min(vx_box, BOX_MAX), -BOX_MAX)
        vy_box = max(min(vy_box, BOX_MAX), -BOX_MAX)


        # -----------------------------
        # COMBINE CONTROLLERS
        # -----------------------------
        vx = vx_vis + vx_box
        vy = vy_vis + vy_box

        vx = max(min(vx, MAX_SPEED), -MAX_SPEED)
        vy = max(min(vy, MAX_SPEED), -MAX_SPEED)


        # -----------------------------
        # DESCENT CONDITION
        # -----------------------------
        if angle_total <= ANGLE_DESCEND and horizontal_error < 0.25:
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
            # adaptive pose limits based on altitude
            xy_jump_limit = max(0.25, 0.15 * altitude)
            z_jump_limit  = max(0.25, 0.10 * altitude)
            rate_limit    = max(1.5,  0.8  * altitude)
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

        print(
            f"BOX dx={dx:.2f} dy={dy:.2f} "
            f"| v_vis=({vx_vis:.2f},{vy_vis:.2f}) "
            f"| v_box=({vx_box:.2f},{vy_box:.2f})"
        )


        # -----------------------------
        # SEND COMMAND
        # -----------------------------
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(vx, vy, vz, 0)
        )

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
    print("Saving HOME position...")
    async for home in drone.telemetry.home():
        home_lat = home.latitude_deg
        home_lon = home.longitude_deg
        print("Home saved:", home_lat, home_lon)
        break

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
            lat = pos.latitude_deg
            lon = pos.longitude_deg
            break

        dist_home = distance_m(lat, lon, home_lat, home_lon)


        # read tag
        try:
            data,_ = sock.recvfrom(1024)
            found,_,_,_ = struct.unpack("ffff",data)
        except BlockingIOError:
            found = 0.0

        # tag visible?
        if (found and altitude < RTL_CAPTURE_HEIGHT and dist_home < 2.0):
            if tag_seen_time is None:
                tag_seen_time = asyncio.get_event_loop().time()

            # stable detection?
            if asyncio.get_event_loop().time() - tag_seen_time > TAG_TIMEOUT:
                print("Tag confirmed → TAKEOVER")
                # if abs(x_cam) < 0.5 and abs(y_cam) < 0.5:
                # cancel RTL safely
                print("Cancelling AUTO safely")

                await drone.action.hold()

                # 🔴 CRITICAL: wait until PX4 mode really changes
                # async for mode in drone.telemetry.flight_mode():
                #     print("Current mode:", mode)

                #     if mode != "Mission" and mode != "ReturnToLaunch":
                #         break

                #     await asyncio.sleep(0.2)

                # print("PX4 exited AUTO stack")

                await drone.action.hold()
                print("Requested HOLD — waiting for stabilization")

                # 1️⃣ Wait until PX4 exits AUTO modes
                async for mode in drone.telemetry.flight_mode():
                    print("Current mode:", mode)
                    if mode.name not in ["MISSION", "RETURN_TO_LAUNCH"]:
                        break
                    await asyncio.sleep(0.2)

                print("PX4 exited AUTO stack")

                # 2️⃣ Wait until horizontal speed is small
                print("Waiting for drone to stop moving...")

                stable_time = None

                async for vel in drone.telemetry.velocity_ned():

                    speed = math.sqrt(vel.north_m_s**2 + vel.east_m_s**2)

                    if speed < 0.15:  # nearly stopped
                        if stable_time is None:
                            stable_time = asyncio.get_event_loop().time()

                        # must stay slow for 1 second
                        if asyncio.get_event_loop().time() - stable_time > 1.0:
                            break
                    else:
                        stable_time = None

                    await asyncio.sleep(0.1)

                print("Drone velocity stabilized")

                # 3️⃣ Extra hover wait (your requested 3 seconds)
                print("Holding hover for 3 seconds before takeover...")

                start_wait = asyncio.get_event_loop().time()

                while asyncio.get_event_loop().time() - start_wait < 3.0:
                    try:
                        ok = await safe_set_velocity(drone, 0, 0, 0)
                        if not ok:
                            print("Attempting MAVSDK reconnect...")
                            await drone.connect(system_address="udpin://0.0.0.0:14540")
                            await wait_for_connection(drone)
                    except:
                        pass

                    await asyncio.sleep(0.1)

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
