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

async def run():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected")
            break

    offboard_active = False

    # Control gains
    Kp_xy = 0.8
    max_xy_vel = 1.0
    descend_rate = 0.25
    center_threshold = 0.05
    landing_threshold = 0.18

    print("Precision landing monitor running...")

    while True:

        try:
            data, _ = sock.recvfrom(1024)
            found, x, y, z = struct.unpack("ffff", data)

        except BlockingIOError:
            await asyncio.sleep(0.02)
            continue

        if found > 0.5:

            # FIRST TIME seeing tag → take control
            if not offboard_active:
                print("Tag detected → taking control")

                # Send initial setpoint
                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(0, 0, 0, 0)
                )

                await asyncio.sleep(0.2)

                try:
                    await drone.offboard.start()
                    offboard_active = True
                    print("OFFBOARD started")
                except OffboardError as e:
                    print("Offboard start failed:", e)
                    continue

            # Coordinate transform (camera down-facing)
            x_uav = -y
            y_uav = x

            vx = Kp_xy * x_uav
            vy = Kp_xy * y_uav

            # Clamp velocities
            vx = max(min(vx, max_xy_vel), -max_xy_vel)
            vy = max(min(vy, max_xy_vel), -max_xy_vel)

            # Descend only when centered
            if abs(x) < center_threshold and abs(y) < center_threshold:
                vz = descend_rate
            else:
                vz = 0.0

            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(vx, vy, vz, 0.0)
            )

            print(f"x={x:.2f} y={y:.2f} z={z:.2f}")

            # Land when close
            if z < landing_threshold:
                print("Landing")
                await drone.offboard.stop()
                await drone.action.land()
                break

        else:
            # No tag detected
            if offboard_active:
                # If we already took control but lost tag,
                # hover instead of releasing control
                await drone.offboard.set_velocity_ned(
                    VelocityNedYaw(0, 0, 0, 0)
                )

        await asyncio.sleep(0.03)

asyncio.run(run())
