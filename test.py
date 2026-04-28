import asyncio
import math
import time
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
from mavsdk.offboard import Attitude

# -----------------------------
# CONFIG
# -----------------------------
TEST_HZ = 50            # change: 10 / 20 / 50
TEST_MODE = "sin"       # "sin" or "step"
AMPLITUDE = 0.2         # m/s (keep small)
DURATION = 30           # seconds

# smoothing (important for realism)
USE_SMOOTHING = True
ALPHA = 0.2

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

    # Prime offboard (required)
    print("Priming offboard...")
    for _ in range(20):
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0, 0, 0, 0)
        )
        await asyncio.sleep(0.02)
    await drone.action.arm()

    await drone.offboard.start()
    print("Offboard started")

    dt_target = 1.0 / TEST_HZ
    prev_time = time.time()

    # previous values for smoothing
    prev_vx, prev_vy, prev_vz = 0.0, 0.0, 0.0

    start_time = time.time()
    print(f"Running {TEST_MODE} test at {TEST_HZ} Hz")

    while time.time() - start_time < DURATION:
        loop_start = time.time()

        # measure actual dt
        now = time.time()
        dt = now - prev_time
        prev_time = now

        t = now - start_time

        # -----------------------------
        # INPUT SIGNAL
        # -----------------------------
        # if TEST_MODE == "sin":
        #     vx = AMPLITUDE * math.sin(2 * math.pi * 0.5 * t)
        # elif TEST_MODE == "step":
        #     vx = AMPLITUDE if int(t) % 2 == 0 else -AMPLITUDE
        # else:
        #     vx = 0.0

        vy = 0.0
        vx = 0.0
        if TEST_MODE == "sin":
            vz = -0.3 * math.sin(2 * math.pi * 0.5 * t)   # negative = upward in NED
        elif TEST_MODE == "step":
            vz = -0.3 if int(t) % 2 == 0 else 0.3
        else:
            vz = 0.0
        yaw_rate = 0.0

        # -----------------------------
        # SMOOTHING (important)
        # -----------------------------
        if USE_SMOOTHING:
            vx = ALPHA * vx + (1 - ALPHA) * prev_vx
            vy = ALPHA * vy + (1 - ALPHA) * prev_vy
            vz = ALPHA * vz + (1 - ALPHA) * prev_vz

        prev_vx, prev_vy, prev_vz = vx, vy, vz

        # -----------------------------
        # SEND COMMAND
        # -----------------------------
        if TEST_MODE == "sin":
            thrust = 0.5 + 0.1 * math.sin(2 * math.pi * 0.5 * t)
        elif TEST_MODE == "step":
            thrust = 0.6 if int(t) % 2 == 0 else 0.4
        else:
            thrust = 0.5

        await drone.offboard.set_attitude(
            Attitude(0.0, 0.0, 0.0, thrust)
        )

        # print occasionally
        if int(t * 10) % 10 == 0:
            print(f"dt: {dt:.3f}s (~{1/dt:.1f} Hz), vx: {vx:.2f}")

        # -----------------------------
        # RATE CONTROL
        # -----------------------------
        elapsed = time.time() - loop_start
        if elapsed < dt_target:
            await asyncio.sleep(dt_target - elapsed)

    print("Stopping...")
    await drone.offboard.stop()


if __name__ == "__main__":
    asyncio.run(run())