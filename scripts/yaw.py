import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed


# -----------------------------
# TEST PARAMETERS
# -----------------------------
TEST_MODE = "circle"      # "yaw" or "circle"
DURATION = 30          # seconds to run test

# Circle params (tune these)
VX = 0.3              # forward speed
YAW_RATE = 20.0        # deg/s
VZ_COMP = 0       # upward compensation (try 0.0 and -0.05)


async def run():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14550")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    print("Connected")

    # Initial neutral setpoint (required)
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    )
    await asyncio.sleep(0.2)

    await drone.offboard.start()
    print("Offboard started")

    print(f"Running {TEST_MODE} test for {DURATION}s")
    start = asyncio.get_event_loop().time()

    while True:
        t = asyncio.get_event_loop().time() - start
        if t > DURATION:
            break

        if TEST_MODE == "yaw":
            # PURE YAW → should NOT drop
            vx = 0.0
            vy = 0.0
            vz = 0.0
            yaw_rate = 25.0

        elif TEST_MODE == "circle":
            # CIRCULAR MOTION → this is what you're debugging
            growth_rate = 0.02
            vx = VX + growth_rate * t
            vy = 0.0
            vz = VZ_COMP
            yaw_rate = YAW_RATE

        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(vx, vy, vz, yaw_rate)
        )

        print(f"cmd → vx:{vx:.2f} vy:{vy:.2f} vz:{vz:.2f} yaw:{yaw_rate}")
        await asyncio.sleep(0.1)

    # STOP
    print("Stopping")
    for _ in range(5):
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        )
        await asyncio.sleep(0.1)

    await drone.offboard.stop()


if __name__ == "__main__":
    asyncio.run(run())
