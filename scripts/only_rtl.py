import asyncio
from mavsdk import System


async def run():

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14550")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    print("Connected")

    print("Monitoring mission progress...")

    async for progress in drone.mission.mission_progress():

        print(f"Mission progress: {progress.current}/{progress.total}")

        # Mission finished
        if progress.total != 0 and progress.current == progress.total:
            print("Mission complete → triggering RTL")

            await drone.action.return_to_launch()

            print("RTL activated")
            break


if __name__ == "__main__":
    asyncio.run(run())