import asyncio
from mavsdk import System

async def main():
    drone = System()
    await drone.connect(system_address="udp://:14550")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break

    print("Connected.\n")
    print("👉 Now upload mission from MAVProxy and watch outputs...\n")

    # ---------------------------
    # TASK 1: Mission Progress
    # ---------------------------
    async def mission_progress_task():
        async for progress in drone.mission.mission_progress():
            print(f"[MISSION_PROGRESS] {progress.current}/{progress.total}")

    # ---------------------------
    # TASK 2: Mission Finished
    # ---------------------------
    async def mission_finished_task():
        while True:
            try:
                is_finished = await drone.mission.is_mission_finished()
                print(f"[MISSION_FINISHED] {is_finished}")
            except Exception as e:
                print(f"[MISSION_FINISHED ERROR] {e}")
            await asyncio.sleep(1)

    # ---------------------------
    # TASK 3: Flight Mode
    # ---------------------------
    async def flight_mode_task():
        async for mode in drone.telemetry.flight_mode():
            print(f"[FLIGHT_MODE] {mode}")

    # ---------------------------
    # TASK 4: Try downloading mission periodically
    # ---------------------------
    async def mission_download_task():
        while True:
            try:
                mission = await drone.mission.download_mission()
                items = mission.mission_items
                print(f"[MISSION_DOWNLOAD] items={items}")
            except Exception as e:
                print(f"[MISSION_DOWNLOAD ERROR] {e}")
            await asyncio.sleep(5)

    # Run all together
    await asyncio.gather(
        mission_progress_task(),
        mission_finished_task(),
        flight_mode_task(),
        mission_download_task()
    )

if __name__ == "__main__":
    asyncio.run(main())