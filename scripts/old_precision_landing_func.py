async def precision_land(drone):

    print("Starting precision landing")

    while True:
        latest = None

        # -----------------------------
        # Get latest UDP packet
        # -----------------------------
        while True:
            try:
                data, _ = sock.recvfrom(1024)
                latest = data
            except BlockingIOError:
                break

        if latest is None:
            await asyncio.sleep(0.02)
            continue

        packet_id, found, x_cam, y_cam, z_cam = struct.unpack("Iffff", latest)

        print(f"TAG: ", packet_id, found, x_cam, y_cam, z_cam)

        # -----------------------------
        # Convert cm → meters
        # -----------------------------
        x_cam /= 100.0
        y_cam /= 100.0
        z_cam /= 100.0

        # -----------------------------
        # If tag not found → hover
        # -----------------------------
        if found < 0.5:
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
            print("NO TAG")
            await asyncio.sleep(0.1)
            continue

        # -----------------------------
        # Camera → BODY frame
        # -----------------------------
        x_body = -y_cam
        y_body = x_cam

        # -----------------------------
        # Deadband
        # -----------------------------
        if abs(x_body) < DEADBAND:
            x_body = 0.0
        if abs(y_body) < DEADBAND:
            y_body = 0.0

        # -----------------------------
        # Angle check
        # -----------------------------
        angle_x = math.atan2(x_body, z_cam)
        angle_y = math.atan2(y_body, z_cam)
        angle_total = math.sqrt(angle_x**2 + angle_y**2)

        print(f"Angle: {math.degrees(angle_total):.1f}")

        # -----------------------------
        # Velocity control
        # -----------------------------
        max_vel = min(MAX_SPEED, 0.05 + 0.3 * z_cam)

        kp_dynamic = KP_MOVE * min(1.0, z_cam / 2.0)

        vx = kp_dynamic * x_body
        vy = kp_dynamic * y_body

        # Clamp
        vx = max(min(vx, max_vel), -max_vel)
        vy = max(min(vy, max_vel), -max_vel)

        # -----------------------------
        # Descent logic
        # -----------------------------
        if angle_total <= ANGLE_DESCEND:
            vz = DESCENT_RATE
        else:
            vz = 0.0

        print(f"vx: {vx:.2f}  vy: {vy:.2f}  vz: {vz:.2f}")

        # -----------------------------
        # Check altitude
        # -----------------------------
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        if altitude < LAND_HEIGHT:
            print("Switching to LAND mode")
            await drone.action.land()
            return

        # -----------------------------
        # Send velocity
        # -----------------------------
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(vx, vy, vz, 0.0)
        )

        await asyncio.sleep(0.1)