import socket
import struct
import math
import time

# -----------------------------
# UDP INPUT
# -----------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# -----------------------------
# MODIFIED CONTROL PARAMETERS
# -----------------------------
KP_MOVE = 0.4                 # reduced gain
ABS_MAX_SPEED = 0.15          # global safety cap
DESCENT_RATE = 0.15           # slower descent
ANGLE_DESCEND = 0.20          # ~11.5 degrees
DEADBAND = 0.02               # 2 cm deadband

print("STABLE SIM controller running...")

while True:

    try:
        data, _ = sock.recvfrom(1024)
        found, x_cam, y_cam, z_cam = struct.unpack("ffff", data)

        # convert cm → meters
        x_cam /= 100.0
        y_cam /= 100.0
        z_cam /= 100.0

    except BlockingIOError:
        time.sleep(0.05)
        continue

    if found < 0.5:
        print("NO TAG → vx=0 vy=0 vz=0")
        continue

    # -----------------------------
    # Camera → BODY frame
    # -----------------------------
    x_body = -y_cam
    y_body =  x_cam

    # -----------------------------
    # Deadband
    # -----------------------------
    if abs(x_body) < DEADBAND:
        x_body = 0.0
    if abs(y_body) < DEADBAND:
        y_body = 0.0

    # -----------------------------
    # Angle calculation
    # -----------------------------
    angle_x = math.atan2(x_body, z_cam)
    angle_y = math.atan2(y_body, z_cam)
    angle_total = math.sqrt(angle_x**2 + angle_y**2)

    # -----------------------------
    # Proportional control
    # -----------------------------
    vx = KP_MOVE * x_body
    vy = KP_MOVE * y_body

    # -----------------------------
    # Height-scaled clamp
    # -----------------------------
    adaptive_max = 0.03 + 0.02 * z_cam
    max_allowed = min(adaptive_max, ABS_MAX_SPEED)

    vx = max(min(vx, max_allowed), -max_allowed)
    vy = max(min(vy, max_allowed), -max_allowed)

    # -----------------------------
    # Descent gating
    # -----------------------------
    vz = DESCENT_RATE if angle_total <= ANGLE_DESCEND else 0.0

    angle_deg = math.degrees(angle_total)

    print(
        f"x={x_body:.2f}m y={y_body:.2f}m z={z_cam:.2f}m | "
        f"angle={angle_deg:.1f}° | "
        f"vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}"
    )

    time.sleep(0.1)
