import socket
import struct
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from opencv.lib_aruco_pose import ArucoSingleTracker
import math
UDP_IP = "127.0.0.1"
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

latest_frame = None


width = 1920
height = 1080
fov = 1.204

fx = width / (2 * math.tan(fov/2))
fy = fx
cx = width / 2
cy = height / 2

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

camera_distortion = np.zeros((5,1))

# camera_matrix = np.loadtxt("opencv/cameraMatrix.txt", delimiter=",")
# camera_distortion = np.loadtxt("opencv/cameraDistortion.txt", delimiter=",")

aruco = ArucoSingleTracker(
    id_to_find=72,
    marker_size=10.0,
    camera_matrix=camera_matrix,
    camera_distortion=camera_distortion
)

def callback(msg):
    width = msg.width
    height = msg.height

    img = np.frombuffer(msg.data, dtype=np.uint8)
    img = img.reshape((height, width, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    found, x, y, z = aruco.track(img)

    # --- SANITY FILTERS ---
    if found:

        # 1️⃣ Reject impossible depth
        if z < 15 or z > 800.0:
            found = False

        # 2️⃣ Reject huge lateral jumps (tracker hallucination)
        # tag must lie inside camera cone
        max_lateral = z * 1.2
        if abs(x) > max_lateral or abs(y) > max_lateral:
            found = False

        # 3️⃣ Reject pose that implies tag behind camera
        if z <= 0:
            found = False

    if found:
        data = struct.pack("ffff", 1.0, x, y, z)
    else:
        data = struct.pack("ffff", 0.0, 0.0, 0.0, 0.0)

    sock.sendto(data, (UDP_IP, UDP_PORT))
    print("TAG:", found, x, y, z)



node = Node()
topic = "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
node.subscribe(Image, topic, callback)

print("Vision node running...")

import time
while True:
    time.sleep(1)
