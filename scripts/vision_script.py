import socket
import struct
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from opencv.lib_aruco_pose import ArucoSingleTracker

UDP_IP = "127.0.0.1"
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

latest_frame = None

camera_matrix = np.loadtxt("opencv/cameraMatrix.txt", delimiter=",")
camera_distortion = np.loadtxt("opencv/cameraDistortion.txt", delimiter=",")

aruco = ArucoSingleTracker(
    id_to_find=72,
    marker_size=0.1,
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

    if found:
        data = struct.pack("ffff", 1.0, x, y, z)
    else:
        data = struct.pack("ffff", 0.0, 0.0, 0.0, 0.0)

    sock.sendto(data, (UDP_IP, UDP_PORT))


node = Node()
topic = "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
node.subscribe(Image, topic, callback)

print("Vision node running...")

import time
while True:
    time.sleep(1)
