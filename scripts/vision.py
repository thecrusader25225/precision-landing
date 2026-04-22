import socket
import struct
import numpy as np
import cv2
import math
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from opencv.lib_aruco_pose import ArucoSingleTracker

# -----------------------------
# UDP OUTPUT
# -----------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
packet_id = 0 
# -----------------------------
# CAMERA MODEL (unchanged)
# -----------------------------

camera_matrix = np.loadtxt("/home/marg/precision-landing/opencv/cameraMatrix.txt", delimiter=',')
camera_distortion = np.loadtxt("/home/marg/precision-landing/opencv/cameraDistortion.txt", delimiter=',')



aruco = ArucoSingleTracker(
    id_to_find=72,
    marker_size=15.0,
    camera_matrix=camera_matrix,
    camera_distortion=camera_distortion
)

# -----------------------------
# GSTREAMER PIPELINE
# -----------------------------
Gst.init(None)

pipeline = Gst.parse_launch(
    "libcamerasrc ! "
    "video/x-raw,format=NV12,width=3280,height=2464,framerate=15/1 ! "
    "queue ! videoconvert ! video/x-raw,format=BGR ! "
    "appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true"
)

appsink = pipeline.get_by_name("appsink")
pipeline.set_state(Gst.State.PLAYING)

print("Camera pipeline started")

# -----------------------------
# FRAME LOOP
# -----------------------------
while True:

    sample = appsink.emit("pull-sample")
    if sample is None:
        continue

    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)

    w = 3280
    h = 2464

    success, mapinfo = buf.map(Gst.MapFlags.READ)
    if not success:
        continue

    frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
    frame = frame.reshape((h, w, 3))
    buf.unmap(mapinfo)

    # -----------------------------
    # ARUCO TRACKING (UNCHANGED)
    # -----------------------------
    detections = aruco.track(frame)

    tag72 = detections.get(72, None)
    tagX  = detections.get(10, None)  # your orientation tag

    def valid(tag):
        if tag is None:
            return None

        x, y, z = tag

        if z <= 0 or z > 800:
            return None

        max_lateral = z * 5.0
        if abs(x) > max_lateral or abs(y) > max_lateral:
            return None

        return (x, y, z)

    tag72 = valid(tag72)
    tagX  = valid(tagX)
    # -----------------------------
    # SEND UDP
    # -----------------------------
    packet_id += 1

    if tag72:
        f1, x1, y1, z1 = 1.0, *tag72
    else:
        f1, x1, y1, z1 = 0.0, 0.0, 0.0, 0.0

    if tagX:
        f2, x2, y2, z2 = 1.0, *tagX
    else:
        f2, x2, y2, z2 = 0.0, 0.0, 0.0, 0.0

    data = struct.pack(
        "Iffffffff",
        packet_id,
        f1, x1, y1, z1,
        f2, x2, y2, z2
    )
    yaw_error=0
    if f1 > 0.5 and f2 > 0.5:
        dx = (x1 - x2)
        dy = y1 - y2

        # camera forward = -y_cam → mapped to x_body
        yaw_error = math.atan2(dx, -dy)*180/math.pi
        yaw_valid = True
        

    sock.sendto(data, (UDP_IP, UDP_PORT))
    print(f"72: {f1}, {x1} {y1} {z1}\n{yaw_error} deg\nX: {f2}, {x2} {y2} {z2}")
