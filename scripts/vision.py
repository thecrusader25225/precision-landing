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
width = 640
height = 480
fov = 1.204

fx = width / (2 * math.tan(fov/2))
fy = fx
cx = width / 2
cy = height / 2

camera_matrix = np.loadtxt("/home/marg/precision-landing/opencv/cameraMatrix.txt", delimiter=',')
camera_distortion = np.loadtxt("/home/marg/precision-landing/opencv/cameraDistortion.txt", delimiter=',')

# --- scale calibration to 8mp resolution ---
calib_width = 3264
calib_height = 2448

scale_x = width / calib_width
scale_y = height / calib_height

camera_matrix[0,0] *= scale_x
camera_matrix[1,1] *= scale_y
camera_matrix[0,2] *= scale_x
camera_matrix[1,2] *= scale_y

aruco = ArucoSingleTracker(
    id_to_find=72,
    marker_size=10.0,
    camera_matrix=camera_matrix,
    camera_distortion=camera_distortion
)

# -----------------------------
# GSTREAMER PIPELINE
# -----------------------------
Gst.init(None)

#pipeline = Gst.parse_launch(
#    "libcamerasrc ! "
#    "video/x-raw,width=640,height=480,framerate=30/1 ! "
#    "videoconvert ! video/x-raw,format=BGR ! "
#    "appsink name=sink emit-signals=false max-buffers=1 drop=true"
#)

pipeline = Gst.parse_launch(
    "libcamerasrc ! "
    "video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! "
    "tee name=t "

    # -------- RTMP branch --------
    #"t. ! queue ! videoconvert ! video/x-raw,format=I420 ! "
    #"x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast ! "
    #"h264parse ! flvmux streamable=true ! "
    #"rtmpsink location=\"rtmp://100.78.97.114:1935/stream\" "

    # -------- Vision branch --------
    "t. ! queue ! videoconvert ! video/x-raw,format=BGR ! "
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

    w = 640
    h = 480

    success, mapinfo = buf.map(Gst.MapFlags.READ)
    if not success:
        continue

    frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
    frame = frame.reshape((h, w, 3))
    buf.unmap(mapinfo)

    # -----------------------------
    # ARUCO TRACKING (UNCHANGED)
    # -----------------------------
    found, x, y, z = aruco.track(frame)

    if found:

        # depth sanity
        if z > 800:
            found = False

        # lateral sanity
        max_lateral = z * 5.0
        if abs(x) > max_lateral or abs(y) > max_lateral:
            found = False

        # behind camera
        if z <= 0:
            found = False

    # -----------------------------
    # SEND UDP
    # -----------------------------
    packet_id +=1
    if found:
        data = struct.pack("Iffff",packet_id, 1.0, x, y, z)
    else:
        data = struct.pack("Iffff",packet_id, 0.0, 0.0, 0.0, 0.0)

    sock.sendto(data, (UDP_IP, UDP_PORT))

    print("TAG:",packet_id, found, x, y, z)
