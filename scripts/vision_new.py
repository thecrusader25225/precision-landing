import socket
import struct
import numpy as np
import cv2
import math
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import sys
from opencv.lib_aruco_pose import ArucoSingleTracker
import time

# -----------------------------
# UDP OUTPUT
# -----------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
packet_id = 0 
record_suffix = None
output_file = None
if len(sys.argv) > 1:
    record_suffix = sys.argv[1]

if record_suffix:
    output_file = f"vid{record_suffix}.mkv"
# -----------------------------
# CAMERA MODEL (unchanged)
# -----------------------------

camera_matrix = np.loadtxt("/home/marg/precision-landing/opencv/cameraMatrix.txt", delimiter=',')
camera_distortion = np.loadtxt("/home/marg/precision-landing/opencv/cameraDistortion.txt", delimiter=',')
res_x = 640
res_y = 360
# scaling
calib_width = 3264
calib_height = 2448

scale_x = res_x / calib_width
scale_y = res_y / calib_height

camera_matrix[0, 0] *= scale_x
camera_matrix[1, 1] *= scale_y
camera_matrix[0, 2] *= scale_x
camera_matrix[1, 2] *= scale_y

# now scale AGAIN for detection resolution
scale_x_det =  640/ res_x
scale_y_det = 360/ res_y

camera_matrix_small = camera_matrix.copy()
camera_matrix_small[0, 0] *= scale_x_det
camera_matrix_small[1, 1] *= scale_y_det
camera_matrix_small[0, 2] *= scale_x_det
camera_matrix_small[1, 2] *= scale_y_det

# --- UNDISTORT SETUP (pinhole model) ---
DIM = (res_x, res_y)

K = camera_matrix_small
D = camera_distortion

# alpha: 0 = crop (best accuracy), 1 = keep full FOV (more black borders)
alpha = 0.5

new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, DIM, alpha, DIM)

map1, map2 = cv2.initUndistortRectifyMap(
    K, D, None, new_K, DIM, cv2.CV_16SC2
)

def rescale(tag):
    if tag is None:
        return None
    x, y, z = tag
    return (x * scale_x_det, y * scale_y_det, z)

aruco = ArucoSingleTracker(
    id_to_find=72,
    marker_size=15.0,
    camera_matrix=camera_matrix,
    camera_distortion=camera_distortion
)
aruco_small = ArucoSingleTracker(
    id_to_find=72,
    marker_size=15.0,
    camera_matrix=new_K,          # ← updated
    camera_distortion=np.zeros_like(D)  # ← effectively no distortion after remap
)
# -----------------------------
# GSTREAMER PIPELINE
# -----------------------------
Gst.init(None)

pipeline_str = (
"libcamerasrc ! "
"video/x-raw,format=NV12,width=3280,height=2464,framerate=15/1 ! "

#--------downscale------------
"videoconvert ! videoscale ! "
"video/x-raw,width=640,height=360 ! "

"tee name=t "
#--------vision-------------
"t. ! queue ! videoconvert ! video/x-raw,format=GRAY8 ! "
"appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true "
#--------remote stream---------------
    #"t. ! queue ! videoconvert ! video/x-raw,format=GRAY8 ! "
    #"x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast ! "
    #"h264parse ! flvmux streamable=true ! "
    #"rtmpsink location=\"rtmp://100.78.97.114:1935/stream\" "
)

if output_file:
    pipeline_str += (
        "t. ! queue ! videoconvert ! videoscale ! "
        "video/x-raw,width=960,height=540 ! "
        "x264enc tune=zerolatency bitrate=8000 speed-preset=ultrafast ! "
        f"matroskamux ! filesink location={output_file}"
    )

pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("appsink")
pipeline.set_state(Gst.State.PLAYING)

print("Camera pipeline started")
saved = False


# -----------------------------
# FRAME LOOP
# -----------------------------
TARGET_FPS = 8
FRAME_TIME = 1 / TARGET_FPS
last_process_time = 0

# --- VIDEO RECORDER (undistorted) ---
fourcc = cv2.VideoWriter_fourcc(*'X264')  # or 'MJPG' if X264 fails
out = cv2.VideoWriter(
    f"undist_{record_suffix}.mkv" if record_suffix else "undist.mkv",
    fourcc,
    TARGET_FPS,
    (res_x, res_y),
    False   # grayscale
)
while True:
    loop_start = time.time()
    sample = appsink.emit("pull-sample")
    if sample is None:
        continue
    now = time.time()
    if now - last_process_time < FRAME_TIME:
        continue   # <-- HARD DROP 
    last_process_time = now
    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)

    w = res_x
    h = res_y

    success, mapinfo = buf.map(Gst.MapFlags.READ)
    if not success:
        continue

    frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
    frame = frame.reshape((h, w))
    frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    buf.unmap(mapinfo)
    out.write(frame)

    # -----------------------------
    # ARUCO TRACKING (UNCHANGED)
    # -----------------------------
    # small = cv2.resize(frame, (960, 540))
    detections = aruco_small.track(frame)
    #detections = aruco.track(frame)

    tag72 = detections.get(72, None)
    tagX  = detections.get(10, None)  # your orientation tag

    #tag72 = rescale(detections.get(72, None))
    #tagX  = rescale(detections.get(10, None))
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
    #if not saved:
     #   cv2.imwrite("/home/marg/debug_frame.png", frame)
      #  print("Saved debug frame")
       # saved = True
    sock.sendto(data, (UDP_IP, UDP_PORT))
    print(f"72: {f1}, {x1} {y1} {z1}\n{yaw_error} deg\nX: {f2}, {x2} {y2} {z2}")
    elapsed = time.time() - loop_start
    if elapsed < FRAME_TIME:
        time.sleep(FRAME_TIME - elapsed)
out.release()