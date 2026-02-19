import collections
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import time
import math
import argparse
import socket
import struct

from dronekit import connect, VehicleMode, LocationGlobalRelative

# ---------------- UDP VISION INPUT ----------------
UDP_IP = "127.0.0.1"
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.05)

def get_marker_from_udp():
    try:
        data, _ = sock.recvfrom(1024)
        found, x, y, z = struct.unpack("ffff", data)
        if found == 1.0:
            return True, x*100.0, y*100.0, z*100.0  # meters → cm
        else:
            return False, 0, 0, 0
    except socket.timeout:
        return False, 0, 0, 0

# ---------------- UTILITY FUNCTIONS ----------------

def get_location_metres(original_location, dNorth, dEast):
    earth_radius = 6378137.0
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    return(newlat, newlon)

def marker_position_to_angle(x, y, z):
    angle_x = math.atan2(x, z)
    angle_y = math.atan2(y, z)
    return angle_x, angle_y

def camera_to_uav(x_cam, y_cam):
    x_uav = -y_cam
    y_uav = x_cam
    return x_uav, y_uav

def uav_to_ne(x_uav, y_uav, yaw_rad):
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    north = x_uav*c - y_uav*s
    east  = x_uav*s + y_uav*c
    return north, east

def check_angle_descend(angle_x, angle_y, angle_desc):
    return math.sqrt(angle_x**2 + angle_y**2) <= angle_desc

# ---------------- CONNECT VEHICLE ----------------

parser = argparse.ArgumentParser()
parser.add_argument('--connect', default='127.0.0.1:14550')
args = parser.parse_args()

print("Connecting...")
vehicle = connect(args.connect, wait_ready=True)

print("Setting GUIDED mode")
vehicle.mode = VehicleMode("GUIDED")
while vehicle.mode.name != "GUIDED":
    time.sleep(0.5)

print("Vehicle ready")

# ---------------- PARAMETERS ----------------

rad_2_deg = 180.0/math.pi
deg_2_rad = 1.0/rad_2_deg

freq_send = 5
angle_descend = 30 * deg_2_rad
land_alt_cm = 50

descending = False
time_0 = time.time()

# ---------------- MAIN LOOP ----------------

while True:

    marker_found, x_cm, y_cm, z_cm = get_marker_from_udp()

    if not marker_found:
        continue

    x_cm, y_cm = camera_to_uav(x_cm, y_cm)
    uav_location = vehicle.location.global_relative_frame

    # use baro at higher altitude
    if uav_location.alt >= 5.0:
        z_cm = uav_location.alt * 100.0

    angle_x, angle_y = marker_position_to_angle(x_cm, y_cm, z_cm)

    if time.time() >= time_0 + 1.0/freq_send:
        time_0 = time.time()

        print("\nAltitude = %.0f cm" % z_cm)
        print("Marker x=%4.0f y=%4.0f -> ax=%.1f ay=%.1f deg"
              % (x_cm, y_cm, angle_x*rad_2_deg, angle_y*rad_2_deg))

        north, east = uav_to_ne(x_cm, y_cm, vehicle.attitude.yaw)
        print("Marker N=%4.0f E=%4.0f" % (north, east))

        marker_lat, marker_lon = get_location_metres(
            uav_location, north*0.01, east*0.01)

        # ---------- DESCENT LOGIC ----------
        if check_angle_descend(angle_x, angle_y, angle_descend):
            if not descending:
                print("LOW ERROR → START DESCENT")
                descending = True

        if descending:
            descend_step = 0.3  # meters per update

            location_marker = LocationGlobalRelative(
                uav_location.lat,
                uav_location.lon,
                uav_location.alt - descend_step
            )
        else:
            location_marker = LocationGlobalRelative(
                marker_lat,
                marker_lon,
                uav_location.alt
            )

        vehicle.simple_goto(location_marker)

        print("Commanding → Lat %.7f Lon %.7f"
              % (location_marker.lat, location_marker.lon))

        # ---------- FINAL LAND ----------
        if z_cm <= land_alt_cm:
            print("FINAL LAND TRIGGERED")
            vehicle.mode = VehicleMode("LAND")
            break
