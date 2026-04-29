import cv2
import numpy as np
import glob

CHECKERBOARD = (9, 6)   # inner corners
SQUARE_SIZE = 0.025     # meters (adjust to your print)

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

images = glob.glob("calib_images/*.jpg")  # put your images here

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

N_OK = len(objpoints)
print("Valid images:", N_OK)

K = np.zeros((3,3))
D = np.zeros((4,1))
rvecs = []
tvecs = []

rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

print("RMS:", rms)
print("K:\n", K)
print("D:\n", D)

np.savetxt("cameraMatrix_fisheye.txt", K, delimiter=',')
np.savetxt("cameraDistortion_fisheye.txt", D, delimiter=',')