import cv2
import numpy as np
import glob

CHECKERBOARD = (9, 6)        # inner corners
SQUARE_SIZE = 0.135          # meters (match your print)

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE

objpoints, imgpoints = [], []

images = sorted(glob.glob("images/calib_*.jpg"))

for f in images:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ok, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                            cv2.CALIB_CB_ADAPTIVE_THRESH +
                                            cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        cv2.cornerSubPix(
            gray, corners, (5,5), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        objpoints.append(objp)
        imgpoints.append(corners)

print("Used images:", len(objpoints))

K = np.zeros((3,3))
D = np.zeros((4,1))
rvecs, tvecs = [], []

rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpoints, gray.shape[::-1],
    K, D, rvecs, tvecs,
    flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-7)
)

print("RMS:", rms)
print("K:\n", K)
print("D:\n", D)

np.savetxt("K_fisheye.txt", K, delimiter=",")
np.savetxt("D_fisheye.txt", D, delimiter=",")