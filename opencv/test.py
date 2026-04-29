import cv2

img = cv2.imread("pattern_chessboard.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(
    gray,
    (9,6),
    cv2.CALIB_CB_ADAPTIVE_THRESH +
    cv2.CALIB_CB_NORMALIZE_IMAGE
)

print("Detected:", ret)