import numpy as np
import cv2
import cv2.aruco as aruco


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([
        [-marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)

    rvecs, tvecs = [], []

    for c in corners:
        _, rvec, tvec = cv2.solvePnP(
            marker_points,
            c,
            mtx,
            distortion,
            False,
            cv2.SOLVEPNP_ITERATIVE
        )
        rvecs.append(rvec)
        tvecs.append(tvec)

    return rvecs, tvecs, None


class ArucoSingleTracker:
    def __init__(self, id_to_find, marker_size, camera_matrix, camera_distortion):
        self.id_to_find = id_to_find
        self.marker_size = marker_size
        self._camera_matrix = camera_matrix
        self._camera_distortion = camera_distortion

        self._dictionary = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self._parameters = aruco.DetectorParameters()
        self._detector = aruco.ArucoDetector(self._dictionary, self._parameters)

    def track(self, frame, verbose=False):
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        if len(frame.shape) == 2:
            # already grayscale
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self._detector.detectMarkers(gray)

        results = {}

        if ids is not None:

            rvecs, tvecs, _ = my_estimatePoseSingleMarkers(
                corners,
                self.marker_size,
                self._camera_matrix,
                self._camera_distortion
            )

            for i, marker_id in enumerate(ids.flatten()):
                tvec = tvecs[i].flatten()
                x, y, z = tvec

                results[int(marker_id)] = (x, y, z)

                if verbose:
                    print(f"[ID {marker_id}] x={x:.1f} y={y:.1f} z={z:.1f}")

        return results
