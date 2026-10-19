"""
Pattern detector and feature extraction module.
Supports checkerboards (with SB sub-pixel engine) and circle grids.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class DetectionResult:
    found: bool
    corners: Optional[np.ndarray] = None  # (N, 1, 2) float32
    sharpness: float = 0.0
    coverage_zone: Tuple[int, int] = (0, 0)
    object_points: Optional[np.ndarray] = None  # (N, 3) float32 for charuco/aprilgrid


def compute_sharpness(image_gray: np.ndarray) -> float:
    """Compute image sharpness using Laplacian variance (higher = sharper)."""
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    return float(laplacian.var())


def generate_object_points(
    rows: int,
    cols: int,
    square_size_mm: float,
    pattern_type: str = "checkerboard"
) -> np.ndarray:
    """
    Generate 3D object coordinates for the calibration target in mm.
    Z is set to 0 (planar assumption).
    """
    obj_pts = np.zeros((rows * cols, 3), np.float32)

    if pattern_type == "asymmetric_circles":
        idx = 0
        for r in range(rows):
            for c in range(cols):
                obj_pts[idx] = [
                    (2 * c + r % 2) * square_size_mm,
                    r * square_size_mm,
                    0.0
                ]
                idx += 1
    else:
        # Standard checkerboard or symmetric circles
        obj_pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    return obj_pts


class PatternDetector:
    def __init__(
        self,
        rows: int = 5,
        cols: int = 8,
        square_size_mm: float = 40.0,
        pattern_type: str = "checkerboard",
        marker_size_mm: Optional[float] = None,
        aruco_dict_name: Optional[str] = None
    ):
        self.rows = rows
        self.cols = cols
        self.square_size_mm = square_size_mm
        self.pattern_type = pattern_type.lower()
        self.pattern_size = (cols, rows)  # OpenCV expects (cols, rows)
        self.marker_size_mm = marker_size_mm
        self.aruco_dict_name = aruco_dict_name

        if self.pattern_type in ("charuco", "aprilgrid", "kalibr"):
            self.object_points = None
            if self.pattern_type == "charuco":
                dict_name = self.aruco_dict_name or "DICT_6X6_250"
                dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_6X6_250)
                self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
                m_size = self.marker_size_mm if (self.marker_size_mm and self.marker_size_mm > 0) else (0.75 * self.square_size_mm)
                self.charuco_board = cv2.aruco.CharucoBoard((cols, rows), self.square_size_mm, m_size, self.dictionary)
                if hasattr(cv2.aruco, "CharucoDetector"):
                    params = cv2.aruco.DetectorParameters()
                    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                    params.adaptiveThreshWinSizeMin = 3
                    params.adaptiveThreshWinSizeMax = 23
                    params.adaptiveThreshWinSizeStep = 4
                    params.minMarkerPerimeterRate = 0.01
                    charuco_params = cv2.aruco.CharucoParameters()
                    self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board, charuco_params, params)
            elif self.pattern_type in ("aprilgrid", "kalibr"):
                dict_name = self.aruco_dict_name or "DICT_APRILTAG_36h11"
                dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_APRILTAG_36h11)
                self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
                self.spacing_mm = self.marker_size_mm if (self.marker_size_mm and self.marker_size_mm > 0) else (0.3 * self.square_size_mm)

                # Precompute canonical 3D object points for all tags in Kalibr / AprilGrid target
                # Tag (r, c) origin is at (c * stride, r * stride).
                # Note: In OpenCV ArUco detector for AprilTag (tag36h11), the canonical marker corners
                # are oriented such that Corner 0 is top-right, Corner 1 is top-left,
                # Corner 2 is bottom-left, Corner 3 is bottom-right relative to target plane (X right, Y down).
                stride = self.square_size_mm + self.spacing_mm
                self.tag_3d_corners_canonical = {}
                self.tag_3d_corners_kalibr = {}
                for r in range(rows):
                    for c in range(cols):
                        t_id = r * cols + c
                        x0 = c * stride
                        y0 = r * stride
                        # Canonical OpenCV clockwise from top-left:
                        self.tag_3d_corners_canonical[t_id] = np.array([
                            [x0, y0, 0.0],
                            [x0 + self.square_size_mm, y0, 0.0],
                            [x0 + self.square_size_mm, y0 + self.square_size_mm, 0.0],
                            [x0, y0 + self.square_size_mm, 0.0]
                        ], dtype=np.float32)
                        # Kalibr / mirrored AprilGrid:
                        self.tag_3d_corners_kalibr[t_id] = np.array([
                            [x0 + self.square_size_mm, y0, 0.0],
                            [x0, y0, 0.0],
                            [x0, y0 + self.square_size_mm, 0.0],
                            [x0 + self.square_size_mm, y0 + self.square_size_mm, 0.0]
                        ], dtype=np.float32)
                self.tag_3d_corners = None

                refine_method = getattr(cv2.aruco, "CORNER_REFINE_APRILTAG", getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1))

                def _create_aruco_params(border_bits: int):
                    if hasattr(cv2.aruco, "DetectorParameters_create"):
                        p = cv2.aruco.DetectorParameters_create()
                    else:
                        p = cv2.aruco.DetectorParameters()
                    p.cornerRefinementMethod = refine_method
                    p.markerBorderBits = border_bits
                    p.adaptiveThreshWinSizeMin = 3
                    p.adaptiveThreshWinSizeMax = 23
                    p.adaptiveThreshWinSizeStep = 4
                    p.minMarkerPerimeterRate = 0.01
                    return p

                self.aruco_params_2bit = _create_aruco_params(2)
                self.aruco_params_1bit = _create_aruco_params(1)

                if hasattr(cv2.aruco, "ArucoDetector"):
                    self.aruco_detector = cv2.aruco.ArucoDetector(self.dictionary, self.aruco_params_2bit)
                    self.aruco_detector_1bit = cv2.aruco.ArucoDetector(self.dictionary, self.aruco_params_1bit)
                else:
                    self.aruco_detector = None
                    self.aruco_detector_1bit = None
        else:
            self.object_points = generate_object_points(rows, cols, square_size_mm, self.pattern_type)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect pattern corners in image with sub-pixel precision."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        sharpness = compute_sharpness(gray)
        obj_points = None

        if self.pattern_type == "checkerboard":
            # 1. Try findChessboardCornersSB (Sector-Based: modern, sub-pixel accurate, robust to non-flat paper)
            flags = (
                cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY
            )
            found, corners = cv2.findChessboardCornersSB(gray, self.pattern_size, flags=flags)

            # Fallback to standard detector if SB fails
            if not found:
                flags_std = (
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    | cv2.CALIB_CB_NORMALIZE_IMAGE
                    | cv2.CALIB_CB_FAST_CHECK
                )
                found, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags_std)
                if found and corners is not None:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        elif self.pattern_type == "circles":
            flags = cv2.CALIB_CB_SYMMETRIC_GRID
            found, corners = cv2.findCirclesGrid(gray, self.pattern_size, flags=flags)

        elif self.pattern_type == "asymmetric_circles":
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID
            found, corners = cv2.findCirclesGrid(gray, self.pattern_size, flags=flags)

        elif self.pattern_type == "charuco":
            found = False
            corners = None
            c, ids = None, None
            if hasattr(self, "charuco_detector"):
                c, ids, _, _ = self.charuco_detector.detectBoard(gray)
                if c is None or len(c) < 6:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    c2, ids2, _, _ = self.charuco_detector.detectBoard(clahe.apply(gray))
                    if c2 is not None and (c is None or len(c2) > len(c)):
                        c, ids = c2, ids2
            else:
                params = cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, "DetectorParameters_create") else cv2.aruco.DetectorParameters()
                mc, mi, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=params)
                if mc is not None and len(mc) > 0:
                    _, c, ids = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, self.charuco_board)

            if c is not None and ids is not None and len(c) >= 6:
                if hasattr(self.charuco_board, "matchImagePoints"):
                    obj_m, img_m = self.charuco_board.matchImagePoints(c, ids)
                else:
                    cb_corners = self.charuco_board.getChessboardCorners() if hasattr(self.charuco_board, "getChessboardCorners") else self.charuco_board.chessboardCorners
                    obj_m = cb_corners[ids.flatten()]
                    img_m = c
                if len(img_m) >= 6:
                    found = True
                    corners = img_m.reshape(-1, 1, 2).astype(np.float32)
                    obj_points = obj_m.reshape(-1, 3).astype(np.float32)

        elif self.pattern_type in ("aprilgrid", "kalibr"):
            found = False
            corners = None
            mc, mi = None, None
            def _detect_with(gray_img, p_det, p_params):
                if p_det is not None:
                    return p_det.detectMarkers(gray_img)
                return cv2.aruco.detectMarkers(gray_img, self.dictionary, parameters=p_params)

            # 1. Try 2-bit border (standard Kalibr / AprilGrid)
            mc, mi, _ = _detect_with(gray, self.aruco_detector, self.aruco_params_2bit)
            if mc is None or len(mc) < 3:
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray)
                mc2, mi2, _ = _detect_with(gray_clahe, self.aruco_detector, self.aruco_params_2bit)
                if mc2 is not None and (mc is None or len(mc2) > len(mc)):
                    mc, mi = mc2, mi2

            # 2. Try 1-bit border fallback
            if mc is None or len(mc) < 3:
                mc1, mi1, _ = _detect_with(gray, self.aruco_detector_1bit, self.aruco_params_1bit)
                if mc1 is not None and (mc is None or len(mc1) > len(mc)):
                    mc, mi = mc1, mi1

            if mc is not None and mi is not None and len(mc) >= 3:
                # If board corner orientation has not been calibrated, determine whether canonical or Kalibr ordering has lower residual
                if self.tag_3d_corners is None:
                    def _eval_ordering(corner_dict):
                        obj_p, img_p = [], []
                        for i, tid in enumerate(mi.flatten()):
                            if tid in corner_dict:
                                obj_p.append(corner_dict[tid])
                                img_p.append(mc[i][0])
                        if len(obj_p) < 3:
                            return 99999.0, 0
                        all_o = np.concatenate(obj_p, axis=0)
                        all_i = np.concatenate(img_p, axis=0)
                        H, inliers = cv2.findHomography(all_o[:, :2], all_i, cv2.RANSAC, 4.0)
                        if H is None or inliers is None:
                            return 99999.0, 0
                        pred = cv2.perspectiveTransform(all_o[:, :2].reshape(-1, 1, 2), H).reshape(-1, 2)
                        err = np.mean(np.linalg.norm(pred - all_i, axis=1))
                        return float(err), int(np.sum(inliers))

                    err_can, in_can = _eval_ordering(self.tag_3d_corners_canonical)
                    err_kal, in_kal = _eval_ordering(self.tag_3d_corners_kalibr)
                    if in_kal > in_can or (in_kal == in_can and err_kal < err_can):
                        self.tag_3d_corners = self.tag_3d_corners_kalibr
                    else:
                        self.tag_3d_corners = self.tag_3d_corners_canonical

                obj_list = []
                img_list = []
                for i, t_id in enumerate(mi.flatten()):
                    if t_id in self.tag_3d_corners:
                        obj_list.append(self.tag_3d_corners[t_id])
                        img_list.append(mc[i][0])

                if len(obj_list) >= 3:
                    found = True
                    corners = np.concatenate(img_list, axis=0).reshape(-1, 1, 2).astype(np.float32)
                    obj_points = np.concatenate(obj_list, axis=0).reshape(-1, 3).astype(np.float32)
        else:
            raise ValueError(f"Unsupported pattern type: {self.pattern_type}")

        if found and corners is not None:
            corners = corners.reshape(-1, 1, 2).astype(np.float32)
            # Determine center of detected corners to classify coverage zone
            cx = float(np.mean(corners[:, 0, 0]))
            cy = float(np.mean(corners[:, 0, 1]))
            h, w = gray.shape[:2]
            zone_x = min(int(cx / (w / 3)), 2)
            zone_y = min(int(cy / (h / 3)), 2)

            return DetectionResult(
                found=True,
                corners=corners,
                sharpness=sharpness,
                coverage_zone=(zone_y, zone_x),
                object_points=obj_points
            )

        return DetectionResult(found=False, sharpness=sharpness)

    def draw_corners(self, image: np.ndarray, corners: np.ndarray, found: bool = True) -> np.ndarray:
        """Render detected pattern corners on image."""
        vis = image.copy()
        if not found or corners is None or len(corners) == 0:
            return vis
        if self.pattern_type in ("checkerboard", "circles", "asymmetric_circles"):
            cv2.drawChessboardCorners(vis, self.pattern_size, corners, found)
        elif self.pattern_type == "charuco":
            for pt in corners[:, 0, :]:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
                cv2.circle(vis, (x, y), 6, (0, 0, 255), 1)
        elif self.pattern_type in ("aprilgrid", "kalibr"):
            for pt in corners[:, 0, :]:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(vis, (x, y), 3, (0, 255, 255), -1)
                cv2.circle(vis, (x, y), 5, (255, 0, 0), 1)
        return vis
