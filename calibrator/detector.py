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
        rows: int = 6,
        cols: int = 9,
        square_size_mm: float = 25.0,
        pattern_type: str = "checkerboard"
    ):
        self.rows = rows
        self.cols = cols
        self.square_size_mm = square_size_mm
        self.pattern_type = pattern_type.lower()
        self.pattern_size = (cols, rows)  # OpenCV expects (cols, rows)
        self.object_points = generate_object_points(rows, cols, square_size_mm, self.pattern_type)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect pattern corners in image with sub-pixel precision."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        sharpness = compute_sharpness(gray)

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
                coverage_zone=(zone_y, zone_x)
            )

        return DetectionResult(found=False, sharpness=sharpness)

    def draw_corners(self, image: np.ndarray, corners: np.ndarray, found: bool = True) -> np.ndarray:
        """Render detected pattern corners with rainbow gradient lines."""
        vis = image.copy()
        cv2.drawChessboardCorners(vis, self.pattern_size, corners, found)
        return vis
