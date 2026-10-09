"""
Camera calibration engine with paper target support, error scoring, and exports.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import cv2
import numpy as np
import yaml


@dataclass
class ViewResult:
    frame_idx: int
    error: float
    rvec: np.ndarray
    tvec: np.ndarray


@dataclass
class CalibrationResult:
    rms_error: float
    camera_matrix: np.ndarray          # 3x3
    dist_coeffs: np.ndarray            # 1xN (5, 8, or 12 params)
    image_size: Tuple[int, int]        # (width, height)
    distortion_model: str              # standard, rational, thin_prism
    per_view_results: List[ViewResult]
    fov_deg: Tuple[float, float]       # (hfov, vfov)
    focal_length_mm: Optional[float] = None


class CameraCalibrator:
    def __init__(self, distortion_model: str = "standard"):
        self.distortion_model = distortion_model.lower()

    def _get_flags(self) -> int:
        flags = 0
        if self.distortion_model == "rational":
            # 8-parameter rational model: k1..k6, p1, p2 (vital for non-rigid / curved paper targets)
            flags |= cv2.CALIB_RATIONAL_MODEL
        elif self.distortion_model == "thin_prism":
            flags |= cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
        elif self.distortion_model == "standard":
            flags = 0  # 5-param model (k1, k2, p1, p2, k3)
        else:
            raise ValueError(f"Unknown distortion model: {self.distortion_model}")
        return flags

    def calibrate(
        self,
        object_points_list: List[np.ndarray],
        image_points_list: List[np.ndarray],
        image_size: Tuple[int, int],  # (width, height)
        frame_indices: Optional[List[int]] = None
    ) -> CalibrationResult:
        """
        Run full camera calibration.
        Returns CalibrationResult with camera matrix, distortion coefficients,
        and per-view reprojection errors.
        """
        if len(object_points_list) < 4:
            raise ValueError("At least 4 valid views are required for calibration.")

        if frame_indices is None:
            frame_indices = list(range(len(object_points_list)))

        flags = self._get_flags()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points_list,
            image_points_list,
            image_size,
            None,
            None,
            flags=flags,
            criteria=criteria
        )

        # Compute per-view reprojection errors
        per_view: List[ViewResult] = []
        total_sq_err = 0.0
        total_points = 0

        for i, (obj, img, rvec, tvec) in enumerate(zip(object_points_list, image_points_list, rvecs, tvecs)):
            proj, _ = cv2.projectPoints(obj, rvec, tvec, mtx, dist)
            err = cv2.norm(img, proj, cv2.NORM_L2) / np.sqrt(len(obj))
            per_view.append(ViewResult(
                frame_idx=frame_indices[i],
                error=float(err),
                rvec=rvec,
                tvec=tvec
            ))
            total_sq_err += float(cv2.norm(img, proj, cv2.NORM_L2SQR))
            total_points += len(obj)

        overall_rms = float(np.sqrt(total_sq_err / total_points))

        # Field of View calculation
        fx, fy = mtx[0, 0], mtx[1, 1]
        w, h = image_size
        hfov = float(2 * np.arctan(w / (2 * fx)) * (180.0 / np.pi))
        vfov = float(2 * np.arctan(h / (2 * fy)) * (180.0 / np.pi))

        return CalibrationResult(
            rms_error=overall_rms,
            camera_matrix=mtx,
            dist_coeffs=dist.reshape(-1),
            image_size=image_size,
            distortion_model=self.distortion_model,
            per_view_results=per_view,
            fov_deg=(hfov, vfov)
        )

    def filter_outliers(
        self,
        calib_result: CalibrationResult,
        object_points_list: List[np.ndarray],
        image_points_list: List[np.ndarray],
        frame_indices: List[int],
        max_remove_ratio: float = 0.15
    ) -> Tuple[CalibrationResult, List[int]]:
        """
        Prune outlier frames that have disproportionately high reprojection errors,
        then re-run calibration. Returns improved CalibrationResult and removed indices.
        """
        errors = [v.error for v in calib_result.per_view_results]
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        threshold = mean_err + 1.5 * std_err

        # Keep frames with acceptable error
        keep_mask = [e <= threshold for e in errors]
        num_to_remove = len(errors) - sum(keep_mask)
        max_allowed = int(len(errors) * max_remove_ratio)

        if num_to_remove > max_allowed:
            # Only remove worst max_allowed
            sorted_indices = np.argsort(errors)
            keep_indices = set(sorted_indices[:len(errors) - max_allowed])
            keep_mask = [i in keep_indices for i in range(len(errors))]

        if sum(keep_mask) < 4:
            return calib_result, []  # Not enough frames to prune

        filtered_obj = [object_points_list[i] for i, k in enumerate(keep_mask) if k]
        filtered_img = [image_points_list[i] for i, k in enumerate(keep_mask) if k]
        filtered_idx = [frame_indices[i] for i, k in enumerate(keep_mask) if k]
        removed_indices = [frame_indices[i] for i, k in enumerate(keep_mask) if not k]

        if not removed_indices:
            return calib_result, []

        improved_result = self.calibrate(
            filtered_obj,
            filtered_img,
            calib_result.image_size,
            filtered_idx
        )
        return improved_result, removed_indices

    def undistort_image(
        self,
        image: np.ndarray,
        calib_result: CalibrationResult,
        balance_crop: float = 0.0
    ) -> np.ndarray:
        """Undistort an image using the calibrated camera parameters."""
        w, h = calib_result.image_size
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
            calib_result.camera_matrix,
            calib_result.dist_coeffs,
            (w, h),
            balance_crop,
            (w, h)
        )
        undistorted = cv2.undistort(
            image,
            calib_result.camera_matrix,
            calib_result.dist_coeffs,
            None,
            new_camera_mtx
        )
        return undistorted


def save_calibration_outputs(
    result: CalibrationResult,
    output_dir: Path,
    config: Dict
) -> Dict[str, Path]:
    """Export calibration parameters to JSON, YAML (ROS), and NPZ formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    w, h = result.image_size
    K = result.camera_matrix
    D = result.dist_coeffs.tolist()

    # 1. JSON (OpenCV & Computer Vision standard)
    json_path = output_dir / "camera_calibration.json"
    json_data = {
        "calibration_info": {
            "model": result.distortion_model,
            "rms_reprojection_error_px": round(result.rms_error, 4),
            "image_width": w,
            "image_height": h,
            "fov_horizontal_deg": round(result.fov_deg[0], 2),
            "fov_vertical_deg": round(result.fov_deg[1], 2),
            "num_views_used": len(result.per_view_results)
        },
        "camera_matrix": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "matrix_3x3": K.tolist()
        },
        "distortion_coefficients": D,
        "config": config
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # 2. YAML (ROS camera_info compliant)
    yaml_path = output_dir / "camera_info.yaml"
    # P projection matrix for ROS
    P = np.zeros((3, 4))
    P[:3, :3] = K
    yaml_data = {
        "image_width": w,
        "image_height": h,
        "camera_name": "calibrated_camera",
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [float(x) for x in K.flatten()]
        },
        "distortion_model": "rational_polynomial" if result.distortion_model == "rational" else "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": len(D),
            "data": [float(x) for x in D]
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [float(x) for x in P.flatten()]
        }
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    # 3. NPZ (Direct NumPy loading)
    npz_path = output_dir / "camera_calibration.npz"
    np.savez(
        npz_path,
        camera_matrix=K,
        dist_coeffs=result.dist_coeffs,
        image_size=np.array([w, h]),
        rms_error=result.rms_error
    )

    return {
        "json": json_path,
        "yaml": yaml_path,
        "npz": npz_path
    }
