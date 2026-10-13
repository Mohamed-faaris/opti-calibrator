"""
Camera Calibrator Package.
"""

from .detector import PatternDetector, DetectionResult
from .frame_selector import OptimalFrameSelector, AdaptiveImportanceFrameSelector, CoverageTracker, FrameCandidate
from .camera_calibrator import CameraCalibrator, CalibrationResult, save_calibration_outputs

from .visualizer import CameraHUD, plot_coverage_heatmap, plot_reprojection_errors, create_undistort_comparison
from .stream_manager import open_camera_stream, enumerate_v4l2_streams, prompt_select_stream, find_stream_by_name
from .loader import CameraCalibration, load_calibration, load_from_npz, load_from_json, load_from_yaml, list_calibrations
from .sync_devices import sync_device_nodes, find_matching_stream, print_sync_table

__all__ = [
    "PatternDetector",
    "DetectionResult",
    "OptimalFrameSelector",
    "AdaptiveImportanceFrameSelector",
    "CoverageTracker",
    "FrameCandidate",
    "CameraCalibrator",
    "CalibrationResult",
    "save_calibration_outputs",
    "CameraHUD",
    "plot_coverage_heatmap",
    "plot_reprojection_errors",
    "create_undistort_comparison",
    "open_camera_stream",
    "enumerate_v4l2_streams",
    "prompt_select_stream",
    "find_stream_by_name",
    "CameraCalibration",
    "load_calibration",
    "load_from_npz",
    "load_from_json",
    "load_from_yaml",
    "list_calibrations",
    "sync_device_nodes",
    "find_matching_stream",
    "print_sync_table",
]

