"""
Camera Calibrator Package.
"""

from .detector import PatternDetector, DetectionResult
from .frame_selector import OptimalFrameSelector, CoverageTracker, FrameCandidate
from .camera_calibrator import CameraCalibrator, CalibrationResult, save_calibration_outputs
from .visualizer import CameraHUD, plot_coverage_heatmap, plot_reprojection_errors, create_undistort_comparison
from .stream_manager import open_camera_stream, enumerate_v4l2_streams, prompt_select_stream

__all__ = [
    "PatternDetector",
    "DetectionResult",
    "OptimalFrameSelector",
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
]
