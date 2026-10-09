"""
Camera Calibrator Package.
"""

from .detector import PatternDetector, DetectionResult
from .frame_selector import OptimalFrameSelector, CoverageTracker, FrameCandidate
from .camera_calibrator import CameraCalibrator, CalibrationResult, save_calibration_outputs
from .visualizer import CameraHUD, plot_coverage_heatmap, plot_reprojection_errors, create_undistort_comparison

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
]
