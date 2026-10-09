"""
Visualization module: live camera HUD overlay, coverage heatmaps, error plots,
and side-by-side undistortion comparisons.
"""

from pathlib import Path
from typing import List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from .camera_calibrator import CalibrationResult
from .frame_selector import CoverageTracker


class CameraHUD:
    """Renders real-time guidance HUD onto OpenCV frames during live capture."""

    def __init__(self, grid_rows: int = 3, grid_cols: int = 3):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def draw_hud(
        self,
        frame: np.ndarray,
        coverage: CoverageTracker,
        sharpness: float,
        captured_count: int,
        target_count: int,
        is_steady: bool,
        found_pattern: bool,
        message: str = ""
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw semi-transparent grid overlay
        overlay = vis.copy()
        cell_w = int(w / self.grid_cols)
        cell_h = int(h / self.grid_rows)

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                count = coverage.counts[r, c]
                x1, y1 = c * cell_w, r * cell_h
                x2, y2 = (c + 1) * cell_w, (r + 1) * cell_h

                if count >= 3:
                    color = (0, 200, 0)      # Green: well covered
                    alpha = 0.2
                elif count >= 1:
                    color = (0, 215, 255)    # Yellow: partial
                    alpha = 0.15
                else:
                    color = (0, 0, 255)      # Red: missing
                    alpha = 0.1

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 80, 80), 1)

        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

        # Top status bar background
        cv2.rectangle(vis, (0, 0), (w, 50), (20, 20, 20), -1)

        # Text: Pattern detection status
        pattern_str = "Pattern: DETECTED" if found_pattern else "Pattern: SEARCHING..."
        p_color = (0, 255, 0) if found_pattern else (100, 100, 255)
        cv2.putText(vis, pattern_str, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, p_color, 2)

        # Text: Sharpness
        sharp_color = (0, 255, 0) if is_steady else (0, 165, 255)
        cv2.putText(vis, f"Sharpness: {int(sharpness)}", (260, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sharp_color, 2)

        # Text: Captured count
        cov_pct = int(coverage.coverage_ratio() * 100)
        cv2.putText(vis, f"Frames: {captured_count}/{target_count} ({cov_pct}% cov)", (460, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Bottom guidance bar
        cv2.rectangle(vis, (0, h - 45), (w, h), (20, 20, 20), -1)
        missing = coverage.missing_zones()
        if missing:
            guidance = f"Move board to: {', '.join(missing[:3])}"
        else:
            guidance = "Excellent coverage! Press SPACE or 'C' to calibrate"

        if message:
            guidance = message

        cv2.putText(vis, guidance, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, "[SPACE]: Capture  [C]: Calibrate  [Q]: Quit", (w - 380, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return vis


def plot_coverage_heatmap(
    all_corners: List[np.ndarray],
    image_size: Tuple[int, int],
    save_path: Path
) -> None:
    """Generate 2D spatial density heatmap of corner coverage."""
    w, h = image_size
    pts_x = []
    pts_y = []

    for corners in all_corners:
        for pt in corners[:, 0, :]:
            pts_x.append(pt[0])
            pts_y.append(pt[1])

    plt.figure(figsize=(9, 6), dpi=150)
    plt.style.use("dark_background")

    # 2D Hexbin density
    hb = plt.hexbin(pts_x, pts_y, gridsize=30, cmap="viridis", mincnt=1, extent=[0, w, 0, h])
    cb = plt.colorbar(hb, label="Sample Density")
    cb.ax.tick_params(colors="white")

    plt.gca().invert_yaxis()
    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.title(f"Calibration Spatial Coverage Map ({len(all_corners)} Views)", fontsize=13, pad=12)
    plt.xlabel("Sensor Width (px)")
    plt.ylabel("Sensor Height (px)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_reprojection_errors(
    result: CalibrationResult,
    save_path: Path
) -> None:
    """Generate bar chart of per-view reprojection errors."""
    indices = [v.frame_idx for v in result.per_view_results]
    errors = [v.error for v in result.per_view_results]
    mean_err = float(np.mean(errors))

    plt.figure(figsize=(10, 5), dpi=150)
    plt.style.use("dark_background")

    bars = plt.bar(range(len(indices)), errors, color="#6366f1", width=0.6, edgecolor="#818cf8")

    # Highlight worst frame
    worst_idx = int(np.argmax(errors))
    bars[worst_idx].set_color("#ef4444")

    plt.axhline(mean_err, color="#10b981", linestyle="--", linewidth=1.5, label=f"Mean Error: {mean_err:.3f} px")
    plt.axhline(result.rms_error, color="#f59e0b", linestyle=":", linewidth=1.5, label=f"RMS Error: {result.rms_error:.3f} px")

    plt.xticks(range(len(indices)), [str(i) for i in indices], rotation=45, fontsize=8)
    plt.title("Per-Frame Reprojection Error (px)", fontsize=13, pad=12)
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Error (pixels)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_undistort_comparison(
    original_img: np.ndarray,
    undistorted_img: np.ndarray,
    save_path: Path
) -> None:
    """Generate side-by-side comparison image with grid lines showing distortion removal."""
    h, w = original_img.shape[:2]
    vis_orig = original_img.copy()
    vis_undist = undistorted_img.copy()

    # Draw reference straight grid lines to verify curvature removal
    line_color = (0, 0, 255)
    for x in range(int(w / 4), w, int(w / 4)):
        cv2.line(vis_orig, (x, 0), (x, h), line_color, 1)
        cv2.line(vis_undist, (x, 0), (x, h), line_color, 1)
    for y in range(int(h / 4), h, int(h / 4)):
        cv2.line(vis_orig, (0, y), (w, y), line_color, 1)
        cv2.line(vis_undist, (0, y), (w, y), line_color, 1)

    # Add labels
    cv2.putText(vis_orig, "ORIGINAL (Distorted)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(vis_undist, "UNDISTORTED", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    combined = np.hstack([vis_orig, vis_undist])
    cv2.imwrite(str(save_path), combined)
