"""
Frame selection and spatial coverage analyzer module.
Filters blurry frames and selects the most diverse subset for optimal calibration.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class FrameCandidate:
    frame_idx: int
    image: np.ndarray
    corners: np.ndarray
    sharpness: float
    coverage_zone: Tuple[int, int]
    center: Tuple[float, float]
    area: float
    selected: bool = False


class CoverageTracker:
    def __init__(self, grid_rows: int = 3, grid_cols: int = 3):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.counts = np.zeros((grid_rows, grid_cols), dtype=int)

    def record_corners(self, corners: np.ndarray, img_w: int, img_h: int) -> None:
        """Mark spatial cells covered by the corners."""
        cell_w = img_w / self.grid_cols
        cell_h = img_h / self.grid_rows

        for pt in corners[:, 0, :]:
            x, y = pt[0], pt[1]
            c = min(int(x / cell_w), self.grid_cols - 1)
            r = min(int(y / cell_h), self.grid_rows - 1)
            if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                self.counts[r, c] += 1

    def coverage_ratio(self) -> float:
        """Percentage of grid cells that have at least one corner sample."""
        return float(np.count_nonzero(self.counts) / (self.grid_rows * self.grid_cols))

    def missing_zones(self) -> List[str]:
        """Human-readable names of under-represented zones."""
        zone_names = [
            ["Top-Left", "Top-Center", "Top-Right"],
            ["Mid-Left", "Center", "Mid-Right"],
            ["Bottom-Left", "Bottom-Center", "Bottom-Right"]
        ]
        missing = []
        for r in range(min(self.grid_rows, 3)):
            for c in range(min(self.grid_cols, 3)):
                if self.counts[r, c] < 2:
                    missing.append(zone_names[r][c])
        return missing


class OptimalFrameSelector:
    def __init__(
        self,
        target_frames: int = 25,
        min_sharpness: float = 30.0,
        grid_rows: int = 4,
        grid_cols: int = 5
    ):
        self.target_frames = target_frames
        self.min_sharpness = min_sharpness
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def select(self, candidates: List[FrameCandidate], img_w: int, img_h: int) -> List[FrameCandidate]:
        """
        Greedy spatial-diversity selection algorithm.
        Picks sharpest frame first, then iteratively picks frames that fill
        uncovered spatial bins and board orientations.
        """
        valid = [c for c in candidates if c.sharpness >= self.min_sharpness]
        if not valid:
            valid = candidates  # Fallback if threshold was too strict

        if len(valid) <= self.target_frames:
            for c in valid:
                c.selected = True
            return valid

        # Spatial grid occupancy
        cell_w = img_w / self.grid_cols
        cell_h = img_h / self.grid_rows
        occupancy = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

        selected: List[FrameCandidate] = []
        remaining = valid.copy()

        # Step 1: Pick the overall sharpest frame as baseline
        best_initial = max(remaining, key=lambda c: c.sharpness)
        selected.append(best_initial)
        remaining.remove(best_initial)
        self._update_occupancy(best_initial.corners, occupancy, cell_w, cell_h)

        # Step 2: Iteratively select candidate maximizing (novelty * 0.7 + sharpness * 0.3)
        while len(selected) < self.target_frames and remaining:
            best_score = -1.0
            best_cand = None

            for cand in remaining:
                novelty = self._compute_novelty(cand.corners, occupancy, cell_w, cell_h)
                # Normalized sharpness score in [0, 1]
                sharp_score = min(cand.sharpness / 500.0, 1.0)
                score = novelty * 0.75 + sharp_score * 0.25

                if score > best_score:
                    best_score = score
                    best_cand = cand

            if best_cand is None:
                break

            selected.append(best_cand)
            remaining.remove(best_cand)
            self._update_occupancy(best_cand.corners, occupancy, cell_w, cell_h)

        for s in selected:
            s.selected = True

        return selected

    def _update_occupancy(
        self,
        corners: np.ndarray,
        occupancy: np.ndarray,
        cell_w: float,
        cell_h: float
    ) -> None:
        for pt in corners[:, 0, :]:
            c = min(int(pt[0] / cell_w), self.grid_cols - 1)
            r = min(int(pt[1] / cell_h), self.grid_rows - 1)
            occupancy[r, c] += 1

    def _compute_novelty(
        self,
        corners: np.ndarray,
        occupancy: np.ndarray,
        cell_w: float,
        cell_h: float
    ) -> float:
        """Measure how many corners fall into cells with low current count."""
        score = 0.0
        for pt in corners[:, 0, :]:
            c = min(int(pt[0] / cell_w), self.grid_cols - 1)
            r = min(int(pt[1] / cell_h), self.grid_rows - 1)
            count = occupancy[r, c]
            # Higher reward if cell is empty or lightly sampled
            score += 1.0 / (1.0 + count)
        return float(score / len(corners))
