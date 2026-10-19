"""
Adaptive frame selection and spatial coverage analyzer module.

Implements an information-theoretic importance selection algorithm:
- Rejects motion-blurred or low-quality photos.
- Measures pairwise corner displacement, board orientation (tilt/rotation), and scale.
- Prunes redundant / duplicate photos that don't add new geometric constraints.
- Automatically decides the optimal number of photos based on information gain,
  instead of forcing a rigid arbitrary limit (like 25).
- Retains all sharp, geometrically diverse photos for high-precision calibration and downstream datasets.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class FrameCandidate:
    frame_idx: int
    image: Optional[np.ndarray]
    corners: np.ndarray
    sharpness: float
    coverage_zone: Tuple[int, int]
    center: Tuple[float, float]
    area: float
    selected: bool = False
    importance_score: float = 0.0
    object_points: Optional[np.ndarray] = None


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


def compute_board_features(corners: np.ndarray, img_w: int, img_h: int) -> dict:
    """
    Extract geometric signature from checkerboard corners:
    - Normalized center (cx, cy) in [0, 1]
    - Normalized area ratio
    - Dominant tilt angle (degrees)
    - Aspect ratio distortion (indicative of pitch/yaw perspective slant)
    """
    pts = corners[:, 0, :]
    cx = float(np.mean(pts[:, 0])) / max(1, img_w)
    cy = float(np.mean(pts[:, 1])) / max(1, img_h)

    hull = cv2.convexHull(pts.astype(np.float32))
    area = float(cv2.contourArea(hull)) / max(1.0, float(img_w * img_h))

    # Compute tilt vector using first row or principal axis
    dx = float(pts[-1, 0] - pts[0, 0])
    dy = float(pts[-1, 1] - pts[0, 1])
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))

    # Compute bounding rect aspect
    x, y, w, h = cv2.boundingRect(pts.astype(np.float32))
    aspect = float(w) / max(1.0, float(h))

    return {
        "center": (cx, cy),
        "area": area,
        "angle": angle_deg,
        "aspect": aspect,
        "pts": pts
    }


def compute_frame_similarity(feat_a: dict, feat_b: dict, img_diag: float) -> float:
    """
    Compute similarity between two frame poses in [0.0, 1.0].
    1.0 = identical position, angle, and scale.
    0.0 = completely distinct view / different area of the sensor.
    """
    # 1. Corner point displacement (average Euclidean distance normalized by diagonal)
    pts_a = feat_a["pts"]
    pts_b = feat_b["pts"]
    if pts_a.shape == pts_b.shape:
        mean_disp = float(np.mean(np.linalg.norm(pts_a - pts_b, axis=1))) / max(1.0, img_diag)
    else:
        dx = (feat_a["center"][0] - feat_b["center"][0])
        dy = (feat_a["center"][1] - feat_b["center"][1])
        mean_disp = float(np.sqrt(dx*dx + dy*dy))

    # 2. Angle difference (degrees, wrapped in [0, 180])
    angle_diff = abs(feat_a["angle"] - feat_b["angle"]) % 180.0
    if angle_diff > 90.0:
        angle_diff = 180.0 - angle_diff
    angle_sim = max(0.0, 1.0 - (angle_diff / 45.0))

    # 3. Area / scale similarity
    area_a = max(feat_a["area"], 1e-6)
    area_b = max(feat_b["area"], 1e-6)
    area_ratio = min(area_a, area_b) / max(area_a, area_b)

    # Position similarity (1.0 when identical position, decreases sharply as distance increases)
    pos_sim = max(0.0, 1.0 - (mean_disp / 0.15))


    # Overall similarity is weighted blend
    similarity = 0.50 * pos_sim + 0.30 * angle_sim + 0.20 * area_ratio
    return float(np.clip(similarity, 0.0, 1.0))


class AdaptiveImportanceFrameSelector:
    """
    Self-optimizing frame selector.
    Automatically chooses the number of photos by evaluating:
      1. Sharpness & feature contrast
      2. Pairwise similarity (prunes near-duplicates where similarity > max_similarity_threshold)
      3. Geometric importance (spatial coverage across sensor periphery, center, and extreme tilts)
      4. Dynamic stopping criteria when additional frames yield diminishing marginal information gain.
    """

    def __init__(
        self,
        min_sharpness: float = 30.0,
        similarity_threshold: float = 0.82,  # Frames with >82% pose similarity are considered duplicates
        min_frames: int = 12,                # Lower bound for reliable numerical convergence
        max_frames: Optional[int] = None,    # None = no arbitrary cap, automatically picks all good distinct photos
        grid_rows: int = 4,
        grid_cols: int = 5
    ):
        self.min_sharpness = min_sharpness
        self.similarity_threshold = similarity_threshold
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def select(
        self,
        candidates: List[FrameCandidate],
        img_w: int,
        img_h: int
    ) -> List[FrameCandidate]:
        """
        Runs importance-driven deduplication and adaptive selection.
        Returns the optimal subset of FrameCandidate objects.
        """
        # 1. Filter out blurry frames
        valid = [c for c in candidates if c.sharpness >= self.min_sharpness]
        if len(valid) < self.min_frames:
            # Fallback: keep top frames by sharpness if strict threshold was too harsh
            valid = sorted(candidates, key=lambda c: c.sharpness, reverse=True)[:max(self.min_frames, len(candidates))]

        if len(valid) <= self.min_frames:
            for c in valid:
                c.selected = True
                c.importance_score = 1.0
            return valid

        img_diag = float(np.sqrt(img_w * img_w + img_h * img_h))

        # Precompute geometric features for all candidates
        features = [compute_board_features(c.corners, img_w, img_h) for c in valid]

        # Spatial grid occupancy to track camera coverage
        cell_w = img_w / self.grid_cols
        cell_h = img_h / self.grid_rows
        occupancy = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

        # 2. Step 1: Start with the highest sharpness anchor frame
        sharpest_idx = int(np.argmax([c.sharpness for c in valid]))
        selected_indices = [sharpest_idx]
        self._record_occupancy(valid[sharpest_idx].corners, occupancy, cell_w, cell_h)
        valid[sharpest_idx].importance_score = 1.0

        remaining_indices = [i for i in range(len(valid)) if i != sharpest_idx]

        # 3. Iterative greedy selection by Information Importance Gain
        # Runs until marginal gain drops below threshold OR no non-duplicate frames remain
        while remaining_indices:
            best_gain = -1.0
            best_idx = None

            for i in remaining_indices:
                cand = valid[i]
                cand_feat = features[i]

                # Max similarity to already selected photos
                max_sim = 0.0
                for sel_i in selected_indices:
                    sim = compute_frame_similarity(cand_feat, features[sel_i], img_diag)
                    if sim > max_sim:
                        max_sim = sim

                # If this frame is almost identical to an already selected photo, skip it (deduplicate)
                if max_sim >= self.similarity_threshold:
                    continue

                # Uniqueness / novelty score: 1.0 - max_similarity
                uniqueness = 1.0 - max_sim

                # Spatial novelty: rewards samples in uncovered or boundary grid cells
                spatial_novelty = self._compute_spatial_novelty(cand.corners, occupancy, cell_w, cell_h)

                # Tilt / boundary bonus: frames near the image boundaries constrain lens distortion most
                cx_norm, cy_norm = cand_feat["center"]
                dist_from_center = np.sqrt((cx_norm - 0.5)**2 + (cy_norm - 0.5)**2) * 2.0  # [0, 1.414]
                boundary_bonus = float(np.clip(dist_from_center, 0.0, 1.0))

                # Normalized sharpness score in [0, 1]
                sharp_score = min(cand.sharpness / 400.0, 1.0)

                # Overall information importance gain
                # High priority to: uniqueness (novel pose) + spatial coverage + lens boundary
                importance_gain = (
                    0.40 * uniqueness +
                    0.30 * spatial_novelty +
                    0.15 * boundary_bonus +
                    0.15 * sharp_score
                )

                if importance_gain > best_gain:
                    best_gain = importance_gain
                    best_idx = i

            # If no remaining photo provides sufficient new information or all are duplicates
            if best_idx is None or best_gain < 0.15:
                # If we haven't reached min_frames yet, relax threshold to reach min_frames
                if len(selected_indices) < self.min_frames and remaining_indices:
                    # Pick next sharpest remaining
                    best_idx = max(remaining_indices, key=lambda idx: valid[idx].sharpness)
                    best_gain = 0.20
                else:
                    # Naturally saturated information - stop adding duplicate frames!
                    break

            # Add to selected set
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            self._record_occupancy(valid[best_idx].corners, occupancy, cell_w, cell_h)
            valid[best_idx].importance_score = float(best_gain)

            # Check if user set an explicit max_frames cap
            if self.max_frames and len(selected_indices) >= self.max_frames:
                break

        selected_candidates = [valid[i] for i in selected_indices]
        for c in selected_candidates:
            c.selected = True

        return selected_candidates

    def _record_occupancy(
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

    def _compute_spatial_novelty(
        self,
        corners: np.ndarray,
        occupancy: np.ndarray,
        cell_w: float,
        cell_h: float
    ) -> float:
        score = 0.0
        for pt in corners[:, 0, :]:
            c = min(int(pt[0] / cell_w), self.grid_cols - 1)
            r = min(int(pt[1] / cell_h), self.grid_rows - 1)
            count = occupancy[r, c]
            # Higher reward for under-sampled cells
            score += 1.0 / (1.0 + count)
        return float(score / len(corners))


# Backward compatibility alias
OptimalFrameSelector = AdaptiveImportanceFrameSelector
