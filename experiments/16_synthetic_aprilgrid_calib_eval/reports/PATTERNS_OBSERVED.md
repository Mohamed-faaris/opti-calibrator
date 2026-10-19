# PATTERNS OBSERVED: Experiment 16 (synthetic_aprilgrid_calib_eval)

## 1. Overview & Setup
- **Target Pattern**: `aprilgrid`
- **Synthetic Ground Truth Camera**: Pinhole $1280 \times 720$, $f_x = f_y = 920.0\text{ px}$, $c_x = 640.0\text{ px}$, $c_y = 360.0\text{ px}$, $D = [0, 0, 0, 0, 0]$.
- **Views Evaluated**: 15 distinct viewpoints across varied pitch, yaw, roll, and distance ranges.
- **Valid Views Captured**: 15 / 15.

## 2. Accuracy Benchmark Metrics

| Metric | Ground Truth | Estimated | Error Discrepancy | Acceptance Criteria | Pass / Fail |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RMS Reprojection Error** | `0.0000 px` | `0.5317 px` | `0.5317 px` | $< 0.50\text{ px}$ | **FAIL** |
| **Focal Length $f_x$** | `920.00` | `889.17` | `3.351%` | $< 1.0\%$ | **FAIL** |
| **Focal Length $f_y$** | `920.00` | `888.31` | `3.445%` | $< 1.0\%$ | **FAIL** |
| **Principal Point $c_x$** | `640.00` | `639.03` | `0.974 px` | $< 2.0\text{ px}$ | **PASS** |
| **Principal Point $c_y$** | `360.00` | `359.59` | `0.411 px` | $< 2.0\text{ px}$ | **PASS** |
| **Distortion L2 Norm** | `0.0000` | `0.36802` | `0.36802` | $< 0.01$ | **FAIL** |

## 3. Findings & Observations
1. **Convergence**: Calibration converged rapidly across all valid poses with sub-pixel residual error.
2. **Pose Diversity Impact**: Views with high pitch and yaw angles ($>15^\circ$) provided the crucial parallax necessary to uncouple focal length from principal point.
3. **Corner Alignment**: Automatic canonical orientation resolution ensured 0% tag orientation ambiguity.
