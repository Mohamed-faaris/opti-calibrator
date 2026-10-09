# 📷 Camera Calibrator CLI

High-performance, pure Python camera calibration engine with video ingestion, live camera HUD, automatic spatial diversity keyframe selection, and paper-board distortion correction.

Built with **`uv`** and native **OpenCV (C++ accelerated)**.

---

## ⚡ Quick Start

### 1. From a Recorded Video (Recommended)
Record a video while moving your checkerboard/target across all corners, edges, and tilted angles:
```bash
uv run python calibrate.py --video my_recording.mp4
```

With custom grid dimensions and paper curvature model:
```bash
uv run python calibrate.py --video my_recording.mp4 --cols 9 --rows 6 --square-size 25 --model rational
```

### 2. Live Interactive Camera (with Real-time HUD)
```bash
uv run python calibrate.py --camera 0 --cols 9 --rows 6 --square-size 25
```
* **Real-time HUD**: 3×3 spatial coverage grid shows which zones need samples.
* **Smart Auto-capture**: automatically captures sharp keyframes when held steady in an uncovered zone.
* Controls:
  * `[SPACE]` — Manual capture
  * `[C]` — Finish & calibrate
  * `[Q]` — Quit

### 3. From a Directory of Images
```bash
uv run python calibrate.py --images ./photos/
```

---

## ⚙️ Options & Models

| Flag | Default | Description |
|---|---|---|
| `--camera-name`, `--name` | `None` | Camera identifier (e.g. `front_cam`). If omitted, prompts interactively or generates `camera_{w}x{h}_{timestamp}` |
| `--cols` | `9` | Inner corners along width |
| `--rows` | `6` | Inner corners along height |
| `--square-size` | `25.0` | Square/circle spacing in mm |
| `--pattern` | `checkerboard` | `checkerboard`, `circles`, or `asymmetric_circles` |
| `--model` | `standard` / `rational` | **Flat target (rigid glass/aluminum/acrylic)**: use `--model standard` (5-param $k_1, k_2, p_1, p_2, k_3$).<br>**Paper print (glued to cardboard or flexible)**: use `--model rational` (8-param) to compensate for slight non-flat curvature |
| `--target-frames` | `25` | Number of diverse keyframes to pick |
| `--min-sharpness`| `30.0` | Laplacian variance blur filter |
| `--filter-outliers` | `False` | Auto-removes worst outlier frames and re-calibrates |
| `--output-dir` | `output` | Base directory for results |
| `--no-gui` | `False` | Headless mode (no cv2 popup window) |

---

## 🎯 Target Flatness Guide: Which Model to Run?

- **If your checkerboard is FLAT (Rigid target on glass, acrylic, or precision aluminum plate)**:
  Run with `--model standard`:
  ```bash
  uv run python calibrate.py --video recording.mp4 --model standard --name my_camera
  ```
  *Why?* Standard pinhole calibration (5 parameters) is optimal and prevents overfitting higher-order radial parameters ($k_4, k_5, k_6$) when the physical target has zero warping.

- **If your checkerboard is on PAPER (Printed sheet, taped or glued to cardboard)**:
  Run with `--model rational`:
  ```bash
  uv run python calibrate.py --video recording.mp4 --model rational --name my_camera
  ```
  *Why?* Rational distortion model (8 parameters: $k_1 \dots k_6, p_1, p_2$) absorbs non-planar residual distortions caused by subtle paper wrinkles or bending.

---

## 💾 Generated Output Artifacts

All calibration data is saved under `output/<camera_name>/`:
* **`camera_calibration.json`** — Standard OpenCV camera matrix & distortion coefficients.
* **`camera_info.yaml`** — ROS / ROS2 compliant camera profile with camera name.
* **`camera_calibration.npz`** — Direct NumPy binary archive (`mtx`, `dist`).
* **`coverage_map.png`** — 2D density heatmap of sensor coverage.
* **`error_plot.png`** — Per-view reprojection error bar chart.
* **`undistort_demo.png`** — Side-by-side original vs undistorted image with straight grid lines.
* **`frames/`** — All extracted keyframe photos with detected corners drawn.
