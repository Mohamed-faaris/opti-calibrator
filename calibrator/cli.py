#!/usr/bin/env python3
"""
Camera Calibrator CLI
High-performance camera calibration for video files, live webcams, and image folders.
Supports standard, rational (paper targets), and thin prism distortion models.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.table import Table

from .camera_calibrator import CameraCalibrator, save_calibration_outputs
from .detector import PatternDetector
from .frame_selector import CoverageTracker, FrameCandidate, OptimalFrameSelector
from .visualizer import CameraHUD, create_undistort_comparison, plot_coverage_heatmap, plot_reprojection_errors

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Camera Calibrator CLI — Video & Live Stream Calibration Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Interactive mode
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive step-by-step CLI setup wizard"
    )

    # Camera naming & session resume
    parser.add_argument(
        "--camera-name",
        "--name",
        type=str,
        default=None,
        help="Name/identifier for camera (e.g. 'front_cam', 'ov9281'). If omitted, you will be prompted or a pattern will be used."
    )
    parser.add_argument(
        "--continue",
        "--resume",
        dest="resume_dir",
        type=str,
        default=None,
        help="Continue previous calculation from folder path or camera name (e.g. 'output/econ-lap0' or 'econ-lap0')"
    )

    # Input modes (optional if resuming/refining from an existing folder)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--video", type=str, help="Path to input video file (mp4, mkv, mov, avi)")
    input_group.add_argument("--camera", type=int, help="Live camera index (e.g. 0 for built-in or USB webcam)")
    input_group.add_argument("--images", type=str, help="Directory containing calibration image files")

    # Board geometry
    parser.add_argument("--cols", type=int, default=9, help="Number of inner corners along width (columns)")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner corners along height (rows)")
    parser.add_argument("--square-size", type=float, default=25.0, help="Square side length or circle spacing in mm")
    parser.add_argument(
        "--pattern",
        type=str,
        default="checkerboard",
        choices=["checkerboard", "circles", "asymmetric_circles"],
        help="Type of calibration pattern"
    )

    # Lens & model customization
    parser.add_argument(
        "--model",
        type=str,
        default="rational",
        choices=["standard", "rational", "thin_prism"],
        help="Distortion model. 'rational' (8-param) is recommended for paper prints with slight non-flat curvature"
    )

    # Frame filtering & optimization
    parser.add_argument("--target-frames", type=int, default=25, help="Number of optimal diverse keyframes to use")
    parser.add_argument("--min-sharpness", type=float, default=30.0, help="Minimum Laplacian variance to reject motion blur")
    parser.add_argument("--sample-fps", type=float, default=3.0, help="Sampling frequency (frames per second) for video files")
    parser.add_argument("--filter-outliers", action="store_true", help="Automatically remove high-error outlier frames and re-calibrate")

    # Outputs
    parser.add_argument("--output-dir", type=str, default="output", help="Directory where calibration outputs are saved")
    parser.add_argument("--save-frames", action="store_true", default=True, help="Save detected keyframe images with annotations")
    parser.add_argument("--no-gui", action="store_true", help="Disable OpenCV imshow windows (useful in headless environments)")

    return parser.parse_args()


def process_video_file(
    video_path: Path,
    detector: PatternDetector,
    sample_fps: float,
    min_sharpness: float
) -> List[FrameCandidate]:
    """Extract frames from a video file, detect pattern, and filter blurry ones."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        console.print(f"[bold red]Error: Could not open video file '{video_path}'[/bold red]")
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(fps / sample_fps))

    candidates: List[FrameCandidate] = []
    frame_idx = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Scanning '{video_path.name}'...", total=total_video_frames)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                detection = detector.detect(frame)
                if detection.found and detection.sharpness >= min_sharpness:
                    cx = float(np.mean(detection.corners[:, 0, 0]))
                    cy = float(np.mean(detection.corners[:, 0, 1]))
                    # Compute approximate board area on sensor
                    hull = cv2.convexHull(detection.corners[:, 0, :])
                    area = float(cv2.contourArea(hull))

                    candidates.append(FrameCandidate(
                        frame_idx=frame_idx,
                        image=frame,
                        corners=detection.corners,
                        sharpness=detection.sharpness,
                        coverage_zone=detection.coverage_zone,
                        center=(cx, cy),
                        area=area
                    ))

            frame_idx += 1
            progress.update(task, advance=1)

    cap.release()
    return candidates


def run_live_camera_capture(
    camera_idx: int,
    detector: PatternDetector,
    target_count: int,
    min_sharpness: float
) -> List[FrameCandidate]:
    """Interactive OpenCV window with real-time HUD and smart auto-capture."""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        console.print(f"[bold red]Error: Could not access camera index {camera_idx}[/bold red]")
        sys.exit(1)

    # Try setting reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hud = CameraHUD(grid_rows=3, grid_cols=3)
    coverage = CoverageTracker(grid_rows=3, grid_cols=3)
    candidates: List[FrameCandidate] = []
    frame_counter = 0

    console.print(Panel(
        "[bold green]Live Camera Capture Started[/bold green]\n\n"
        "• Hold the board inside each grid zone until it turns GREEN\n"
        "• Tilt the board at slight angles (pitch & yaw)\n"
        "• [yellow]Auto-capture[/yellow] triggers when board is steady in a new zone\n"
        "• Press [bold cyan][SPACE][/bold cyan] to capture manually\n"
        "• Press [bold cyan][C][/bold cyan] to calibrate when ready\n"
        "• Press [bold cyan][Q][/bold cyan] to abort",
        title="Interactive Guidance"
    ))

    last_auto_capture_time = 0.0
    window_name = "Camera Calibrator (Press 'C' to Calibrate)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        h, w = frame.shape[:2]

        if not window_initialized:
            # If camera feed is small (e.g. 640x480), open window with comfortable width
            if w < 960:
                disp_w = 960
                disp_h = int(960 * (h / w))
                cv2.resizeWindow(window_name, disp_w, disp_h)
            window_initialized = True

        detection = detector.detect(frame)
        is_steady = detection.sharpness >= min_sharpness
        now = time.time()

        # Auto capture condition: pattern found, steady, and at least 1.0s since last capture
        should_capture = False
        if detection.found and is_steady and (now - last_auto_capture_time > 1.0):
            # Prioritize under-represented zones, but continue collecting until 'C' is pressed
            zr, zc = detection.coverage_zone
            if coverage.counts[zr, zc] < 6 or (len(candidates) < target_count) or (detection.sharpness > min_sharpness * 1.3):
                should_capture = True
                last_auto_capture_time = now

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # ESC or Q
            break
        elif key in (ord('c'), ord('C')):
            if len(candidates) >= 4:
                console.print(f"\n[bold green]✓ Capture finished! Collected {len(candidates)} frames. Applying target frames selection...[/bold green]")
                break
            else:
                console.print("[yellow]Need at least 4 captured frames before calibrating![/yellow]")
        elif key == 32:  # SPACE
            if detection.found:
                should_capture = True
                last_auto_capture_time = now
            else:
                console.print("[yellow]No pattern detected in current frame to capture[/yellow]")

        if should_capture and detection.found:
            coverage.record_corners(detection.corners, w, h)
            cx = float(np.mean(detection.corners[:, 0, 0]))
            cy = float(np.mean(detection.corners[:, 0, 1]))
            hull = cv2.convexHull(detection.corners[:, 0, :])
            candidates.append(FrameCandidate(
                frame_idx=frame_counter,
                image=frame.copy(),
                corners=detection.corners,
                sharpness=detection.sharpness,
                coverage_zone=detection.coverage_zone,
                center=(cx, cy),
                area=float(cv2.contourArea(hull)),
                selected=True
            ))

        # Render display frame with HUD
        vis = frame.copy()
        if detection.found:
            vis = detector.draw_corners(vis, detection.corners, True)

        hud_frame = hud.draw_hud(
            vis,
            coverage,
            detection.sharpness,
            len(candidates),
            target_count,
            is_steady,
            detection.found
        )

        cv2.imshow(window_name, hud_frame)

    cap.release()
    cv2.destroyAllWindows()
    return candidates


def process_images_directory(
    images_dir: Path,
    detector: PatternDetector,
    min_sharpness: float
) -> List[FrameCandidate]:
    """Process a folder of image files."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts])

    if not image_paths:
        console.print(f"[bold red]No supported image files found in '{images_dir}'[/bold red]")
        sys.exit(1)

    candidates: List[FrameCandidate] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task(f"Processing {len(image_paths)} images...", total=len(image_paths))

        for idx, img_path in enumerate(image_paths):
            frame = cv2.imread(str(img_path))
            if frame is None:
                progress.update(task, advance=1)
                continue

            detection = detector.detect(frame)
            if detection.found and detection.sharpness >= min_sharpness:
                cx = float(np.mean(detection.corners[:, 0, 0]))
                cy = float(np.mean(detection.corners[:, 0, 1]))
                hull = cv2.convexHull(detection.corners[:, 0, :])
                candidates.append(FrameCandidate(
                    frame_idx=idx,
                    image=frame,
                    corners=detection.corners,
                    sharpness=detection.sharpness,
                    coverage_zone=detection.coverage_zone,
                    center=(cx, cy),
                    area=float(cv2.contourArea(hull)),
                    selected=True
                ))

            progress.update(task, advance=1)

    return candidates


def cli_arg_provided(*names: str) -> bool:
    """Check if any of the given argument names were explicitly passed via command line."""
    for name in names:
        for arg in sys.argv[1:]:
            if arg == name or arg.startswith(f"{name}="):
                return True
    return False


def resolve_resume_dir(resume_str: str, base_output_dir: Path) -> Path:
    """Find previous calibration directory by relative/absolute path or camera name."""
    p = Path(resume_str)
    if p.exists() and p.is_dir():
        return p.resolve()
    alt = base_output_dir / resume_str
    if alt.exists() and alt.is_dir():
        return alt.resolve()
    console.print(f"[bold red]Resume directory not found: '{resume_str}' (also searched '{alt}')[/bold red]")
    sys.exit(1)


def load_previous_session(
    resume_path: Path,
    detector: PatternDetector
) -> Tuple[List[FrameCandidate], dict, Optional[Tuple[int, int]]]:
    """
    Loads previous calibration session data from a folder.
    Tries keyframe_data.npz first, falling back to scanning frames/*.jpg.
    Returns (candidates, saved_json_config, image_size).
    """
    calib_json_path = resume_path / "camera_calibration.json"
    saved_json = {}
    if calib_json_path.exists():
        try:
            with open(calib_json_path, "r", encoding="utf-8") as f:
                saved_json = json.load(f)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read {calib_json_path.name}: {e}[/yellow]")

    image_size = None
    if "calibration_info" in saved_json:
        ci = saved_json["calibration_info"]
        if "image_width" in ci and "image_height" in ci:
            image_size = (int(ci["image_width"]), int(ci["image_height"]))

    candidates: List[FrameCandidate] = []
    keyframe_data_path = resume_path / "keyframe_data.npz"

    if keyframe_data_path.exists():
        console.print(f"[cyan]Loading cached keyframe points from {keyframe_data_path.name}...[/cyan]")
        data = np.load(keyframe_data_path, allow_pickle=True)
        corners_arr = data["corners"]  # (M, N, 1, 2)
        indices_arr = data["frame_indices"]
        sharpness_arr = data["sharpness"]
        if "image_size" in data:
            image_size = (int(data["image_size"][0]), int(data["image_size"][1]))

        frames_dir = resume_path / "frames"
        for i in range(len(indices_arr)):
            frame_idx = int(indices_arr[i])
            corners = corners_arr[i].astype(np.float32)
            sharpness = float(sharpness_arr[i])

            img_file = frames_dir / f"frame_{frame_idx:04d}.jpg"
            img = cv2.imread(str(img_file)) if img_file.exists() else None
            if img is not None and image_size is None:
                h, w = img.shape[:2]
                image_size = (w, h)

            pts = corners[:, 0, :]
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            w = image_size[0] if image_size else 1280
            h = image_size[1] if image_size else 720
            zx = min(int(cx / (w / 3)), 2)
            zy = min(int(cy / (h / 3)), 2)
            hull = cv2.convexHull(pts.astype(np.float32))
            area = float(cv2.contourArea(hull))

            candidates.append(FrameCandidate(
                frame_idx=frame_idx,
                image=img,
                corners=corners,
                sharpness=sharpness,
                coverage_zone=(zx, zy),
                center=(cx, cy),
                area=area,
                selected=True
            ))
        console.print(f"[green]✓ Loaded {len(candidates)} cached keyframes from previous session[/green]")

    else:
        # Fallback: scan frames/ directory
        frames_dir = resume_path / "frames"
        if frames_dir.exists():
            img_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
            if img_files:
                console.print(f"[cyan]Scanning {len(img_files)} previous frames in {frames_dir.name}/...[/cyan]")
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("[cyan]Re-detecting corners...", total=len(img_files))
                    for img_file in img_files:
                        img = cv2.imread(str(img_file))
                        if img is None:
                            progress.update(task, advance=1)
                            continue
                        if image_size is None:
                            h, w = img.shape[:2]
                            image_size = (w, h)

                        digits = re.findall(r'\d+', img_file.stem)
                        idx = int(digits[-1]) if digits else len(candidates) + 1

                        det_res = detector.detect(img)
                        if det_res.found and det_res.corners is not None:
                            pts = det_res.corners[:, 0, :]
                            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
                            hull = cv2.convexHull(pts.astype(np.float32))
                            candidates.append(FrameCandidate(
                                frame_idx=idx,
                                image=img,
                                corners=det_res.corners,
                                sharpness=det_res.sharpness,
                                coverage_zone=det_res.coverage_zone,
                                center=(cx, cy),
                                area=float(cv2.contourArea(hull)),
                                selected=True
                            ))
                        progress.update(task, advance=1)
                console.print(f"[green]✓ Successfully detected patterns in {len(candidates)}/{len(img_files)} previous frames[/green]")

    return candidates, saved_json, image_size


def resolve_camera_name(provided_name: Optional[str], image_size: Tuple[int, int]) -> str:
    """
    Resolves camera name. If not provided via args, interactively prompts user.
    If empty or non-interactive, falls back to a clean camera_{w}x{h}_{timestamp} pattern.
    """
    if provided_name and provided_name.strip():
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', provided_name.strip())

    w, h = image_size
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_pattern = f"camera_{w}x{h}_{now_str}"

    if sys.stdin.isatty():
        try:
            console.print("\n[bold yellow]Camera name not specified.[/bold yellow]")
            user_input = console.input(
                f"Enter camera name [dim](Press ENTER for '{default_pattern}')[/dim]: "
            ).strip()
            if user_input:
                return re.sub(r'[^a-zA-Z0-9_\-]', '_', user_input)
        except (KeyboardInterrupt, EOFError):
            console.print()

    console.print(f"[dim]Using generated camera name: '{default_pattern}'[/dim]")
    return default_pattern


def display_results_table(result, camera_name: str, removed_outliers: List[int] = None):
    """Render rich terminal summary table of calibration results."""
    K = result.camera_matrix
    D = result.dist_coeffs

    rms_style = "bold green" if result.rms_error < 0.5 else "bold yellow" if result.rms_error < 1.0 else "bold red"
    quality_label = "EXCELLENT (< 0.5 px)" if result.rms_error < 0.5 else "ACCEPTABLE (< 1.0 px)" if result.rms_error < 1.0 else "POOR (> 1.0 px)"

    table = Table(title="📐 Calibration Summary", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", width=28)
    table.add_column("Value", style="white")

    table.add_row("Camera Name", f"[bold green]{camera_name}[/bold green]")
    table.add_row("RMS Reprojection Error", f"[{rms_style}]{result.rms_error:.4f} pixels ({quality_label})[/{rms_style}]")
    table.add_row("Distortion Model", f"[bold]{result.distortion_model.upper()}[/bold] ({len(D)} parameters)")
    table.add_row("Resolution", f"{result.image_size[0]} × {result.image_size[1]} px")
    table.add_row("Focal Length (fx, fy)", f"{K[0, 0]:.2f} px, {K[1, 1]:.2f} px")
    table.add_row("Principal Point (cx, cy)", f"{K[0, 2]:.2f} px, {K[1, 2]:.2f} px")
    table.add_row("Field of View (H × V)", f"{result.fov_deg[0]:.1f}° × {result.fov_deg[1]:.1f}°")
    table.add_row("Radial Distortion (k1, k2)", f"{D[0]:.6f}, {D[1]:.6f}")
    if len(D) >= 5:
        table.add_row("Tangential Distortion (p1, p2)", f"{D[2]:.6f}, {D[3]:.6f}")
    if len(D) >= 8:
        table.add_row("Paper Higher-Order (k4, k5, k6)", f"{D[4]:.6f}, {D[5]:.6f}, {D[6]:.6f}")

    table.add_row("Views Used", f"{len(result.per_view_results)} frames")
    if removed_outliers:
        table.add_row("Outliers Filtered", f"[yellow]{len(removed_outliers)} frames removed ({removed_outliers})[/yellow]")

    console.print(table)


def find_existing_calibrations(base_output_dir: Path) -> List[Tuple[str, Path, dict]]:
    """Scan base_output_dir for existing calibration folders containing camera_calibration.json."""
    results = []
    if not base_output_dir.exists():
        return results
    for p in sorted(base_output_dir.iterdir()):
        if p.is_dir() and (p / "camera_calibration.json").exists():
            cfg = {}
            try:
                with open(p / "camera_calibration.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                pass
            results.append((p.name, p, cfg))
    return results


def run_interactive_wizard(args, base_output_dir: Path):
    """Interactive wizard to guide user step-by-step through calibration setup."""
    console.print(Panel(
        "[bold cyan]Camera Calibrator — Interactive CLI Wizard[/bold cyan]\n"
        "[dim]Easily configure and calibrate any camera without memorizing CLI arguments.[/dim]",
        title="🧙 Interactive Mode",
        border_style="cyan"
    ))

    existing = find_existing_calibrations(base_output_dir)

    if existing:
        console.print("\n[bold]Select an action:[/bold]")
        console.print("  [cyan]1[/cyan]) 🆕 Start new camera calibration")
        console.print("  [cyan]2[/cyan]) 🔄 Continue / refine an existing camera calibration")
        action = Prompt.ask("Choose action", choices=["1", "2"], default="1")
    else:
        action = "1"

    if action == "2":
        # Continue previous session
        console.print("\n[bold]Detected Existing Calibrations:[/bold]")
        choices = []
        for i, (name, path, cfg) in enumerate(existing, 1):
            ci = cfg.get("calibration_info", {})
            rms = ci.get("rms_reprojection_error_px", "N/A")
            res = f"{ci.get('image_width', '?')}x{ci.get('image_height', '?')}"
            views = ci.get("num_views_used", "?")
            console.print(f"  [cyan]{i}[/cyan]) [bold green]{name}[/bold green] [dim]({res}, {views} views, RMS: {rms} px)[/dim]")
            choices.append(str(i))
        choices.append("other")
        console.print("  [cyan]other[/cyan]) Specify custom folder path")

        sel = Prompt.ask("Select calibration to continue", choices=choices, default="1")
        if sel == "other":
            while True:
                custom_path = Prompt.ask("Enter folder path")
                p = Path(custom_path)
                if p.exists() and p.is_dir():
                    args.resume_dir = str(p)
                    break
                console.print(f"[red]Directory '{custom_path}' does not exist. Try again.[/red]")
        else:
            args.resume_dir = str(existing[int(sel) - 1][1])

        console.print(f"\n[bold]What would you like to do with '{Path(args.resume_dir).name}'?[/bold]")
        console.print("  [cyan]1[/cyan]) ➕ Add more frames (from a new video, camera, or images)")
        console.print("  [cyan]2[/cyan]) ⚡ Re-calibrate existing frames (with outlier filter or model change)")
        sub_action = Prompt.ask("Choose option", choices=["1", "2"], default="1")

        if sub_action == "1":
            console.print("\n[bold]Select new input source:[/bold]")
            console.print("  [cyan]1[/cyan]) Recorded video file")
            console.print("  [cyan]2[/cyan]) Live webcam stream")
            console.print("  [cyan]3[/cyan]) Directory of images")
            input_type = Prompt.ask("Input source", choices=["1", "2", "3"], default="1")
            if input_type == "1":
                while True:
                    vpath = Prompt.ask("Enter video file path")
                    if Path(vpath).exists():
                        args.video = vpath
                        break
                    console.print(f"[red]File '{vpath}' not found. Try again.[/red]")
            elif input_type == "2":
                args.camera = IntPrompt.ask("Enter camera index", default=0)
            else:
                while True:
                    ipath = Prompt.ask("Enter images directory")
                    if Path(ipath).exists():
                        args.images = ipath
                        break
                    console.print(f"[red]Directory '{ipath}' not found. Try again.[/red]")

        args.filter_outliers = Confirm.ask("Automatically filter high-error outlier frames?", default=True)
        return args

    # Start new calibration
    console.print("\n[bold]1. Choose Input Source:[/bold]")
    console.print("  [cyan]1[/cyan]) 🎥 Recorded video file (.mp4, .mkv, .mov, .avi)")
    console.print("  [cyan]2[/cyan]) 📹 Live interactive webcam (with real-time HUD)")
    console.print("  [cyan]3[/cyan]) 📁 Folder of photos/images")
    input_choice = Prompt.ask("Select input", choices=["1", "2", "3"], default="1")

    if input_choice == "1":
        while True:
            vpath = Prompt.ask("Enter video file path")
            if Path(vpath).exists():
                args.video = vpath
                break
            console.print(f"[red]File '{vpath}' not found. Try again.[/red]")
    elif input_choice == "2":
        args.camera = IntPrompt.ask("Enter live camera device index", default=0)
    else:
        while True:
            ipath = Prompt.ask("Enter image directory path")
            if Path(ipath).exists():
                args.images = ipath
                break
            console.print(f"[red]Directory '{ipath}' not found. Try again.[/red]")

    # Camera name
    console.print("\n[bold]2. Camera Identification:[/bold]")
    cam_name = Prompt.ask("Enter camera name [dim](Press ENTER for auto-generated)[/dim]", default="")
    args.camera_name = cam_name.strip() if cam_name.strip() else None

    # Pattern
    console.print("\n[bold]3. Calibration Target Pattern:[/bold]")
    console.print("  [cyan]1[/cyan]) 🏁 Checkerboard (most common)")
    console.print("  [cyan]2[/cyan]) ⚪ Circles Grid")
    console.print("  [cyan]3[/cyan]) 🔘 Asymmetric Circles Grid")
    pat_choice = Prompt.ask("Target pattern", choices=["1", "2", "3"], default="1")
    args.pattern = {"1": "checkerboard", "2": "circles", "3": "asymmetric_circles"}[pat_choice]

    # Grid dimensions
    console.print("\n[bold]4. Target Dimensions (Inner Corners):[/bold]")
    console.print("[dim]Inner corners = (number of black/white squares - 1)[/dim]")
    args.cols = IntPrompt.ask("Inner corners along width (columns)", default=9)
    args.rows = IntPrompt.ask("Inner corners along height (rows)", default=6)
    args.square_size = FloatPrompt.ask("Square size or circle spacing (in mm)", default=25.0)

    # Flatness & distortion model
    console.print("\n[bold]5. Target Flatness & Distortion Model:[/bold]")
    console.print("  [cyan]1[/cyan]) 📏 FLAT / RIGID target (Glass, acrylic, or precision aluminum plate)")
    console.print("      -> Uses standard 5-parameter model (prevents overfitting)")
    console.print("  [cyan]2[/cyan]) 📄 PAPER PRINT (Printed sheet glued/taped to cardboard)")
    console.print("      -> Uses rational 8-parameter model (absorbs subtle paper curvature)")
    flat_choice = Prompt.ask("Target type", choices=["1", "2"], default="1")
    args.model = "standard" if flat_choice == "1" else "rational"

    # Outlier filter
    args.filter_outliers = Confirm.ask("\nAutomatically filter high-error outlier frames?", default=True)

    return args


def main():
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if interactive wizard should be triggered
    has_cli_input = bool(args.video or (args.camera is not None) or args.images or args.resume_dir)
    if args.interactive or (not has_cli_input and sys.stdin.isatty()):
        args = run_interactive_wizard(args, base_output_dir)

    # Validate that at least one input or resume folder is provided
    has_input = bool(args.video or (args.camera is not None) or args.images or args.resume_dir)
    if not has_input:
        console.print("[bold red]Error: No input specified. Provide --video, --camera, --images, or --continue <folder>.[/bold red]")
        sys.exit(1)

    resume_path = None
    saved_json = {}
    prev_image_size = None

    if args.resume_dir:
        resume_path = resolve_resume_dir(args.resume_dir, base_output_dir)
        console.print(f"[bold cyan]Resuming previous session from:[/bold cyan] {resume_path}")

        calib_json = resume_path / "camera_calibration.json"
        if calib_json.exists():
            try:
                with open(calib_json, "r", encoding="utf-8") as f:
                    saved_json = json.load(f)
            except Exception:
                pass

        saved_cfg = saved_json.get("config", {})
        if not cli_arg_provided("--cols") and "cols" in saved_cfg:
            args.cols = int(saved_cfg["cols"])
        if not cli_arg_provided("--rows") and "rows" in saved_cfg:
            args.rows = int(saved_cfg["rows"])
        if not cli_arg_provided("--square-size") and "square_size_mm" in saved_cfg:
            args.square_size = float(saved_cfg["square_size_mm"])
        if not cli_arg_provided("--pattern") and "pattern" in saved_cfg:
            args.pattern = saved_cfg["pattern"]
        if not cli_arg_provided("--model") and "model" in saved_cfg:
            args.model = saved_cfg["model"]
        if not cli_arg_provided("--camera-name", "--name") and "camera_name" in saved_json:
            args.camera_name = saved_json["camera_name"]

    target_output_dir = resume_path.resolve() if resume_path else base_output_dir.resolve()
    console.print(Panel(
        f"[bold]Target:[/bold] {args.pattern.capitalize()} ({args.cols}×{args.rows} inner corners, {args.square_size}mm)\n"
        f"[bold]Distortion Model:[/bold] {args.model.capitalize()}\n"
        f"[bold]Output Directory:[/bold] {target_output_dir}",
        title="🚀 Camera Calibrator Initialized"
    ))

    # 1. Initialize detector
    detector = PatternDetector(
        rows=args.rows,
        cols=args.cols,
        square_size_mm=args.square_size,
        pattern_type=args.pattern
    )

    # 2. Load previous candidates if resuming
    prev_candidates: List[FrameCandidate] = []
    if resume_path:
        prev_candidates, _, prev_image_size = load_previous_session(resume_path, detector)

    # 3. Extract and detect candidate frames from new input if provided
    new_candidates: List[FrameCandidate] = []
    if args.video:
        new_candidates = process_video_file(Path(args.video), detector, args.sample_fps, args.min_sharpness)
    elif args.camera is not None:
        new_candidates = run_live_camera_capture(args.camera, detector, args.target_frames, args.min_sharpness)
    elif args.images:
        new_candidates = process_images_directory(Path(args.images), detector, args.min_sharpness)

    # Offset new frame indices so they don't collide with previous
    if prev_candidates and new_candidates:
        max_prev_idx = max((f.frame_idx for f in prev_candidates), default=0)
        for cand in new_candidates:
            cand.frame_idx += max_prev_idx

    all_candidates = prev_candidates + new_candidates

    if len(all_candidates) < 4:
        console.print(f"[bold red]Found only {len(all_candidates)} valid frames with pattern detected. Minimum 4 required.[/bold red]")
        sys.exit(1)

    if prev_candidates and new_candidates:
        console.print(f"[green]✓ Total frames available: {len(all_candidates)} ({len(prev_candidates)} previous + {len(new_candidates)} new)[/green]")
    else:
        console.print(f"[green]✓ Total valid frames: {len(all_candidates)}[/green]")

    # 4. Determine image dimensions and resolve camera name
    image_size = None
    for cand in all_candidates:
        if cand.image is not None:
            h, w = cand.image.shape[:2]
            image_size = (w, h)
            break
    if image_size is None:
        image_size = prev_image_size or (1280, 720)

    if resume_path:
        camera_name = args.camera_name or resume_path.name
        camera_output_dir = resume_path
    else:
        camera_name = resolve_camera_name(args.camera_name, image_size)
        camera_output_dir = base_output_dir / camera_name
        camera_output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Select optimal diverse frames
    target_count = args.target_frames
    if not cli_arg_provided("--target-frames") and prev_candidates and new_candidates:
        target_count = max(args.target_frames, len(prev_candidates) + 15)

    if len(all_candidates) > target_count:
        selector = OptimalFrameSelector(target_frames=target_count, min_sharpness=args.min_sharpness)
        selected_frames = selector.select(all_candidates, image_size[0], image_size[1])
        console.print(f"[green]✓ Selected {len(selected_frames)} spatially optimal frames from {len(all_candidates)} collected frames[/green]")
    else:
        selected_frames = all_candidates
        console.print(f"[green]✓ Using all {len(selected_frames)} collected frames for calibration[/green]")

    # 6. Run Calibration
    calibrator = CameraCalibrator(distortion_model=args.model)
    obj_points = [detector.object_points for _ in selected_frames]
    img_points = [f.corners for f in selected_frames]
    frame_indices = [f.frame_idx for f in selected_frames]

    with console.status("[bold cyan]Computing camera intrinsics and lens distortion...[/bold cyan]"):
        result = calibrator.calibrate(obj_points, img_points, image_size, frame_indices)

    removed_outliers = []
    if args.filter_outliers:
        with console.status("[bold cyan]Pruning high-error outlier frames...[/bold cyan]"):
            result, removed_outliers = calibrator.filter_outliers(
                result, obj_points, img_points, frame_indices
            )
            if removed_outliers:
                console.print(f"[yellow]Filtered {len(removed_outliers)} outlier frames for improved accuracy[/yellow]")

    # 7. Display Summary
    display_results_table(result, camera_name, removed_outliers)

    # 8. Save parameters (JSON, YAML, NPZ)
    config_dict = {
        "camera_name": camera_name,
        "pattern": args.pattern,
        "rows": args.rows,
        "cols": args.cols,
        "square_size_mm": args.square_size,
        "model": args.model
    }
    saved_files = save_calibration_outputs(result, camera_output_dir, config_dict, camera_name=camera_name)

    # 9. Save keyframe images & keyframe data cache
    if args.save_frames:
        frames_dir = camera_output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        for frame_obj in selected_frames:
            if frame_obj.frame_idx not in removed_outliers and frame_obj.image is not None:
                cv2.imwrite(str(frames_dir / f"frame_{frame_obj.frame_idx:04d}.jpg"), frame_obj.image)

    # Save keyframe coordinates for fast resuming in the future
    active_selected = [f for f in selected_frames if f.frame_idx not in removed_outliers]
    if active_selected:
        np.savez_compressed(
            camera_output_dir / "keyframe_data.npz",
            corners=np.array([f.corners for f in active_selected], dtype=np.float32),
            frame_indices=np.array([f.frame_idx for f in active_selected], dtype=np.int32),
            sharpness=np.array([f.sharpness for f in active_selected], dtype=np.float32),
            image_size=np.array([image_size[0], image_size[1]], dtype=np.int32)
        )

    # 10. Generate Visualizations (Coverage Heatmap, Error Bar Chart, Undistort Demo)
    heatmap_path = camera_output_dir / "coverage_map.png"
    active_corners = [f.corners for f in selected_frames if f.frame_idx not in removed_outliers]
    plot_coverage_heatmap(active_corners, image_size, heatmap_path)

    error_plot_path = camera_output_dir / "error_plot.png"
    plot_reprojection_errors(result, error_plot_path)

    undistort_path = camera_output_dir / "undistort_demo.png"
    sample_frame = next((f.image for f in selected_frames if f.image is not None), None)
    if sample_frame is not None:
        undistorted_sample = calibrator.undistort_image(sample_frame, result)
        create_undistort_comparison(sample_frame, undistorted_sample, undistort_path)

    console.print(Panel(
        f"• [cyan]Camera Name:[/cyan] [bold green]{camera_name}[/bold green]\n"
        f"• [cyan]OpenCV JSON:[/cyan] {saved_files['json']}\n"
        f"• [cyan]ROS YAML:[/cyan] {saved_files['yaml']}\n"
        f"• [cyan]NumPy NPZ:[/cyan] {saved_files['npz']}\n"
        f"• [cyan]Keyframe Data:[/cyan] {camera_output_dir / 'keyframe_data.npz'}\n"
        f"• [cyan]Coverage Map:[/cyan] {heatmap_path}\n"
        f"• [cyan]Error Plot:[/cyan] {error_plot_path}\n"
        f"• [cyan]Undistort Demo:[/cyan] {undistort_path}\n"
        f"• [cyan]Saved Frames:[/cyan] {camera_output_dir / 'frames'}",
        title="💾 All Calibration Artifacts Saved Successfully"
    ))

    # Optional interactive view
    if not args.no_gui and args.camera is None and sample_frame is not None:
        try:
            demo_img = cv2.imread(str(undistort_path))
            if demo_img is not None:
                h, w = demo_img.shape[:2]
                if w > 1600:
                    scale = 1600 / w
                    demo_img = cv2.resize(demo_img, (1600, int(h * scale)))
                cv2.imshow("Original vs Undistorted (Press any key to close)", demo_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
