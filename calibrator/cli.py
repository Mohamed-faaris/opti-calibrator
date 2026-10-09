#!/usr/bin/env python3
"""
Camera Calibrator CLI
High-performance camera calibration for video files, live webcams, and image folders.
Supports standard, rational (paper targets), and thin prism distortion models.
"""

import argparse
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
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

    # Camera naming
    parser.add_argument(
        "--camera-name",
        "--name",
        type=str,
        default=None,
        help="Name/identifier for camera (e.g. 'front_cam', 'ov9281'). If omitted, you will be prompted or a pattern will be used."
    )

    # Input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        h, w = frame.shape[:2]
        detection = detector.detect(frame)
        is_steady = detection.sharpness >= min_sharpness
        now = time.time()

        # Auto capture condition: pattern found, steady, and at least 1.5s since last capture
        should_capture = False
        if detection.found and is_steady and (now - last_auto_capture_time > 1.5):
            # Check if this zone needs more samples
            zr, zc = detection.coverage_zone
            if coverage.counts[zr, zc] < 4 or len(candidates) < target_count:
                should_capture = True
                last_auto_capture_time = now

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # ESC or Q
            break
        elif key in (ord('c'), ord('C')):
            if len(candidates) >= 4:
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

        cv2.imshow("Camera Calibrator (Press 'C' to Calibrate)", hud_frame)

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


def main():
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold]Target:[/bold] {args.pattern.capitalize()} ({args.cols}×{args.rows} inner corners, {args.square_size}mm)\n"
        f"[bold]Distortion Model:[/bold] {args.model.capitalize()}\n"
        f"[bold]Base Output Directory:[/bold] {base_output_dir.resolve()}",
        title="🚀 Camera Calibrator Initialized"
    ))

    # 1. Initialize detector
    detector = PatternDetector(
        rows=args.rows,
        cols=args.cols,
        square_size_mm=args.square_size,
        pattern_type=args.pattern
    )

    # 2. Extract and detect candidate frames
    if args.video:
        candidates = process_video_file(Path(args.video), detector, args.sample_fps, args.min_sharpness)
    elif args.camera is not None:
        candidates = run_live_camera_capture(args.camera, detector, args.target_frames, args.min_sharpness)
    else:
        candidates = process_images_directory(Path(args.images), detector, args.min_sharpness)

    if len(candidates) < 4:
        console.print(f"[bold red]Found only {len(candidates)} valid frames with pattern detected. Minimum 4 required.[/bold red]")
        sys.exit(1)

    console.print(f"[green]✓ Detected valid pattern in {len(candidates)} frames[/green]")

    # 3. Determine image dimensions and resolve camera name
    img_h, img_w = candidates[0].image.shape[:2]
    image_size = (img_w, img_h)

    camera_name = resolve_camera_name(args.camera_name, image_size)
    camera_output_dir = base_output_dir / camera_name
    camera_output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Select optimal diverse frames
    selector = OptimalFrameSelector(target_frames=args.target_frames, min_sharpness=args.min_sharpness)
    selected_frames = selector.select(candidates, img_w, img_h)

    console.print(f"[green]✓ Selected {len(selected_frames)} spatially optimal frames for calibration[/green]")

    # 5. Run Calibration
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

    # 6. Display Summary
    display_results_table(result, camera_name, removed_outliers)

    # 7. Save parameters (JSON, YAML, NPZ)
    config_dict = {
        "camera_name": camera_name,
        "pattern": args.pattern,
        "rows": args.rows,
        "cols": args.cols,
        "square_size_mm": args.square_size,
        "model": args.model
    }
    saved_files = save_calibration_outputs(result, camera_output_dir, config_dict, camera_name=camera_name)

    # 8. Save keyframe images
    if args.save_frames:
        frames_dir = camera_output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        for frame_obj in selected_frames:
            if frame_obj.frame_idx not in removed_outliers:
                annotated = detector.draw_corners(frame_obj.image, frame_obj.corners, True)
                cv2.imwrite(str(frames_dir / f"frame_{frame_obj.frame_idx:04d}.jpg"), annotated)

    # 9. Generate Visualizations (Coverage Heatmap, Error Bar Chart, Undistort Demo)
    heatmap_path = camera_output_dir / "coverage_map.png"
    active_corners = [f.corners for f in selected_frames if f.frame_idx not in removed_outliers]
    plot_coverage_heatmap(active_corners, image_size, heatmap_path)

    error_plot_path = camera_output_dir / "error_plot.png"
    plot_reprojection_errors(result, error_plot_path)

    undistort_path = camera_output_dir / "undistort_demo.png"
    sample_frame = selected_frames[0].image
    undistorted_sample = calibrator.undistort_image(sample_frame, result)
    create_undistort_comparison(sample_frame, undistorted_sample, undistort_path)

    console.print(Panel(
        f"• [cyan]Camera Name:[/cyan] [bold green]{camera_name}[/bold green]\n"
        f"• [cyan]OpenCV JSON:[/cyan] {saved_files['json']}\n"
        f"• [cyan]ROS YAML:[/cyan] {saved_files['yaml']}\n"
        f"• [cyan]NumPy NPZ:[/cyan] {saved_files['npz']}\n"
        f"• [cyan]Coverage Map:[/cyan] {heatmap_path}\n"
        f"• [cyan]Error Plot:[/cyan] {error_plot_path}\n"
        f"• [cyan]Undistort Demo:[/cyan] {undistort_path}\n"
        f"• [cyan]Saved Frames:[/cyan] {camera_output_dir / 'frames'}",
        title="💾 All Calibration Artifacts Saved Successfully"
    ))

    # Optional interactive view
    if not args.no_gui and args.camera is None:
        try:
            demo_img = cv2.imread(str(undistort_path))
            if demo_img is not None:
                # Resize for display if huge
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
