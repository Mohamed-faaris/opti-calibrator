"""
Stream Manager Module
Enumerates V4L2 camera devices, identifies available capture streams (RGB, Infrared, Depth),
and provides interactive stream and resolution selection.
"""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

console = Console()


@dataclass
class StreamProfile:
    width: int
    height: int
    format: str = "DEFAULT"
    fps_list: List[float] = field(default_factory=list)

    @property
    def resolution_str(self) -> str:
        return f"{self.width}×{self.height}"


@dataclass
class CameraDeviceStream:
    card_name: str
    device_node: str
    index: int
    stream_type: str
    formats: List[str]
    profiles: List[StreamProfile]

    @property
    def display_name(self) -> str:
        return f"{self.card_name} [{self.device_node}] ({self.stream_type})"


def detect_stream_type(formats: List[str], card_name: str) -> str:
    """Infer stream type from pixel formats and card name."""
    fmts_upper = [f.strip().upper() for f in formats]
    if any(f in fmts_upper for f in ["Z16", "INVZ", "INVR", "INRI", "INVI"]):
        return "Depth"
    elif any(f in fmts_upper for f in ["GREY", "Y8I", "Y12I", "Y8", "Y16"]):
        return "Infrared / Greyscale"
    elif any(f in fmts_upper for f in ["MJPG", "JPEG"]):
        return "Color (MJPEG)"
    elif any(f in fmts_upper for f in ["YUYV", "UYVY", "NV12", "RGB3", "BGR3"]):
        if "depth" in card_name.lower():
            return "RGB / Color"
        return "Color / Video"
    return "Video Capture"


def enumerate_v4l2_streams() -> List[CameraDeviceStream]:
    """
    Enumerate all connected V4L2 video capture devices using v4l2-ctl.
    Filters out metadata nodes and groups streams with supported resolutions.
    """
    try:
        raw_devices = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True)
    except Exception:
        return _fallback_opencv_enumeration()

    device_nodes = {}
    current_card = "Unknown Camera"
    for line in raw_devices.split("\n"):
        line_clean = line.strip()
        if not line_clean:
            continue
        if not line.startswith("\t") and not line.startswith(" "):
            current_card = line_clean.rstrip(":")
        elif line_clean.startswith("/dev/video"):
            if current_card not in device_nodes:
                device_nodes[current_card] = []
            device_nodes[current_card].append(line_clean)

    streams: List[CameraDeviceStream] = []

    for card, nodes in device_nodes.items():
        for node in nodes:
            try:
                fmts_ext = subprocess.check_output(
                    ["v4l2-ctl", "-d", node, "--list-formats-ext"],
                    stderr=subprocess.DEVNULL,
                    text=True
                )
                if not fmts_ext.strip():
                    continue

                formats: List[str] = []
                profiles_map = {}
                cur_fmt = "DEFAULT"
                cur_w, cur_h = None, None

                for line in fmts_ext.split("\n"):
                    line_s = line.strip()
                    m_fmt = re.search(r"'([A-Z0-9 ]{3,4})'", line_s)
                    if m_fmt:
                        cur_fmt = m_fmt.group(1).strip()
                        if cur_fmt not in formats:
                            formats.append(cur_fmt)
                        cur_w, cur_h = None, None

                    m_size = re.search(r"Size:\s+Discrete\s+(\d+)x(\d+)", line_s)
                    if m_size:
                        cur_w, cur_h = int(m_size.group(1)), int(m_size.group(2))
                        key = (cur_w, cur_h, cur_fmt)
                        if key not in profiles_map:
                            profiles_map[key] = StreamProfile(width=cur_w, height=cur_h, format=cur_fmt)

                    m_fps = re.search(r"\(([0-9.]+)\s+fps\)", line_s)
                    if m_fps and cur_w is not None and cur_h is not None:
                        fps_val = float(m_fps.group(1))
                        key = (cur_w, cur_h, cur_fmt)
                        if key in profiles_map and fps_val not in profiles_map[key].fps_list:
                            profiles_map[key].fps_list.append(fps_val)

                if not profiles_map:
                    continue

                # Sort resolutions descending, preferring MJPG for HD/FHD streams to prevent USB bandwidth saturation
                sorted_profiles = sorted(
                    profiles_map.values(),
                    key=lambda p: (p.width * p.height, 1 if p.format in ("MJPG", "JPEG") else 0, p.width),
                    reverse=True
                )

                digits = re.findall(r"\d+", node)
                idx = int(digits[-1]) if digits else 0
                stype = detect_stream_type(formats, card)

                streams.append(CameraDeviceStream(
                    card_name=card,
                    device_node=node,
                    index=idx,
                    stream_type=stype,
                    formats=formats,
                    profiles=sorted_profiles
                ))
            except Exception:
                continue

    return streams if streams else _fallback_opencv_enumeration()


def _fallback_opencv_enumeration() -> List[CameraDeviceStream]:
    """Fallback enumeration by testing video indices 0..5 with OpenCV."""
    streams = []
    for idx in range(6):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            streams.append(CameraDeviceStream(
                card_name=f"Camera Device {idx}",
                device_node=f"/dev/video{idx}",
                index=idx,
                stream_type="Video Capture",
                formats=["DEFAULT"],
                profiles=[StreamProfile(width=w, height=h)]
            ))
    return streams


def prompt_select_stream(
    streams: List[CameraDeviceStream],
    default_node_or_card: Optional[str] = None,
    default_resolution: Optional[str] = None
) -> Tuple[CameraDeviceStream, Optional[StreamProfile]]:
    """Interactive wizard to pick a camera stream and resolution."""
    if not streams:
        console.print("[yellow]No V4L2 cameras detected automatically. Defaulting to camera 0.[/yellow]")
        dummy = CameraDeviceStream(
            card_name="Default Webcam",
            device_node="/dev/video0",
            index=0,
            stream_type="Video",
            formats=["DEFAULT"],
            profiles=[]
        )
        return dummy, None

    # Step 1: Select camera stream
    table = Table(title="📹 Available Camera Streams", show_header=True, header_style="bold cyan")
    table.add_column("#", style="bold yellow")
    table.add_column("Device Node", style="cyan")
    table.add_column("Camera Card", style="white")
    table.add_column("Stream Type", style="green")
    table.add_column("Top Resolutions", style="dim")

    default_stream_idx = 1
    for i, s in enumerate(streams, 1):
        if default_node_or_card and (
            s.device_node == default_node_or_card
            or default_node_or_card.lower() in s.card_name.lower()
        ):
            default_stream_idx = i
        top_res = ", ".join([p.resolution_str for p in s.profiles[:3]])
        if len(s.profiles) > 3:
            top_res += f" (+{len(s.profiles) - 3} more)"
        table.add_row(str(i), s.device_node, s.card_name, s.stream_type, top_res)

    console.print(table)
    stream_idx = IntPrompt.ask(
        "Select camera stream number",
        choices=[str(i) for i in range(1, len(streams) + 1)],
        default=default_stream_idx
    )
    chosen_stream = streams[stream_idx - 1]

    # Step 2: Select stream resolution
    chosen_profile = None
    if chosen_stream.profiles:
        console.print(f"\n[bold]Select resolution for {chosen_stream.device_node} ({chosen_stream.stream_type}):[/bold]")
        prof_choices = []
        for j, prof in enumerate(chosen_stream.profiles[:12], 1):
            is_rec = ""
            if j == 1 or (prof.width == 1280 and prof.height == 720 and prof.format in ("MJPG", "DEFAULT")):
                is_rec = " [bold green](Recommended)[/bold green]"
            elif prof.width >= 1280 and prof.format in ("MJPG", "JPEG"):
                is_rec = " [bold cyan](Recommended for USB)[/bold cyan]"
            fps_str = f" @ {int(prof.fps_list[0])}fps" if prof.fps_list else ""
            console.print(f"  [cyan]{j}[/cyan]) {prof.resolution_str} ({prof.format}{fps_str}){is_rec}")
            prof_choices.append(str(j))

        prof_choices.append("default")
        console.print("  [cyan]default[/cyan]) Use camera native default")

        rec_default = "1"
        matched_user_default = False
        if default_resolution:
            for k, p in enumerate(chosen_stream.profiles[:12], 1):
                if p.resolution_str == default_resolution:
                    rec_default = str(k)
                    matched_user_default = True
                    if p.format in ("MJPG", "JPEG"):
                        break

        if not matched_user_default:
            for k, p in enumerate(chosen_stream.profiles[:12], 1):
                if p.width == 1280 and p.height == 720 and p.format in ("MJPG", "DEFAULT"):
                    rec_default = str(k)
                    break

        res_choice = Prompt.ask("Choose stream resolution", choices=prof_choices, default=rec_default)
        if res_choice != "default":
            chosen_profile = chosen_stream.profiles[int(res_choice) - 1]

    return chosen_stream, chosen_profile


def find_stream_by_name(
    query: str,
    streams: Optional[List[CameraDeviceStream]] = None
) -> Optional[CameraDeviceStream]:
    """
    Find a camera stream by card name, stream type, or device node substring.
    E.g. query='Intel', 'RealSense RGB', 'Astrap', or '/dev/video4'.
    """
    if streams is None:
        streams = enumerate_v4l2_streams()

    q = query.strip().lower()

    # 1. Exact node match
    for s in streams:
        if s.device_node.lower() == q:
            return s

    # 2. Match node digit (e.g. "4" -> "/dev/video4")
    if q.isdigit():
        target_node = f"/dev/video{q}"
        for s in streams:
            if s.device_node.lower() == target_node:
                return s

    # 3. Match all words in query across card name + stream type (e.g. "realsense rgb")
    q_words = [w for w in q.split() if w]
    for s in streams:
        combined = f"{s.card_name} {s.stream_type} {s.device_node}".lower()
        if all(word in combined for word in q_words):
            return s

    # 4. Substring in card name
    for s in streams:
        if q in s.card_name.lower():
            return s


    return None


def open_camera_stream(
    camera_target: Union[int, str],
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    target_fps: Optional[float] = None,
    target_fourcc: Optional[str] = None
) -> Tuple[cv2.VideoCapture, Tuple[int, int]]:
    """
    Opens camera stream using V4L2 on Linux with requested resolution and parameters.
    camera_target can be:
      - Device index: 0, 4
      - Device node path: "/dev/video4"
      - Camera card name / query: "RealSense RGB", "Astrap", "ORBBEC"
    Returns (cap, (actual_width, actual_height)).
    """
    # If camera_target is a string but not directly a digit or /dev/video*, check if it matches a card name
    matched_stream = None
    if isinstance(camera_target, str) and not camera_target.isdigit() and not camera_target.startswith("/dev/video"):
        matched_stream = find_stream_by_name(camera_target)
        if matched_stream is not None:
            console.print(f"[cyan]Matched camera target '{camera_target}' -> {matched_stream.card_name} [{matched_stream.device_node}] ({matched_stream.stream_type})[/cyan]")
            camera_target = matched_stream.device_node

    # Parse target index or device path
    if isinstance(camera_target, str) and camera_target.isdigit():
        target_handle = int(camera_target)
    elif isinstance(camera_target, str) and camera_target.startswith("/dev/video"):
        # Pass the exact path string '/dev/videoX' to avoid FFMPEG/V4L2 index out of range
        target_handle = camera_target
    else:
        target_handle = camera_target

    # Try V4L2 backend first on Linux
    cap = cv2.VideoCapture(target_handle, cv2.CAP_V4L2)
    if not cap.isOpened() and isinstance(target_handle, str) and target_handle.startswith("/dev/video"):
        digits = re.findall(r"\d+", target_handle)
        if digits:
            cap = cv2.VideoCapture(int(digits[-1]), cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(target_handle)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera stream: {camera_target}")



    # Set buffer size to 1: prevents queue backlog in kernel V4L2 buffers during frame processing
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Resolve FourCC: if not explicitly set and resolution is high (>=720p), prefer MJPG if supported
    fcc = target_fourcc
    if not fcc:
        if matched_stream and "MJPG" in matched_stream.formats:
            fcc = "MJPG"
        elif target_width and target_width >= 1280:
            try:
                node_str = camera_target if isinstance(camera_target, str) and camera_target.startswith("/dev/video") else f"/dev/video{target_handle}"
                out = subprocess.check_output(["v4l2-ctl", "-d", node_str, "--list-formats"], stderr=subprocess.DEVNULL, text=True)
                if "MJPG" in out or "JPEG" in out:
                    fcc = "MJPG"
            except Exception:
                pass

    if fcc:
        clean_fcc = fcc.ljust(4)[:4]
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*clean_fcc))

    # Set requested resolution
    if target_width and target_height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    # Set requested framerate (default to 30 FPS to prevent unconstrained 60-120 FPS USB packet loss)
    effective_fps = target_fps if target_fps is not None else 30.0
    cap.set(cv2.CAP_PROP_FPS, effective_fps)

    # Read a test frame to ensure stream is active and retrieve actual dimensions
    ret, frame = cap.read()
    if not ret or frame is None:
        # Retry once
        ret, frame = cap.read()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fcc_str = "".join([chr((actual_fcc_int >> (8 * i)) & 0xFF) for i in range(4)]).strip()
    backend_name = cap.getBackendName()

    if frame is not None:
        actual_h, actual_w = frame.shape[:2]

    console.print(
        f"[green]✓ Connected to {camera_target} via {backend_name} | "
        f"{actual_w}×{actual_h} px @ {actual_fps:.0f} FPS | FourCC: '{actual_fcc_str}'[/green]"
    )
    return cap, (actual_w, actual_h)


def is_frame_corrupted(frame: Optional[np.ndarray]) -> bool:
    """
    Detect horizontal green, magenta, black, and white corruption bands
    caused by UYVY byte drops or MJPEG packet loss.
    """
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) != 3:
        return True
    h, w, c = frame.shape
    if h < 16 or w < 16 or c != 3:
        return True

    # Subsample rows for fast evaluation (< 0.2 ms)
    sampled = frame[::4, :, :]
    g = sampled[:, :, 1].astype(float)
    r = sampled[:, :, 2].astype(float)

    # In horizontal green / magenta bands, mean row (G - R) disparity exceeds threshold
    row_g_diff = np.mean(g - r, axis=1)
    if float(np.max(np.abs(row_g_diff))) > 45.0:
        return True
    return False
