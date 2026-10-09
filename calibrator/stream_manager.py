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
    fmts_upper = [f.upper() for f in formats]
    if any(f in fmts_upper for f in ["Z16", "INVZ"]):
        return "Depth"
    elif any(f in fmts_upper for f in ["GREY", "Y8I", "Y12I", "Y8"]):
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

                for line in fmts_ext.split("\n"):
                    line_s = line.strip()
                    m_fmt = re.search(r"'([A-Z0-9]{3,4})'", line_s)
                    if m_fmt:
                        cur_fmt = m_fmt.group(1)
                        if cur_fmt not in formats:
                            formats.append(cur_fmt)

                    m_size = re.search(r"Size:\s+Discrete\s+(\d+)x(\d+)", line_s)
                    if m_size:
                        w, h = int(m_size.group(1)), int(m_size.group(2))
                        key = (w, h)
                        if key not in profiles_map:
                            profiles_map[key] = StreamProfile(width=w, height=h, format=cur_fmt)

                if not profiles_map:
                    continue

                # Sort resolutions descending
                sorted_profiles = sorted(
                    profiles_map.values(),
                    key=lambda p: (p.width * p.height, p.width),
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
    streams: List[CameraDeviceStream]
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

    for i, s in enumerate(streams, 1):
        top_res = ", ".join([p.resolution_str for p in s.profiles[:3]])
        if len(s.profiles) > 3:
            top_res += f" (+{len(s.profiles) - 3} more)"
        table.add_row(str(i), s.device_node, s.card_name, s.stream_type, top_res)

    console.print(table)
    stream_idx = IntPrompt.ask(
        "Select camera stream number",
        choices=[str(i) for i in range(1, len(streams) + 1)],
        default=1
    )
    chosen_stream = streams[stream_idx - 1]

    # Step 2: Select stream resolution
    chosen_profile = None
    if chosen_stream.profiles:
        console.print(f"\n[bold]Select resolution for {chosen_stream.device_node} ({chosen_stream.stream_type}):[/bold]")
        prof_choices = []
        for j, prof in enumerate(chosen_stream.profiles[:8], 1):
            is_rec = " [bold green](Recommended)[/bold green]" if j == 1 or (prof.width == 1280 and prof.height == 720) else ""
            console.print(f"  [cyan]{j}[/cyan]) {prof.resolution_str} ({prof.format}){is_rec}")
            prof_choices.append(str(j))

        prof_choices.append("default")
        console.print("  [cyan]default[/cyan]) Use camera native default")

        rec_default = "1"
        # If 1280x720 exists, find its index
        for k, p in enumerate(chosen_stream.profiles[:8], 1):
            if p.width == 1280 and p.height == 720:
                rec_default = str(k)
                break

        res_choice = Prompt.ask("Choose stream resolution", choices=prof_choices, default=rec_default)
        if res_choice != "default":
            chosen_profile = chosen_stream.profiles[int(res_choice) - 1]

    return chosen_stream, chosen_profile


def open_camera_stream(
    camera_target: Union[int, str],
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    target_fps: Optional[float] = None,
    target_fourcc: Optional[str] = None
) -> Tuple[cv2.VideoCapture, Tuple[int, int]]:
    """
    Opens camera stream using V4L2 on Linux with requested resolution and parameters.
    Returns (cap, (actual_width, actual_height)).
    """
    # Parse target index or device path
    if isinstance(camera_target, str) and camera_target.isdigit():
        target_idx = int(camera_target)
    elif isinstance(camera_target, str) and camera_target.startswith("/dev/video"):
        digits = re.findall(r"\d+", camera_target)
        target_idx = int(digits[-1]) if digits else camera_target
    else:
        target_idx = camera_target

    # Try V4L2 backend first on Linux
    cap = cv2.VideoCapture(target_idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(target_idx)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera stream: {camera_target}")

    # Set custom FOURCC format if specified
    if target_fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*target_fourcc))

    # Set requested resolution
    if target_width and target_height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    if target_fps:
        cap.set(cv2.CAP_PROP_FPS, target_fps)

    # Read a test frame to ensure stream is active and retrieve actual dimensions
    ret, frame = cap.read()
    if not ret or frame is None:
        # Retry once
        ret, frame = cap.read()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame is not None:
        actual_h, actual_w = frame.shape[:2]

    console.print(f"[green]✓ Connected to {camera_target} at {actual_w}×{actual_h} px[/green]")
    return cap, (actual_w, actual_h)
