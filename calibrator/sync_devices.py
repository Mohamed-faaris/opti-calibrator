"""
Device Node Sync Utility.

Dynamically scans connected V4L2 cameras and updates output calibration profiles
(JSON, NPZ, and YAML) whenever cameras are unplugged, replugged into different USB ports,
or connected through a USB hub.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
import numpy as np
import yaml

from rich.console import Console
from rich.table import Table

from .stream_manager import enumerate_v4l2_streams, CameraDeviceStream

console = Console()


def clean_card_name(card_name: Optional[str]) -> str:
    """
    Strips hardware-dependent USB bus strings from card names.
    e.g. 'Intel(R) RealSense(TM) Depth Ca (usb-0000:00:14.0-2)' -> 'Intel(R) RealSense(TM) Depth Ca'
         'Astrap: Astrap (usb-0000:00:14.0-1.1)' -> 'Astrap: Astrap'
    """
    if not card_name:
        return ""
    # Strip (usb-...) suffix
    cleaned = re.sub(r"\s*\([^\)]*usb[^\)]*\)", "", card_name).strip()
    return cleaned


def find_matching_stream(
    target_card_name: Optional[str],
    target_stream_type: Optional[str],
    target_resolution: Optional[Tuple[int, int]],
    connected_streams: List[CameraDeviceStream],
    camera_name: Optional[str] = None
) -> Optional[CameraDeviceStream]:
    """
    Matches a calibration profile against currently connected V4L2 streams using:
      1. Clean card name (ignoring USB bus) + stream type
      2. Card name substring or camera_name keyword match
      3. Resolution match when multiple streams share the same card and type
    """
    if not connected_streams:
        return None

    target_card_clean = clean_card_name(target_card_name).lower()
    target_stream_lower = (target_stream_type or "").strip().lower()
    cam_name_lower = (camera_name or "").strip().lower()

    # Normalize aliases
    is_depth_target = "depth" in target_stream_lower
    is_ir_target = any(k in target_stream_lower for k in ["infrared", "grey", "ir"])
    is_color_target = any(k in target_stream_lower for k in ["color", "rgb", "mjpeg"])

    candidates: List[Tuple[int, CameraDeviceStream]] = []

    for stream in connected_streams:
        score = 0
        s_card_clean = clean_card_name(stream.card_name).lower()
        s_stream_lower = stream.stream_type.lower()

        # --- 1. Card Name Matching ---
        if target_card_clean and (target_card_clean == s_card_clean or target_card_clean in s_card_clean or s_card_clean in target_card_clean):
            score += 40
        elif target_card_clean:
            # Check keyword overlap (e.g. 'realsense', 'astrap', 'asus')
            target_words = [w for w in re.split(r"\W+", target_card_clean) if len(w) > 2]
            matched_words = sum(1 for w in target_words if w in s_card_clean)
            if matched_words:
                score += matched_words * 10
        elif cam_name_lower:
            # Check camera name keywords against stream card
            if "astra" in cam_name_lower and "astra" in s_card_clean:
                score += 30
            elif "rl" in cam_name_lower and "realsense" in s_card_clean:
                score += 30
            elif ("econ" in cam_name_lower or "webcam" in cam_name_lower) and ("asus" in s_card_clean or "webcam" in s_card_clean):
                score += 30

        if score == 0:
            continue

        # --- 2. Stream Type Matching ---
        if target_stream_lower:
            if target_stream_lower == s_stream_lower:
                score += 30
            elif is_depth_target and "depth" in s_stream_lower:
                score += 25
            elif is_ir_target and any(k in s_stream_lower for k in ["infrared", "grey", "ir"]):
                score += 25
            elif is_color_target and any(k in s_stream_lower for k in ["color", "rgb", "mjpeg"]):
                score += 25

        # --- 3. Target Resolution Matching ---
        if target_resolution:
            w, h = target_resolution
            supported_res = [(p.width, p.height) for p in stream.profiles]
            if (w, h) in supported_res:
                score += 15

        candidates.append((score, stream))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_stream = candidates[0]
    return best_stream if best_score >= 20 else None


def sync_device_nodes(
    output_dir: Path = Path("output"),
    dry_run: bool = False
) -> List[Dict]:
    """
    Scans all calibration profiles in output_dir, finds current V4L2 device nodes
    by matching camera card name and stream type, and updates the profile files.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.is_dir():
        console.print(f"[bold yellow]Output directory '{output_dir}' does not exist.[/bold yellow]")
        return []

    connected_streams = enumerate_v4l2_streams()
    results = []

    for profile_dir in sorted(output_dir.iterdir()):
        if not profile_dir.is_dir():
            continue

        json_path = profile_dir / "camera_calibration.json"
        npz_path = profile_dir / "camera_calibration.npz"
        yaml_path = profile_dir / "camera_info.yaml"

        if not json_path.exists() and not npz_path.exists():
            continue

        cam_name = profile_dir.name
        old_card = None
        old_node = None
        old_stream = None
        res_tuple = None

        json_data = {}
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                cam_name = json_data.get("camera_name", cam_name)
                dev_info = json_data.get("device_info", {})
                cfg = json_data.get("config", {})
                old_card = dev_info.get("card_name") or cfg.get("camera_card")
                old_node = dev_info.get("device_node") or cfg.get("device_node")
                old_stream = dev_info.get("stream_type") or cfg.get("stream_type")
                calib_info = json_data.get("calibration_info", {})
                if "image_width" in calib_info and "image_height" in calib_info:
                    res_tuple = (int(calib_info["image_width"]), int(calib_info["image_height"]))
            except Exception:
                pass

        if not old_card and npz_path.exists():
            try:
                npz_data = np.load(npz_path)
                old_card = str(npz_data["card_name"]) if "card_name" in npz_data else None
                old_node = str(npz_data["device_node"]) if "device_node" in npz_data else None
                old_stream = str(npz_data["stream_type"]) if "stream_type" in npz_data else None
                if not res_tuple and "image_size" in npz_data:
                    res_tuple = (int(npz_data["image_size"][0]), int(npz_data["image_size"][1]))
            except Exception:
                pass

        matched_stream = find_matching_stream(
            target_card_name=old_card,
            target_stream_type=old_stream,
            target_resolution=res_tuple,
            connected_streams=connected_streams,
            camera_name=cam_name
        )

        status = "Unchanged"
        new_node = old_node
        new_card = old_card
        new_stream = old_stream

        if matched_stream is not None:
            new_node = matched_stream.device_node
            new_card = matched_stream.card_name
            new_stream = matched_stream.stream_type

            if new_node != old_node or new_card != old_card or new_stream != old_stream:
                status = "Updated"
                if not dry_run:
                    # Update JSON
                    if json_path.exists() and json_data:
                        if "device_info" not in json_data:
                            json_data["device_info"] = {}
                        if "config" not in json_data:
                            json_data["config"] = {}
                        json_data["device_info"]["card_name"] = new_card
                        json_data["device_info"]["device_node"] = new_node
                        json_data["device_info"]["stream_type"] = new_stream
                        json_data["config"]["camera_card"] = new_card
                        json_data["config"]["device_node"] = new_node
                        json_data["config"]["stream_type"] = new_stream
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, indent=2)

                    # Update NPZ
                    if npz_path.exists():
                        npz_dict = dict(np.load(npz_path))
                        npz_dict["card_name"] = np.array(new_card)
                        npz_dict["device_node"] = np.array(new_node)
                        npz_dict["stream_type"] = np.array(new_stream)
                        np.savez(npz_path, **npz_dict)

                    # Update YAML if present
                    if yaml_path.exists():
                        try:
                            with open(yaml_path, "r", encoding="utf-8") as f:
                                yaml_data = yaml.safe_load(f)
                            if isinstance(yaml_data, dict):
                                yaml_data["camera_card"] = new_card
                                yaml_data["device_node"] = new_node
                                yaml_data["stream_type"] = new_stream
                                with open(yaml_path, "w", encoding="utf-8") as f:
                                    yaml.dump(yaml_data, f, sort_keys=False)
                        except Exception:
                            pass
        else:
            status = "Disconnected / Not Found"

        results.append({
            "camera_name": cam_name,
            "profile_dir": profile_dir.name,
            "old_node": old_node or "None",
            "new_node": new_node or "None",
            "card_name": new_card or old_card or "Unknown",
            "stream_type": new_stream or old_stream or "Unknown",
            "status": status
        })

    return results


def print_sync_table(results: List[Dict]) -> None:
    """Pretty prints sync results."""
    table = Table(title="🔄 Camera Device Node Sync Results", show_header=True, header_style="bold cyan")
    table.add_column("Profile", style="bold white")
    table.add_column("Old Node", style="dim")
    table.add_column("Current Node", style="bold green")
    table.add_column("Stream Type", style="cyan")
    table.add_column("Card Name", style="white")
    table.add_column("Status", style="yellow")

    for r in results:
        status_color = "green" if r["status"] == "Updated" else ("blue" if r["status"] == "Unchanged" else "red")
        table.add_row(
            r["profile_dir"],
            r["old_node"],
            r["new_node"],
            r["stream_type"],
            clean_card_name(r["card_name"]),
            f"[{status_color}]{r['status']}[/{status_color}]"
        )

    console.print(table)
