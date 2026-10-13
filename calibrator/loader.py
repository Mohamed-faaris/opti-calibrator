"""
Calibration loader module.

Easily load and use camera calibration files (NPZ, JSON, YAML) in external
applications, computer vision pipelines, and OpenCV/ROS workflows.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import cv2
import numpy as np
import yaml


@dataclass
class CameraCalibration:
    """
    Standard camera calibration container.

    Attributes:
        camera_name: Camera identifier or stream name.
        camera_matrix: 3x3 intrinsic matrix K (fx, fy, cx, cy).
        dist_coeffs: 1xN distortion coefficients D.
        image_size: (width, height) resolution in pixels.
        rms_error: Reprojection error in pixels (if available).
        distortion_model: Distortion model name (standard, rational, thin_prism).
        card_name: V4L2 camera card name (e.g. "Intel(R) RealSense(TM) Depth Ca", "Astrap: Astrap").
        device_node: V4L2 device node path (e.g. "/dev/video4").
        stream_type: Stream type (e.g. "RGB / Color", "Color (MJPEG)", "Infrared / Greyscale").
        npz_path: Path to the .npz archive if available.
        source_path: Path to the loaded calibration file or directory.
        raw_metadata: Full metadata dictionary loaded from JSON/YAML if available.
    """
    camera_name: str
    camera_matrix: np.ndarray              # 3x3 float64
    dist_coeffs: np.ndarray                # 1xN float64
    image_size: Tuple[int, int]            # (width, height)
    rms_error: Optional[float] = None
    distortion_model: str = "standard"
    card_name: Optional[str] = None
    device_node: Optional[str] = None
    stream_type: Optional[str] = None
    npz_path: Optional[Path] = None
    source_path: Optional[Path] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    # Cached undistortion maps for high performance
    _map1: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _map2: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _new_camera_matrix: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height(self) -> int:
        return self.image_size[1]

    @property
    def resolution(self) -> Tuple[int, int]:
        """(width, height)"""
        return self.image_size

    @property
    def resolution_str(self) -> str:
        """Formatted resolution e.g. '1280x720'"""
        return f"{self.width}x{self.height}"

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    @property
    def K(self) -> np.ndarray:
        """Alias for camera_matrix."""
        return self.camera_matrix

    @property
    def D(self) -> np.ndarray:
        """Alias for dist_coeffs."""
        return self.dist_coeffs

    def resolve_live_device_node(
        self,
        override_node: Optional[Union[str, int]] = None,
        verbose: bool = True
    ) -> str:
        """
        Dynamically resolve the active V4L2 device node for this camera.

        Resolution logic:
          1. Explicit user override (--camera / device_node_override): used directly.
          2. Check recorded device_node: if it exists, verify whether card_name AND stream_type match.
          3. If recorded node does not exist or its card/stream mismatch:
             Scans connected cameras to find the one matching both card_name and stream_type,
             logs a WARNING that the camera moved, and updates self.device_node.

        Returns:
            Resolved device node path (e.g. '/dev/video8').
        """
        import logging
        from .stream_manager import enumerate_v4l2_streams
        from .sync_devices import find_matching_stream, clean_card_name

        logger = logging.getLogger("calibrator")

        # 1. User explicit override takes top precedence
        if override_node is not None:
            resolved = f"/dev/video{override_node}" if isinstance(override_node, int) or str(override_node).isdigit() else str(override_node)
            return resolved

        connected = enumerate_v4l2_streams()
        recorded_node = self.device_node

        # Helper to check if a stream matches this calibration's card and stream type
        def _stream_matches(s) -> bool:
            card_match = True
            if self.card_name:
                c1 = clean_card_name(self.card_name).lower()
                c2 = clean_card_name(s.card_name).lower()
                card_match = (c1 == c2) or (c1 in c2) or (c2 in c1)
            type_match = True
            if self.stream_type and s.stream_type:
                t1 = self.stream_type.strip().lower()
                t2 = s.stream_type.strip().lower()
                if "depth" in t1:
                    type_match = "depth" in t2
                elif any(k in t1 for k in ["ir", "infrared", "grey"]):
                    type_match = any(k in t2 for k in ["ir", "infrared", "grey"])
                elif any(k in t1 for k in ["color", "rgb", "mjpeg"]):
                    type_match = any(k in t2 for k in ["color", "rgb", "mjpeg"])
                else:
                    type_match = (t1 == t2)
            return card_match and type_match

        # 2. Check if the recorded node is currently active and matches card & stream type
        node_matches = False
        if recorded_node:
            for s in connected:
                if s.device_node.lower() == recorded_node.lower():
                    if _stream_matches(s):
                        node_matches = True
                    else:
                        if verbose:
                            logger.warning(
                                f"Recorded node '{recorded_node}' for '{self.camera_name}' does not match "
                                f"expected stream! Expected Card='{self.card_name}', Stream='{self.stream_type}'; "
                                f"Found Card='{s.card_name}', Stream='{s.stream_type}'."
                            )
                    break

        if node_matches and recorded_node:
            return recorded_node

        # 3. Search and find which connected camera matches card_name and stream_type
        matched = find_matching_stream(
            target_card_name=self.card_name,
            target_stream_type=self.stream_type,
            target_resolution=self.resolution,
            connected_streams=connected,
            camera_name=self.camera_name
        )

        if matched:
            new_node = matched.device_node
            if verbose:
                logger.warning(
                    f"[CAMERA MOVED / RE-MAPPED] Camera '{self.camera_name}' (Stream: '{self.stream_type}') "
                    f"node changed from '{recorded_node or 'None'}' -> '{new_node}' "
                    f"(Card: '{matched.card_name}'). Automatically updated target node."
                )
            self.device_node = new_node
            return new_node

        # Fallback if no matching device found on system
        if verbose:
            logger.warning(
                f"Could not find connected stream matching '{self.card_name}' ({self.stream_type}). "
                f"Falling back to recorded node '{recorded_node}'."
            )
        return recorded_node or self.card_name or self.camera_name

    def open_stream(
        self,
        fps: Optional[float] = None,
        fourcc: Optional[str] = None,
        device_node_override: Optional[Union[str, int]] = None
    ) -> Tuple[cv2.VideoCapture, Tuple[int, int]]:
        """
        Open the live camera stream for this calibration profile.
        
        Args:
            fps: Optional target framerate.
            fourcc: Optional FourCC codec (e.g. 'MJPG').
            device_node_override: Explicit user override (e.g. '/dev/video4' or 4). Takes precedence.
        """
        from .stream_manager import open_camera_stream

        target = self.resolve_live_device_node(override_node=device_node_override)

        return open_camera_stream(
            camera_target=target,
            target_width=self.width,
            target_height=self.height,
            target_fps=fps,
            target_fourcc=fourcc
        )


    def get_optimal_new_camera_matrix(
        self,
        alpha: float = 0.0,
        new_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Compute optimal new camera matrix for undistortion.
        
        Args:
            alpha: 0.0 = all pixels valid (cropped edges), 1.0 = keep all pixels (black borders)
            new_size: Optional target size (w, h). Defaults to original image_size.
        """
        size = new_size if new_size is not None else self.image_size
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            self.image_size,
            alpha,
            size
        )
        return new_K, roi

    def init_undistort_rectify_map(
        self,
        alpha: float = 0.0,
        m1type: int = cv2.CV_16SC2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precompute remap lookups for fast undistortion in real-time loops.
        """
        new_K, _ = self.get_optimal_new_camera_matrix(alpha=alpha)
        map1, map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            new_K,
            self.image_size,
            m1type
        )
        self._map1 = map1
        self._map2 = map2
        self._new_camera_matrix = new_K
        return map1, map2

    def undistort(
        self,
        image: np.ndarray,
        use_cached_maps: bool = True,
        alpha: float = 0.0
    ) -> np.ndarray:
        """
        Undistort an image frame.

        Args:
            image: Input OpenCV BGR or grayscale image with shape (H, W) or (H, W, C).
            use_cached_maps: If True, uses cached initUndistortRectifyMap (significantly faster for live feeds).
            alpha: Crop factor (0.0 to 1.0) when precomputing maps.
        """
        if use_cached_maps:
            if self._map1 is None or self._map2 is None:
                self.init_undistort_rectify_map(alpha=alpha)
            return cv2.remap(image, self._map1, self._map2, cv2.INTER_LINEAR)

        new_K, _ = self.get_optimal_new_camera_matrix(alpha=alpha)
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_K)

    def summary(self) -> str:
        """Human-readable string summary of the calibration."""
        lines = [
            f"Camera Calibration Details:",
            f"  Name:          {self.camera_name}",
            f"  Camera Card:   {self.card_name or 'N/A'}",
            f"  Device Node:   {self.device_node or 'N/A'}",
            f"  Stream Type:   {self.stream_type or 'N/A'}",
            f"  Resolution:    {self.resolution_str} (width={self.width}, height={self.height})",
            f"  RMS Error:     {self.rms_error:.4f} px" if self.rms_error is not None else "  RMS Error:     N/A",
            f"  Dist. Model:   {self.distortion_model}",
            f"  NPZ Location:  {self.npz_path or 'N/A'}",
            f"  Source Path:   {self.source_path or 'In-memory'}",
            f"  Focal Length:  fx={self.fx:.2f}, fy={self.fy:.2f}",
            f"  Center (cx,cy):cx={self.cx:.2f}, cy={self.cy:.2f}",
            f"  Dist. Coeffs:  {np.round(self.dist_coeffs.ravel(), 5).tolist()}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        card_str = f", card='{self.card_name}'" if self.card_name else ""
        node_str = f", node='{self.device_node}'" if self.device_node else ""
        return (
            f"CameraCalibration(camera='{self.camera_name}'{card_str}{node_str}, "
            f"res={self.resolution_str}, "
            f"rms={f'{self.rms_error:.4f}px' if self.rms_error is not None else 'N/A'})"
        )


def _find_calibration_file(target: Union[str, Path]) -> Path:
    """Resolve a target directory or file into a concrete calibration file path."""
    p = Path(target).expanduser().resolve()
    if p.is_file():
        return p

    if p.is_dir():
        # Check standard files in order of preference
        for candidate in ["camera_calibration.npz", "camera_calibration.json", "camera_info.yaml"]:
            f = p / candidate
            if f.is_file():
                return f
        raise FileNotFoundError(
            f"No calibration file (camera_calibration.npz, camera_calibration.json, camera_info.yaml) "
            f"found inside directory '{p}'"
        )

    # Check under output/<target> if relative name given
    default_dir = Path("output") / target
    if default_dir.is_dir():
        return _find_calibration_file(default_dir)

    raise FileNotFoundError(f"Cannot find calibration file or directory: '{target}'")


def load_from_npz(npz_path: Union[str, Path], camera_name: Optional[str] = None) -> CameraCalibration:
    """Load calibration from NumPy .npz archive."""
    path = Path(npz_path).expanduser().resolve()
    data = np.load(path)

    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
    image_size_arr = data["image_size"]
    image_size = (int(image_size_arr[0]), int(image_size_arr[1]))
    rms_error = float(data["rms_error"]) if "rms_error" in data else None

    # Retrieve extra parameters if saved directly in npz
    card_name = str(data["card_name"]) if "card_name" in data and str(data["card_name"]).strip() else None
    device_node = str(data["device_node"]) if "device_node" in data and str(data["device_node"]).strip() else None
    stream_type = str(data["stream_type"]) if "stream_type" in data and str(data["stream_type"]).strip() else None
    loaded_cam_name = str(data["camera_name"]) if "camera_name" in data and str(data["camera_name"]).strip() else None

    # Derive camera name from parent folder, npz array, or argument
    derived_name = camera_name or loaded_cam_name or path.parent.name
    if derived_name in ("", "."):
        derived_name = path.stem

    # Also check companion JSON if present for extra metadata (distortion model name, config, device_info)
    metadata: Dict[str, Any] = {}
    distortion_model = "standard"
    companion_json = path.parent / "camera_calibration.json"
    if companion_json.is_file():
        try:
            with open(companion_json, "r") as f:
                metadata = json.load(f)
                derived_name = camera_name or metadata.get("camera_name", derived_name)
                distortion_model = metadata.get("calibration_info", {}).get("model", distortion_model)
                device_info = metadata.get("device_info", {})
                card_name = card_name or device_info.get("card_name") or metadata.get("config", {}).get("camera_card")
                device_node = device_node or device_info.get("device_node") or metadata.get("config", {}).get("device_node")
                stream_type = stream_type or device_info.get("stream_type") or metadata.get("config", {}).get("stream_type")
        except Exception:
            pass

    return CameraCalibration(
        camera_name=derived_name,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        rms_error=rms_error,
        distortion_model=distortion_model,
        card_name=card_name,
        device_node=device_node,
        stream_type=stream_type,
        npz_path=path,
        source_path=path,
        raw_metadata=metadata
    )


def load_from_json(json_path: Union[str, Path], camera_name: Optional[str] = None) -> CameraCalibration:
    """Load calibration from OpenCV JSON export."""
    path = Path(json_path).expanduser().resolve()
    with open(path, "r") as f:
        data = json.load(f)

    derived_name = camera_name or data.get("camera_name") or path.parent.name
    calib_info = data.get("calibration_info", {})
    w = int(calib_info.get("image_width", 0))
    h = int(calib_info.get("image_height", 0))
    rms_error = calib_info.get("rms_reprojection_error_px")
    model = calib_info.get("model", "standard")

    device_info = data.get("device_info", {})
    cfg = data.get("config", {})
    card_name = device_info.get("card_name") or cfg.get("camera_card")
    device_node = device_info.get("device_node") or cfg.get("device_node")
    stream_type = device_info.get("stream_type") or cfg.get("stream_type")

    matrix_data = data.get("camera_matrix", {})
    if "matrix_3x3" in matrix_data:
        K = np.array(matrix_data["matrix_3x3"], dtype=np.float64)
    else:
        fx = matrix_data.get("fx", 0.0)
        fy = matrix_data.get("fy", 0.0)
        cx = matrix_data.get("cx", 0.0)
        cy = matrix_data.get("cy", 0.0)
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    D = np.array(data.get("distortion_coefficients", []), dtype=np.float64)

    # Companion npz if exists
    npz_candidate = path.parent / "camera_calibration.npz"
    npz_path = npz_candidate if npz_candidate.is_file() else None

    return CameraCalibration(
        camera_name=derived_name,
        camera_matrix=K,
        dist_coeffs=D,
        image_size=(w, h),
        rms_error=float(rms_error) if rms_error is not None else None,
        distortion_model=model,
        card_name=card_name,
        device_node=device_node,
        stream_type=stream_type,
        npz_path=npz_path,
        source_path=path,
        raw_metadata=data
    )


def load_from_yaml(yaml_path: Union[str, Path], camera_name: Optional[str] = None) -> CameraCalibration:
    """Load calibration from ROS camera_info.yaml export."""
    path = Path(yaml_path).expanduser().resolve()
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    derived_name = camera_name or data.get("camera_name") or path.parent.name
    w = int(data.get("image_width", 0))
    h = int(data.get("image_height", 0))

    k_mat = data.get("camera_matrix", {})
    k_data = k_mat.get("data", [])
    K = np.array(k_data, dtype=np.float64).reshape((3, 3))

    d_mat = data.get("distortion_coefficients", {})
    d_data = d_mat.get("data", [])
    D = np.array(d_data, dtype=np.float64)

    model = data.get("distortion_model", "plumb_bob")
    card_name = data.get("camera_card")
    device_node = data.get("device_node")
    stream_type = data.get("stream_type")

    npz_candidate = path.parent / "camera_calibration.npz"
    npz_path = npz_candidate if npz_candidate.is_file() else None

    return CameraCalibration(
        camera_name=derived_name,
        camera_matrix=K,
        dist_coeffs=D,
        image_size=(w, h),
        rms_error=None,
        distortion_model=model,
        card_name=card_name,
        device_node=device_node,
        stream_type=stream_type,
        npz_path=npz_path,
        source_path=path,
        raw_metadata=data
    )


def load_calibration(
    path_or_dir: Union[str, Path],
    camera_name: Optional[str] = None,
    stream_type: Optional[str] = None
) -> CameraCalibration:
    """
    Main loader function. Auto-detects and loads calibration from:
      - .npz file
      - .json file
      - .yaml file
      - Directory containing calibration outputs (e.g. output/astra720b)
      - Camera profile name under 'output/<name>'
      - Camera card name query matching a profile in 'output/'

    Args:
        path_or_dir: Path to calibration file/folder or camera name/card query.
        camera_name: Optional override for camera identifier.
        stream_type: Optional stream type filter (e.g. 'RGB / Color', 'Depth', 'Infrared', 'Color (MJPEG)').
                     When specified, validates that the loaded profile matches this stream type,
                     or searches across profiles for the one matching both query and stream type.
    """
    calib: Optional[CameraCalibration] = None

    # Helper to check stream type compatibility
    def _matches_stream_type(c: CameraCalibration, req_type: str) -> bool:
        if not c.stream_type:
            return True
        r = req_type.strip().lower()
        s = c.stream_type.strip().lower()
        if r == s or r in s or s in r:
            return True
        if "depth" in r and "depth" in s:
            return True
        if any(k in r for k in ["ir", "infrared", "grey"]) and any(k in s for k in ["ir", "infrared", "grey"]):
            return True
        if any(k in r for k in ["color", "rgb", "mjpeg"]) and any(k in s for k in ["color", "rgb", "mjpeg"]):
            return True
        return False

    # 1. First try direct file or directory path resolution
    try:
        file_path = _find_calibration_file(path_or_dir)
        suffix = file_path.suffix.lower()

        if suffix == ".npz":
            calib = load_from_npz(file_path, camera_name=camera_name)
        elif suffix == ".json":
            calib = load_from_json(file_path, camera_name=camera_name)
        elif suffix in (".yaml", ".yml"):
            calib = load_from_yaml(file_path, camera_name=camera_name)

        if calib is not None:
            if stream_type and not _matches_stream_type(calib, stream_type):
                raise ValueError(
                    f"Calibration profile '{calib.camera_name}' (stream_type='{calib.stream_type}') "
                    f"does not match requested stream_type='{stream_type}'"
                )
            return calib
    except FileNotFoundError:
        pass

    # 2. If not found directly by path, search across available profiles in output/
    query = str(path_or_dir).lower()
    all_calibs = list_calibrations()

    # Filter by stream_type first if requested
    if stream_type:
        filtered = [c for c in all_calibs if _matches_stream_type(c, stream_type)]
    else:
        filtered = all_calibs

    # Check matches in filtered list
    for c in filtered:
        if query in c.camera_name.lower():
            return c
        if c.card_name and query in c.card_name.lower():
            return c
        if c.device_node and query in c.device_node.lower():
            return c

    # Fallback: if query matches stream type directly (e.g. load_calibration("realsense", stream_type="depth"))
    if stream_type:
        raise FileNotFoundError(
            f"Cannot find calibration profile matching '{path_or_dir}' with stream_type='{stream_type}'. "
            f"Available profiles: {[f'{c.camera_name} ({c.stream_type})' for c in all_calibs]}"
        )
    raise FileNotFoundError(f"Cannot find calibration profile matching: '{path_or_dir}'")



def list_calibrations(output_dir: Union[str, Path] = "output") -> List[CameraCalibration]:
    """Scan an output directory and return all discovered camera calibrations."""
    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        return []

    calibrations: List[CameraCalibration] = []
    for item in sorted(root.iterdir()):
        if item.is_dir():
            try:
                calib = load_calibration(item)
                calibrations.append(calib)
            except FileNotFoundError:
                continue
    return calibrations
