import os
import cv2
import contextlib
import numpy as np
import re
import struct
from pathlib import Path, PurePosixPath
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union, List, Optional
from urllib.parse import urlparse
from saltup.utils.misc import is_url, extract_extension_from_url
from saltup.utils.data.image.image_utils import ColorMode, ColorsBGR, Image, ImageFormat


@contextlib.contextmanager
def _ffmpeg_capture_options(options: Optional[Dict[str, object]]):
    """Temporarily set OpenCV's FFmpeg demuxer options around a capture open.

    OpenCV's FFmpeg backend reads the ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` env var
    when a ``VideoCapture`` is opened. We use it to pass ``ignore_editlist=1`` so
    the mov/mp4 demuxer ignores the edit list, mapping frame index N to the same
    picture a browser shows once the edit list is neutralized.

    The env var is process-global and only read at open time, so wrap **only**
    the ``cv2.VideoCapture(...)`` call. The previous value is restored on exit.
    """
    if not options:
        yield
        return
    key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    previous = os.environ.get(key)
    os.environ[key] = "|".join(f"{k};{v}" for k, v in options.items())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@dataclass
class VideoReadOptions:
    """Decode-time options for reading a video (format-specific knobs).

    Kept separate from the core read API so the function signatures stay
    format-agnostic and new knobs can be added without churning callers.

    Attributes:
        ignore_edit_list: MP4/MOV only. Tell FFmpeg's mov demuxer to ignore the
            edit list (``ignore_editlist``) so frame index N maps to the same
            picture a browser shows once the edit list is neutralized.
    """
    ignore_edit_list: bool = False

    def to_ffmpeg_capture_options(self) -> Dict[str, object]:
        """Translate to FFmpeg demuxer options for OPENCV_FFMPEG_CAPTURE_OPTIONS."""
        opts: Dict[str, object] = {}
        if self.ignore_edit_list:
            opts["ignore_editlist"] = 1
        return opts


def _open_capture(source, options: Optional[VideoReadOptions]):
    """Open a ``cv2.VideoCapture`` honoring *options* (FFmpeg backend when set)."""
    ffmpeg_opts = options.to_ffmpeg_capture_options() if options else {}
    backend = cv2.CAP_FFMPEG if ffmpeg_opts else cv2.CAP_ANY
    with _ffmpeg_capture_options(ffmpeg_opts or None):
        return cv2.VideoCapture(source, backend)

# =============================================================================
# Module Constants
# =============================================================================

_MIN_QUADRANTS = 2
_MAX_QUADRANTS = 16


def create_avi_from_jpg(folder: str, output_filename: str, fps: int = 4) -> None:
    """
    Creates an MJPEG video in an AVI container from JPEG images in a specified folder.

    Args:
        folder (str): Path to the folder containing the JPEG images.
        output_filename (str): Name of the output AVI video file.
        fps (int, optional): Frames per second for the output video. Defaults to 4.

    Returns:
        None
    """

    # Get a sorted list of JPEG files in the folder
    image_files: List[str] = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")])

    # Read the first image to get its dimensions
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Create a VideoWriter object with the specified output filename and FPS
    fourcc= cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
   
    if not video.isOpened():
        print("Error opening VideoWriter")
        exit()

    # Iterate through the image files and write each one as a frame in the video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Problem during image handling: {image_file}")
            continue
        video.write(frame)

    # Release the VideoWriter object to close the output video file
    video.release()


def convert_ts_to_mp4(input_path: str, output_path: str, input_file_ts: str) -> None:
    '''
    Converts a .ts video file to .mp4 format using FFmpeg.

    Args:
        input_path (str): Directory path of the input .ts video file.
        output_path (str): Directory path where the output .mp4 video will be saved.
        input_file_ts (str): Name of the .ts file to be converted.

    Returns:
        None
    '''
    # Output name
    output_file_ts = input_file_ts.replace("ts", "mp4")
    # Conversion
    subprocess.call(['ffmpeg', '-i', os.path.join(input_path, input_file_ts), "-c", "copy", os.path.join(output_path, output_file_ts)])


def extract_jpg_frames_from_video(
    video_path: str,  
    frames_output_dir: str = "", 
    overwrite: bool = False, 
    start_frame: int = -1, 
    end_frame: int = -1, 
    frame_interval: int = 1, 
    filename_prefix:str=""
) -> int:
    '''Extracts JPG frames from a video file.

    Args:
        video_path (str): Path to the source video file.
        frames_output_dir (str, optional): Destination directory for saving extracted frames.
            If not specified, uses the current working directory.
        overwrite (bool, optional): If True, overwrites any existing files with the same name.
            If False, skips extraction for frames that already have a corresponding file. Default False.
        start_frame (int, optional): Frame number to start extraction from.
            A value of -1 indicates starting from the beginning of the video. Default -1.
        end_frame (int, optional): Frame number to end extraction at.
            A value of -1 indicates continuing until the end of the video. Default -1. 
        frame_interval (int, optional): Frame extraction interval.
            For example, 1 saves every frame, 2 saves one frame every two frames, etc. Default 1.
        filename_prefix (str, optional): Prefix to add to each saved frame filename.
            The final filename format will be: {prefix}{video_filename}_{frame_number}.jpg. Default "".

    Returns:
        int: Total number of successfully saved frames.

    Raises:
        AssertionError: If the specified video file does not exist.
    '''

    if frames_output_dir == "" :
        frames_output_dir = os.getcwd()

    # Get the video path and filename from the path
    video_dir, video_filename = os.path.split(video_path)  
    # Assert the video file exists
    assert os.path.exists(video_path)  

    # Open the video using OpenCV
    capture = cv2.VideoCapture(video_path)  

    # If start isn't specified lets assume 0
    if start_frame < 0:  
        start_frame = 0
    # if end isn't specified assume the end of the video
    if end_frame < 0:
        end_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the starting frame of the capture
    capture.set(1, start_frame)
    # Keep track of which frame we are up to, starting from start
    frame = start_frame
    # A safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    while_safety = 0
    # A count of how many frames we have saved
    saved_count = 0

    # Loop through the frames until the end
    while frame < end_frame:

        # Read an image from the capture
        _, image = capture.read()
        # Break the while if our safety maxs out at 500
        if while_safety > 500: 
            break

        # Skip in case of ''None' value read
        # Not saving in case of bad return
        if image is None:
            # Add 1
            while_safety += 1
            # skip
            continue

        # If this is a frame, write out based on the 'every' argument
        if frame % frame_interval == 0:
            # Reset the safety count
            while_safety = 0
            
            # variable 'path' creation
            path = os.path.join(frames_output_dir, Path(video_filename).stem)
            # Check whether the specified path exists or not
            if not os.path.exists(path):
               # Create a new directory because it does not exist
               os.makedirs(path)
    
            # Create the save path
            save_path = os.path.join(frames_output_dir,Path(video_filename).stem, f"{filename_prefix}{video_filename}_{frame:05d}.jpg")
            # If it doesn't exist or you want to overwrite anyways
            if not os.path.exists(save_path) or overwrite:
                # Save the extracted image
                cv2.imwrite(save_path, image)
                # Increment counter by one
                saved_count += 1

        # Increment frame count
        frame += 1  

    # After the while has finished close the capture
    capture.release()  

    # Return the count of the images we saved
    return saved_count
 
 

@dataclass
class VideoProperties:
    fps: int
    total_frames: int
    width: int
    height: int

    def __iter__(self):
        """Support tuple unpacking: fps, total_frames, width, height = props."""
        yield self.fps
        yield self.total_frames
        yield self.width
        yield self.height


def get_video_properties(video_path: Union[str, Path], max_seconds: float = 15, *, options: Optional[VideoReadOptions] = None) -> VideoProperties:

    """
    Get video properties such as FPS, total frames, width, and height.
    Supports both local file paths and HTTP/HTTPS URLs (e.g. S3 presigned URLs).
    OpenCV uses FFmpeg internally, so URLs are opened directly without downloading.
    - For .ts files (local or remote), FPS is calculated manually using frame timestamps.
    - For other formats, use OpenCV's default implementation.

    Args:
        video_path: Local file path or HTTP/HTTPS URL (e.g. S3 presigned URL).
        max_seconds: Window (in seconds) used to sample frames when computing
            real FPS from PTS deltas (e.g. ``.ts`` format).  A value ``<= 0``
            uses a fixed 60-frame sample (default).  Positive values sample
            at most ``fps * max_seconds`` frames (minimum 60) so the full
            file is never downloaded.  Total frames are always estimated from
            container metadata (duration × real FPS), not by reading every frame.

    Returns:
        tuple: A tuple containing (fps, total_frames, width, height).
            float: The FPS (frames per second).
            int: The total number of frames (or frames counted within *max_seconds*
                when scanning is limited).
            int: The width of the video.
            int: The height of the video.
    """
    _is_url = is_url(video_path)

    if _is_url:
        video_source = video_path
    else:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_source = str(video_path)

    # List of formats that require manual FPS calculation
    custom_formats = ['.ts']

    # Open the video (OpenCV uses FFmpeg internally, supports both files and URLs).
    # options may force the FFmpeg backend with demuxer flags (e.g. ignore_editlist),
    # so total_frames/duration match the neutralized timeline.
    video = _open_capture(video_source, options)
    if not video.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    # Get width and height (usually reliable)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine the file suffix.
    # For presigned URLs, the path may be extension-less; in that case,
    # infer extension from response-content-disposition filename query param.
    if _is_url:
        suffix = extract_extension_from_url(str(video_path))
    else:
        suffix = Path(video_path).suffix.lower()

    # Millisecond budget for frame-scanning modes (None = unlimited)
    limit_ms = max_seconds * 1000.0 if max_seconds > 0 else None

    # If the format is in the custom_formats list, manually calculate FPS and total_frames
    if suffix in custom_formats:
        fps_container = video.get(cv2.CAP_PROP_FPS)
        fc_container  = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_seconds <= 0:
            # Full scan: read every frame for an accurate count and precise FPS.
            total_frames = 0
            frame_timestamps = []
            while True:
                ret, _ = video.read()
                if not ret:
                    break
                total_frames += 1
                frame_timestamps.append(video.get(cv2.CAP_PROP_POS_MSEC))
        else:
            # Partial scan: sample at most fps_container * max_seconds frames
            # (minimum 60) to compute real FPS, then estimate total_frames from
            # container duration so the full file is never downloaded.
            fallback_fps = fps_container if fps_container > 0 else 25.0
            sample_count = max(60, int(fallback_fps * max_seconds))
            frame_timestamps = []
            for _ in range(sample_count):
                ret, _ = video.read()
                if not ret:
                    break
                frame_timestamps.append(video.get(cv2.CAP_PROP_POS_MSEC))

        # Compute real FPS from PTS deltas (common to both paths)
        fps = 0
        if len(frame_timestamps) > 1:
            deltas = [
                frame_timestamps[i + 1] - frame_timestamps[i]
                for i in range(len(frame_timestamps) - 1)
                if frame_timestamps[i + 1] > frame_timestamps[i]
            ]
            if deltas:
                avg_ms = sum(deltas) / len(deltas)
                fps = round(1000.0 / avg_ms) if avg_ms > 0 else 0

        # Fall back to container FPS if PTS-based calculation failed
        if fps == 0 and fps_container > 0:
            fps = round(fps_container)

        # For partial scans estimate total_frames from container metadata
        # so we report the real duration, not just the sampled window.
        if max_seconds > 0:
            if fc_container > 0 and fps_container > 0:
                duration = fc_container / fps_container
                total_frames = int(duration * fps) if fps > 0 else fc_container
            else:
                total_frames = len(frame_timestamps)
        # For full scans total_frames was already counted in the loop above.
    else:
        # Use OpenCV's default implementation for other formats
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(video.get(cv2.CAP_PROP_FPS))

        # Fallback for streams where the container reports no frame count
        if total_frames <= 0 and fps > 0 and limit_ms is not None:
            # Count frames up to max_seconds by reading
            frame_timestamps = []
            while True:
                ret, _ = video.read()
                if not ret:
                    break
                total_frames += 1
                timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
                frame_timestamps.append(timestamp)
                if timestamp >= limit_ms:
                    break
            # Refine fps from real PTS deltas if we have enough samples
            if len(frame_timestamps) > 1:
                deltas = [
                    frame_timestamps[i + 1] - frame_timestamps[i]
                    for i in range(len(frame_timestamps) - 1)
                    if frame_timestamps[i + 1] > frame_timestamps[i]
                ]
                if deltas:
                    avg_ms = sum(deltas) / len(deltas)
                    fps = round(1000.0 / avg_ms) if avg_ms > 0 else fps

    video.release()
    return VideoProperties(fps=fps, total_frames=total_frames, width=width, height=height)

def _infer_codec_from_filename(filename: Union[str, Path]) -> str:
    """
    Infer the video codec based on the file extension.

    Args:
        filename: Path to the output video file.

    Returns:
        A string representing the fourcc codec.
    """
    extension = Path(filename).suffix.lower()
    codec_mapping = {
        '.avi': 'XVID',
        '.mp4': 'mp4v',
        '.mov': 'avc1',
        '.mkv': 'X264',
        '.ts': 'MPEG',   
     }
    return codec_mapping.get(extension, 'XVID')   


def process_video(
    video_input: Union[str, Path],
    callback: Callable[[Image, int, int], Image] = None,
    video_output: Union[str, Path] = None,
    metadata: VideoProperties = None,
    frame_numbers: Optional[List[int]] = None,
    *,
    options: Optional[VideoReadOptions] = None,
):
    """
    Process a video frame by frame, applying a callback to each frame.

    Args:
        video_input: Path to the input video.
        callback: Callback function that receives a frame (as Image), frame number, and total frame count.
        video_output: Path to the output video (optional).
        metadata: VideoProperties object containing video metadata (if not specified, it will be inferred).
        frame_numbers: List of specific frame numbers to process (e.g., [0, 10, 20, 150]).
                      If None, processes all frames sequentially.
        options: VideoReadOptions for decode-time knobs (e.g. ignore_edit_list).
                      When metadata is not supplied, the inferred properties are
                      read with the same options.

    Returns:
        None
    """
    # Open the input video (options may force FFmpeg + ignore_editlist).
    input_video = _open_capture(str(video_input), options)
    if not input_video.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_input}")

    # Get video properties
    if metadata is None:
        metadata = get_video_properties(video_input, options=options)
    input_fps, total_frames, width, height = metadata.fps, metadata.total_frames, metadata.width, metadata.height
    
    # Setup output video if specified
    if video_output:
        codec = _infer_codec_from_filename(video_output)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_fps = metadata.fps if metadata.fps is not None else input_fps
        out = cv2.VideoWriter(str(video_output), fourcc, output_fps, (width, height))
    else:
        out = None
 
    video_input_str = str(video_input)
    if is_url(video_input_str):
        input_suffix = extract_extension_from_url(video_input_str)
    else:
        input_suffix = Path(video_input_str).suffix.lower()
    is_ts_input = input_suffix == '.ts'
    
    if frame_numbers is not None:
        # Selective mode: use cheap frame grabs; only do coarse seeks on seek-friendly formats.
        frames_to_process = sorted(set(frame_numbers))
        current_frame = 0
        seek_gap_threshold = 300
        seek_backoff_frames = 30
 
        for frame_number in frames_to_process:
            if frame_number < 0:
                continue
            if frame_number >= total_frames:
                break
 
            gap = frame_number - current_frame
 
            # Hybrid strategy for large gaps: coarse seek near target, then grab forward.
            if not is_ts_input and gap > seek_gap_threshold:
                seek_to = max(0, frame_number - seek_backoff_frames)
                if input_video.set(cv2.CAP_PROP_POS_FRAMES, seek_to):
                    current_frame = seek_to
 
            # Advance to target by grabbing packets without decoding frame pixels.
            while current_frame < frame_number:
                if not input_video.grab():
                    break
                current_frame += 1
 
            # Stream ended while seeking forward.
            if current_frame != frame_number:
                break
 
            ret, frame = input_video.read()
            if not ret:
                break
 
            # We just consumed frame_number.
            current_frame += 1
 
            # Apply callback
            if callback:
                processed_frame = callback(Image(frame), frame_number, total_frames)
            else:
                processed_frame = Image(frame)
 
            # Write to output if specified
            if out is not None:
                out.write(processed_frame.get_data())
    else:
        # 📹 MODALITÀ SEQUENZIALE: processa tutti i frame (comportamento originale)
        frame_number = 0
        while input_video.isOpened():
            ret, frame = input_video.read()
            if not ret:
                break
            
            # Apply callback
            if callback:
                processed_frame = callback(Image(frame), frame_number, total_frames)
            else:
                processed_frame = Image(frame)
            
            # Write to output if specified
            if out is not None:
                out.write(processed_frame.get_data())
            
            frame_number += 1
    
    # Cleanup
    input_video.release()
    if out is not None:
        out.release()


# =============================================================================
# Frame Preprocessing
# =============================================================================

def preprocess_frame(
    frame: np.ndarray,
    resize: Optional[Tuple[int, int]] = None,
    gray: bool = True,
    blur: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    roi: Optional[Tuple[float, float, float, float]] = None # normalized coordinates (x_min, y_min, x_max, y_max) of the region of interest
) -> np.ndarray:
    """Apply a configurable preprocessing pipeline to a single video frame.

    Operations are applied in order: resize → grayscale → blur → normalize.
    Each step is optional and controlled by its corresponding argument.

    Args:
        frame: Input frame as a BGR NumPy array (H×W×3).
        resize: Target ``(width, height)`` for resizing.  ``None`` skips
            resizing.  Defaults to ``None``.
        gray: If ``True``, convert the frame to single-channel grayscale.
            Defaults to ``True``.
        blur: Gaussian blur kernel size as ``(kW, kH)`` (both must be odd).
            ``None`` skips blurring.  Defaults to ``None``.
        normalize: If ``True``, apply z-score normalization rescaled to
            the 0–255 uint8 range (mean ≈ 128, std ≈ 32).  Useful for
            reducing sensitivity to global illumination changes.
            Defaults to ``False``.
        roi: Normalized coordinates ``(x_min, y_min, x_max, y_max)`` of the region of interest.
            If specified, only this region will be processed. Defaults to ``None``.

    Returns:
        The preprocessed frame as a NumPy array.

    Examples:
        >>> import cv2, numpy as np
        >>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> gray_small = preprocess_frame(frame, resize=(320, 240), gray=True)
        >>> gray_small.shape
        (240, 320)
    """
    # if roi is specified, crop the frame to the region of interest
    if roi is not None:
        x1, y1, x2, y2 = roi
        h, w = frame.shape[:2]

        x1 = int(round(x1 * w))
        x2 = int(round(x2 * w))
        y1 = int(round(y1 * h))
        y2 = int(round(y2 * h))

        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))

        frame = frame[y1:y2, x1:x2]
        
    if resize is not None:
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur is not None:
        frame = cv2.GaussianBlur(frame, blur, 0)
    if normalize:
        frame_f = frame.astype(np.float32)
        mean = float(np.mean(frame_f))
        std = float(np.std(frame_f))
        if std > 1e-6:
            norm = (frame_f - mean) / std
            frame = np.clip(128.0 + 32.0 * norm, 0, 255).astype(np.uint8)
    return frame


# =============================================================================
# Quadrant Grid Helpers
# =============================================================================


def _compute_grid(n_quadrants: int) -> Tuple[int, int]:
    """Find the most square-like (rows, cols) grid for *n_quadrants*.

    The function returns the pair ``(rows, cols)`` where
    ``rows * cols == n_quadrants`` and the difference ``|rows - cols|``
    is minimised.

    Args:
        n_quadrants: Desired number of spatial regions (2–16).

    Returns:
        ``(rows, cols)`` tuple.

    Raises:
        ValueError: If *n_quadrants* is outside the 2–16 range.
    """
    if not (_MIN_QUADRANTS <= n_quadrants <= _MAX_QUADRANTS):
        raise ValueError(
            f"n_quadrants must be between {_MIN_QUADRANTS} and "
            f"{_MAX_QUADRANTS}, got {n_quadrants}."
        )
    best = (1, n_quadrants)
    for r in range(1, int(n_quadrants ** 0.5) + 1):
        if n_quadrants % r == 0:
            c = n_quadrants // r
            if abs(r - c) < abs(best[0] - best[1]):
                best = (r, c)
    return best


def _quadrant_names(n_quadrants: int) -> List[str]:
    """Return a list of canonical quadrant key names.

    Keys are ``'quadrant_1'`` … ``'quadrant_N'`` where *N* equals
    *n_quadrants*.  The ordering follows a **row-major, bottom-to-top**
    scan: the first quadrant is the bottom-right cell of the grid, and
    the last is the top-left cell.  For ``n_quadrants=4`` this matches
    the legacy numbering.
    """
    return [f"quadrant_{i + 1}" for i in range(n_quadrants)]

# =============================================================================
# Quadrant Intensity Analysis
# =============================================================================

def extract_quadrant_intensity(
    frames: List[np.ndarray],
    n_quadrants: int = 4,
) -> Dict[int, Dict[str, float]]:
    """Compute per-frame average pixel intensity in each spatial quadrant.

    The frame is divided into a ``rows × cols`` grid whose total cell
    count equals *n_quadrants*.  The grid shape is chosen to be as
    square as possible (see :func:`_compute_grid`).

    Quadrants are labelled ``'quadrant_1'`` … ``'quadrant_N'`` following
    a **row-major, bottom-to-top** scan (bottom-right first, top-left
    last).  For ``n_quadrants=4`` this is identical to the legacy
    convention.

    Args:
        frames: List of preprocessed frames (grayscale or colour NumPy
            arrays).  All frames must share the same spatial dimensions.
        n_quadrants: Number of spatial regions (2–16).  Defaults to 4.

    Returns:
        Dictionary mapping each frame index to a sub-dictionary of
        quadrant names → float energy values.

    Raises:
        ValueError: If *n_quadrants* is outside the 2–16 range.

    Examples:
        >>> import numpy as np
        >>> white = np.ones((100, 100), dtype=np.uint8) * 255
        >>> result = extract_quadrant_intensity([white])
        >>> result[0]['quadrant_1']
        255.0
        >>> result_9 = extract_quadrant_intensity([white], n_quadrants=9)
        >>> len(result_9[0])
        9
    """
    rows, cols = _compute_grid(n_quadrants)
    names = _quadrant_names(n_quadrants)

    if not frames:
        return {}

    h, w = frames[0].shape[:2]
    n_frames = len(frames)

    # Pre-compute row/col boundaries once
    row_edges = [int(round(r * h / rows)) for r in range(rows + 1)]
    col_edges = [int(round(c * w / cols)) for c in range(cols + 1)]

    # Pre-allocate one array per cell
    cell_arrays = [np.zeros(n_frames, dtype=np.float32) for _ in range(n_quadrants)]

    for i, frame in enumerate(frames):
        # Fill cells in bottom-to-top, right-to-left order to match legacy naming
        cell_idx = 0
        for r in reversed(range(rows)):
            for c in reversed(range(cols)):
                cell_arrays[cell_idx][i] = np.mean(
                    frame[row_edges[r]:row_edges[r + 1], col_edges[c]:col_edges[c + 1]]
                )
                cell_idx += 1

    energies: Dict[int, Dict[str, float]] = {
        i: {name: float(cell_arrays[ci][i]) for ci, name in enumerate(names)}
        for i in range(n_frames)
    }
    return energies


def extract_quadrant_motion(
    frames: List[np.ndarray],
    n_quadrants: int = 4,
) -> Dict[int, Dict[str, float]]:
    """Compute per-frame intensity using absolute frame differences.
    Instead of raw pixel intensity (see :func:`extract_quadrant_intensity`),
    this function measures *change* between consecutive frames.  The
    first frame always yields zero intensity (no previous frame to diff
    against).  This approach reduces sensitivity to slow global
    illumination changes and highlights motion.

    Args:
        frames: List of preprocessed frames.  All frames must share the
            same spatial dimensions.
        n_quadrants: Number of spatial regions (2–16).  Defaults to 4.

    Returns:
        Same structure as :func:`extract_quadrant_intensity` but values
        represent the mean absolute difference per quadrant.

    Raises:
        ValueError: If *n_quadrants* is outside the 2–16 range.
    """
    rows, cols = _compute_grid(n_quadrants)
    names = _quadrant_names(n_quadrants)

    if not frames:
        return {}

    h, w = frames[0].shape[:2]
    n_frames = len(frames)

    row_edges = [int(round(r * h / rows)) for r in range(rows + 1)]
    col_edges = [int(round(c * w / cols)) for c in range(cols + 1)]

    cell_arrays = [np.zeros(n_frames, dtype=np.float32) for _ in range(n_quadrants)]

    prev = frames[0]
    for i in range(n_frames):
        frame = frames[i]
        diff = np.zeros_like(frame) if i == 0 else cv2.absdiff(frame, prev)

        # Bottom-to-top, right-to-left scan (matches legacy naming)
        cell_idx = 0
        for r in reversed(range(rows)):
            for c in reversed(range(cols)):
                cell_arrays[cell_idx][i] = np.mean(
                    diff[row_edges[r]:row_edges[r + 1], col_edges[c]:col_edges[c + 1]]
                )
                cell_idx += 1
        prev = frame

    energies: Dict[int, Dict[str, float]] = {
        i: {name: float(cell_arrays[ci][i]) for ci, name in enumerate(names)}
        for i in range(n_frames)
    }
    return energies

def extract_quadrant_variance(
    frames: List[np.ndarray],
    n_quadrants: int = 4,
    fps: float = 30.0,
) -> Dict[int, Dict[str, float]]:

    rows, cols = _compute_grid(n_quadrants)
    names = _quadrant_names(n_quadrants)
    PIXEL_K = 4.5
    WINDOW_SECONDS = 1.5
    MOVE_WINDOW_SEC = 1
    MIN_SECONDS = 0.0
    SMOOTH_SEC = 0.5
    MOVE_THRESHOLD = 8.0

    def cell_slices(H, W):
        # (y0, y1, x0, x1) for each cell, row-major (top-left first)
        ys = np.linspace(0, H, rows + 1).round().astype(int)
        xs = np.linspace(0, W, cols + 1).round().astype(int)
        return [(ys[r], ys[r + 1], xs[c], xs[c + 1])
                for r in range(rows) for c in range(cols)]
    
    def moving_mask(m, pixel_k):
        # A pixel is "moving" if its temporal std (m = the std_map) is clearly ABOVE the frame's own
        # noise level. That level is estimated robustly, per frame:
        #   med = median(m)                      -> the TYPICAL value (~background noise); most pixels
        #                                           are static so the median ignores the few moving ones
        #   MAD = median(|m - med|)              -> Median Absolute Deviation = a robust "spread"
        #   mad = MAD * 1.4826                    -> rescaled so it equals a standard deviation for
        #                                           Gaussian data -> pixel_k is then "number of sigmas"
        #   threshold = med + pixel_k * mad      -> typical level + pixel_k robust-sigmas
        # median/MAD (not mean/std) are used so the bright moving pixels don't inflate the threshold.
        # A pixel is moving when m > threshold. Self-adapting per frame; fully causal (this frame only).
        med = np.median(m); mad = np.median(np.abs(m - med)) * 1.4826 + 1e-6
        return m > (med + pixel_k * mad)
    
 


    N = max(2, int(round(WINDOW_SECONDS * fps)))
    alpha = 1.0 / N

    mean_acc = sq_acc = None

    centroids, ncent = [], []
    _openk = np.ones((3, 3), np.uint8)               # for morphological opening (denoise mask)

    for frame in frames:
        # brightness compensation PER CELL: subtract each cell's own mean, so a light change or
        # motion in ONE cell does not leak into the others through a shared global mean.
        for (y0, y1, x0, x1) in cell_slices(*frame.shape):
            frame[y0:y1, x0:x1] -= frame[y0:y1, x0:x1].mean()


        # TEMPORAL MOVEMENT MATRIX (std_map): per pixel, keep an EMA of the mean (mean_acc) and of
        # the mean-of-squares (sq_acc) over ~WINDOW_SECONDS (alpha = 1/N). Then, per pixel,
        # variance = mean(x^2) - mean(x)^2  and  std = sqrt(variance). std_map is high where a pixel
        # kept changing during the window, ~0 where static. O(1) memory, no frame buffer.
        if mean_acc is None:
                mean_acc = frame.copy(); sq_acc = frame * frame          # bootstrap on the first frame
        else:
            cv2.accumulateWeighted(frame,        mean_acc, alpha)  # EMA of the mean
            cv2.accumulateWeighted(frame * frame, sq_acc,   alpha)  # EMA of the mean of squares
        std_map = np.sqrt(np.clip(sq_acc - mean_acc * mean_acc, 0, None))

        # SPATIAL movement: find the CENTRE of the moving pixels in each cell via its row/column
        # projections. We just record the centre here; the movement signal (how far the centre
        # shifts over ~1 s) is computed after the loop -> no per-frame speed, no double smoothing.
        mask = moving_mask(std_map, PIXEL_K)
        # morphological OPEN: drop isolated noise pixels, keep real blobs -> the centre doesn't
        # jitter on scattered noise, so an empty cell stays put (no false movement).
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _openk).astype(bool)
        cnorm = [(np.nan, np.nan)] * n_quadrants                  # normalized centre (0..1) per cell
        cframe = [(np.nan, np.nan)] * n_quadrants                 # absolute centre (x, y) for the player
        for qi, (y0, y1, x0, x1) in enumerate(cell_slices(*mask.shape)):
            m = mask[y0:y1, x0:x1]; tot = float(m.sum()); qh, qw = y1 - y0, x1 - x0
            if tot < max(15.0, 0.015 * qh * qw):         # too few moving pixels -> undefined
                continue
            ex = float((np.arange(qw) * m.sum(0)).sum() / tot)   # column profile -> x of centre
            ey = float((np.arange(qh) * m.sum(1)).sum() / tot)   # row profile    -> y of centre
            cnorm[qi]  = (ex / qw, ey / qh)
            cframe[qi] = (x0 + ex, y0 + ey)
        ncent.append(cnorm)
        centroids.append(cframe)        # activity centre per cell (or NaN)
    nc = np.asarray(ncent, dtype=np.float32).reshape(-1, n_quadrants, 2)
    Wm = max(1, int(round(MOVE_WINDOW_SEC * fps)))
    signal = np.zeros((len(nc), n_quadrants), dtype=np.float32)
    for qi in range(n_quadrants):
        a = nc[:, qi]; b = np.roll(a, Wm, axis=0); b[:Wm] = np.nan
        d = 100.0 * np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])
        d[np.isnan(d)] = 0.0
        signal[:, qi] = d   


    # signal: (T, NQ) = per-cell MOVEMENT (% of the cell the activity centre shifted over ~1 s,
    # from process()). Causal:
    #   * trailing-EMA smoothing (past only),
    #   * a cell is ACTIVE when its smoothed movement crosses `move_threshold`, and stays on
    #     until it drops below half that (hysteresis); bursts shorter than min_seconds dropped.
    # A flickering-in-place region barely shifts the centre, so it reads ~0 and does not trip;
    # only genuine spatial movement does.
    T,Q = signal.shape
    ws = max(1, int(round(SMOOTH_SEC * fps))); a_s = 1.0 / ws     # trailing smoothing (lower = less lag)
    min_len = int(round(MIN_SECONDS * fps))
    hi = float(MOVE_THRESHOLD); lo = 0.5 * hi

    smooth = np.empty_like(signal)                    # causal trailing EMA
    for qi in range(Q):
        e = float(signal[0, qi])
        for i in range(T):
            e += a_s * (float(signal[i, qi]) - e); smooth[i, qi] = e

    energies: Dict[int, Dict[str, float]] = {
        i: {name: float(smooth[i, ci]) for ci, name in enumerate(names)}
        for i in range(T)
    }
    return energies
# =============================================================================
# Windowed Segment Aggregation & Filtering
# =============================================================================

def compute_windowed_segment_stats(
    signal: np.ndarray,
    segments: List[Tuple[float, float]],
    fps: float,
    window_size_seconds: float = 10.0,
    agg_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> List[np.ndarray]:
    """Downsample a per-frame signal within each time segment.

    For every ``(start, end)`` segment the function slices the
    corresponding frames from *signal*, splits them into
    non-overlapping windows of *window_size_seconds*, and reduces each
    window to a single scalar via *agg_fn*.

    This is a generic building block — it knows nothing about quadrants
    or variance; it simply aggregates any 1-D per-frame signal over
    time windows.

    Args:
        signal: 1-D NumPy array of per-frame values (length = total
            number of frames in the source video).
        segments: List of ``(start_seconds, end_seconds)`` tuples
            defining the time intervals to process.
        fps: Video frame rate (frames per second).
        window_size_seconds: Duration (seconds) of each non-overlapping
            aggregation window.  Defaults to 10.0.
        agg_fn: Callable that reduces a 1-D window array to a single
            float.  Receives an ``np.ndarray`` and must return a float.
            Defaults to ``np.mean`` when ``None``.

    Returns:
        A list of 1-D NumPy arrays (one per segment) containing the
        aggregated values.

    Examples:
        >>> import numpy as np
        >>> # 300 frames at 30 fps = 10 seconds of data
        >>> signal = np.random.rand(300)
        >>> segments = [(0.0, 10.0)]
        >>> result = compute_windowed_segment_stats(signal, segments, fps=30, window_size_seconds=5.0)
        >>> len(result)          # one segment
        1
        >>> result[0].shape[0]   # 10s / 5s = 2 windows
        2
    """
    if agg_fn is None:
        agg_fn = lambda w: float(np.mean(w))

    window_size_frames = max(1, int(window_size_seconds * fps))
    results: List[np.ndarray] = []

    for start, end in segments:
        start_frame = int(start * fps)
        end_frame = min(int(end * fps) + 1, len(signal))

        segment_data = signal[start_frame:end_frame]
        if len(segment_data) == 0:
            continue

        aggregated: List[float] = []
        for win_start in range(0, len(segment_data), window_size_frames):
            window = segment_data[win_start : win_start + window_size_frames]
            if len(window) > 0:
                aggregated.append(agg_fn(window))

        results.append(np.array(aggregated))

    return results


def filter_segments_by_std(
    segments: List[Tuple[float, float]],
    segment_signals: List[np.ndarray],
    std_threshold: float = 0.01,
) -> Tuple[List[Tuple[float, float]], List[np.ndarray]]:
    """Discard segments whose associated signal has low variability.

    Pairs of ``(segment, signal)`` are dropped when the standard
    deviation of *signal* falls below *std_threshold*.  This is useful
    for removing near-constant regions that are unlikely to contain
    meaningful activity.

    Args:
        segments: Time segments as ``(start, end)`` tuples.
        segment_signals: One 1-D NumPy array per segment (e.g. output
            of :func:`compute_windowed_segment_stats`).
        std_threshold: Minimum standard deviation to keep a segment.
            Defaults to 0.01.

    Returns:
        A 2-tuple ``(kept_segments, kept_signals)``.

    Examples:
        >>> import numpy as np
        >>> segs = [(0, 10), (20, 30)]
        >>> sigs = [np.array([0.5, 0.5, 0.5]), np.array([0.1, 0.9, 0.3])]
        >>> kept_segs, kept_sigs = filter_segments_by_std(segs, sigs, std_threshold=0.1)
        >>> len(kept_segs)  # first segment (constant) is dropped
        1
    """
    kept_segments: List[Tuple[float, float]] = []
    kept_signals: List[np.ndarray] = []

    for i, sig in enumerate(segment_signals):
        if np.std(sig) >= std_threshold:
            kept_segments.append(segments[i])
            kept_signals.append(sig)

    return kept_segments, kept_signals

# =============================================================================
# Header Parsing Helpers
# =============================================================================

from saltup.utils.data.image.image_utils import FileExtensionType

def get_header(path: Union[str, Path]) -> bytes:
    """
    Reads the first 256 KB of a file for header parsing.
    This is sufficient for formats like WAV, FLAC, and MP3 which have headers within this range.
    """
    file_path = Path(path)
    extension_name = file_path.suffix.lower().lstrip(".")

    read_sizes = {
        "micro": 64 * 1024,
        "small": 2000 * 1024,
        "medium": 5000 * 1024,
    }

    first_64kb_ext = {
        FileExtensionType.WMV,
        FileExtensionType.FLV,
        
    }

    first_2mb_ext = {
        FileExtensionType.AVI,
        FileExtensionType.MKV,
        FileExtensionType.WEBM,
    }
    first_5mb_ext = {
        FileExtensionType.MP4,
        FileExtensionType.MOV,
        FileExtensionType.GP,
        FileExtensionType.M3U8
    }

    try:
        extension = FileExtensionType(extension_name)
    except ValueError:
        extension = None

    if extension in first_64kb_ext:
        with open(file_path, "rb") as file:
            return file.read(read_sizes["micro"])

    if extension in first_2mb_ext:
        with open(file_path, "rb") as file:
            return file.read(read_sizes["small"])

    if extension in first_5mb_ext:
        with open(file_path, "rb") as file:
            return file.read(read_sizes["medium"])

    with open(file_path, "rb") as file:
        return file.read(4)


def get_tail(path: Union[str, Path]) -> bytes:
    """
    Reads the last 256 KB of a file for footer parsing.
    This is useful for formats that may have important metadata at the end of the file.
    """
    file_path = Path(path)
    extension_name = file_path.suffix.lower().lstrip(".")

    read_sizes = {
        "medium": 5000 * 1024,
    }
    last_5mb_ext = {
        FileExtensionType.MP4,
        FileExtensionType.MOV,
    }

    try:
        extension = FileExtensionType(extension_name)
    except ValueError:
        extension = None

    if extension in last_5mb_ext:
        with open(file_path, "rb") as file:
            file_size = file.seek(0, os.SEEK_END)
            file.seek(-min(read_sizes["medium"], file_size), os.SEEK_END)
            return file.read()

    with open(file_path, "rb") as file:
        file_size = file.seek(0, os.SEEK_END)
        file.seek(-min(4, file_size), os.SEEK_END)
        return file.read(4)

def parse_avi_header(header: bytes) -> dict:
    """
    Parses the header of an AVI file to extract metadata such as format, resolution, and duration.
    This is a simplified parser that looks for specific byte patterns in the header.
    """
    metadata = {
        "format": "AVI",
        "width": None,
        "height": None,
        "fps": None,
        "bit_depth": None,
    }

    # AVI is a RIFF container with 'AVI ' as form type
    if len(header) >= 12 and header[0:4] == b'RIFF' and header[8:12] == b'AVI ':
        metadata["format"] = "AVI"
        # try to find 'avih' chunk and extract dwWidth/dwHeight
        idx = header.find(b'avih')
        if idx != -1 and idx + 8 + 40 <= len(header):
            # avih: 4 bytes size after 'avih', then data; width at offset 32 within data
            data_start = idx + 8
            try:
                # dwMicroSecPerFrame (uSec per frame) at offset 0
                dwMicroSecPerFrame = int.from_bytes(header[data_start + 0:data_start + 4], 'little')
                if dwMicroSecPerFrame > 0:
                    metadata["fps"] = 1_000_000.0 / float(dwMicroSecPerFrame)

                # dwTotalFrames at offset 16 (total frames in file)
                try:
                    total_frames = int.from_bytes(header[data_start + 16:data_start + 20], 'little')
                    metadata["total_frames"] = total_frames
                except Exception:
                    pass

                width = int.from_bytes(header[data_start + 32:data_start + 36], 'little')
                height = int.from_bytes(header[data_start + 36:data_start + 40], 'little')
                metadata["width"] = width
                metadata["height"] = height
            except Exception:
                pass
        return metadata

    return {"format": "AVI", "error": "Invalid AVI/RIFF header"}


def parse_mp4_header(header: bytes) -> dict:
    """
    Parses the header of an MP4 file to extract metadata such as format, resolution, and duration.
    This is a simplified parser that looks for specific byte patterns in the header.
    """
    metadata = {
        "format": "MP4",
        "width": None,
        "height": None,
        "fps": None,
        "bit_depth": None,
        "duration": None,
    }

    # MP4 files start with 'ftyp' box within the first few bytes
    if b'ftyp' not in header:
        return {"format": "MP4", "error": "Invalid MP4 signature"}

    metadata["format"] = "MP4"

    # Try to find 'moov' box which contains metadata; it may not be in the header if it's at the end of the file
    if b'moov' in header:
        # This is still a naive approach; proper MP4 parsing requires box sizes and nesting
        moov_idx = header.find(b'moov')
        if moov_idx != -1:
            # Look for 'mvhd' box inside 'moov' which contains duration and timescale
            mvhd_idx = header.find(b'mvhd', moov_idx)
            if mvhd_idx != -1 and mvhd_idx + 8 <= len(header):
                try:
                    # mvhd version is at mvhd_idx + 4 (mvhd type + fullbox version)
                    version = header[mvhd_idx + 4]
                    if version == 0 and mvhd_idx + 24 <= len(header):
                        # For mvhd v0: timescale @ +16, duration @ +20 (from type offset)
                        timescale = int.from_bytes(header[mvhd_idx + 16:mvhd_idx + 20], 'big')
                        duration = int.from_bytes(header[mvhd_idx + 20:mvhd_idx + 24], 'big')
                    elif version == 1 and mvhd_idx + 40 <= len(header):
                        # For mvhd v1: timescale @ +28, duration @ +32 (64-bit)
                        timescale = int.from_bytes(header[mvhd_idx + 28:mvhd_idx + 32], 'big')
                        duration = int.from_bytes(header[mvhd_idx + 32:mvhd_idx + 40], 'big')
                    else:
                        timescale = None
                        duration = None

                    if timescale and duration is not None:
                        metadata["duration"] = float(duration) / float(timescale)
                except Exception:
                    pass

            # Look for 'trak' boxes which contain track info; we want the video track
            trak_idx = header.find(b'trak', moov_idx)
            while trak_idx != -1:
                next_trak_idx = header.find(b'trak', trak_idx + 4)
                trak_end = next_trak_idx if next_trak_idx != -1 else len(header)

                # Look for 'tkhd' box inside 'trak' which contains width/height
                tkhd_idx = header.find(b'tkhd', trak_idx)
                if tkhd_idx != -1 and tkhd_idx + 8 <= len(header):
                    try:
                        version = header[tkhd_idx + 4]
                        if version == 0 and tkhd_idx + 88 <= len(header):
                            # tkhd_idx points to box type ('tkhd'), so width/height are at +80/+84
                            width = int.from_bytes(header[tkhd_idx + 80:tkhd_idx + 84], 'big') >> 16
                            height = int.from_bytes(header[tkhd_idx + 84:tkhd_idx + 88], 'big') >> 16
                        elif version == 1 and tkhd_idx + 100 <= len(header):
                            # version 1 has larger timestamps; width/height shift by +12 bytes
                            width = int.from_bytes(header[tkhd_idx + 92:tkhd_idx + 96], 'big') >> 16
                            height = int.from_bytes(header[tkhd_idx + 96:tkhd_idx + 100], 'big') >> 16
                        else:
                            width = None
                            height = None

                        if width and height:
                            metadata["width"] = width
                            metadata["height"] = height
                    except Exception:
                        pass

                # Try to compute FPS from track timing (mdhd) and sample count (stsz)
                if metadata.get("fps") is None:
                    mdhd_idx = header.find(b'mdhd', trak_idx, trak_end)
                    stsz_idx = header.find(b'stsz', trak_idx, trak_end)
                    if mdhd_idx != -1 and stsz_idx != -1:
                        try:
                            track_timescale = None
                            track_duration = None

                            # mdhd version is at mdhd_idx + 4
                            mdhd_version = header[mdhd_idx + 4] if mdhd_idx + 5 <= len(header) else None
                            if mdhd_version == 0 and mdhd_idx + 24 <= len(header):
                                # mdhd v0: timescale @ +16, duration @ +20 (from type offset)
                                track_timescale = int.from_bytes(header[mdhd_idx + 16:mdhd_idx + 20], 'big')
                                track_duration = int.from_bytes(header[mdhd_idx + 20:mdhd_idx + 24], 'big')
                            elif mdhd_version == 1 and mdhd_idx + 40 <= len(header):
                                # mdhd v1: timescale @ +28, duration @ +32 (64-bit)
                                track_timescale = int.from_bytes(header[mdhd_idx + 28:mdhd_idx + 32], 'big')
                                track_duration = int.from_bytes(header[mdhd_idx + 32:mdhd_idx + 40], 'big')

                            # stsz: sample_count @ +12 (from type offset)
                            sample_count = None
                            if stsz_idx + 16 <= len(header):
                                sample_count = int.from_bytes(header[stsz_idx + 12:stsz_idx + 16], 'big')

                            if (
                                track_timescale is not None
                                and track_duration is not None
                                and track_timescale > 0
                                and track_duration > 0
                                and sample_count is not None
                                and sample_count > 0
                            ):
                                duration_seconds = float(track_duration) / float(track_timescale)
                                if duration_seconds > 0:
                                    metadata["fps"] = int(round((float(sample_count) / duration_seconds), 0))
                        except Exception:
                            pass

                # Look for next 'trak' box
                trak_idx = header.find(b'trak', trak_idx + 4)

    return metadata

def parse_mov_header(header: bytes) -> dict:
    """
    Parses the header of a MOV file to extract metadata such as format, resolution, and duration.
    This is a simplified parser that looks for specific byte patterns in the header.
    """
    # MOV files are structurally similar to MP4 (both are based on the ISO Base Media File Format)
    # We can reuse the MP4 parsing logic with minor adjustments if needed
    metadata = parse_mp4_header(header)
    if "error" in metadata:
        return {"format": "MOV", "error": "Invalid MOV/MP4 signature"}
    
    metadata["format"] = "MOV"
    return metadata

def parse_video_header(path:Union[str, Path]) -> dict:
    """
    Parses the video header to extract metadata such as format, resolution, and duration.
    This function dispatches to specific parsers based on the detected format.
    """
    p = Path(path)
    extension_name = p.suffix.lower().lstrip(".")

    try:
        extension = FileExtensionType(extension_name)
    except ValueError:
        return {"error": f"Unsupported extension: {extension_name or 'none'}"}
    try:
        data = get_header(p)
    except Exception as exc:
        return {"error": f"Cannot read file: {exc}"}
    if extension == FileExtensionType.AVI:
        return parse_avi_header(data)
    if extension == FileExtensionType.MP4:
        if b"moov" in data:
            return parse_mp4_header(data)
        try:
            tail = get_tail(p)
        except Exception:
            tail = b""
        return parse_mp4_header(data + tail)
    if extension == FileExtensionType.MOV:
        if b"moov" in data:
            return parse_mov_header(data)
        try:
            tail = get_tail(p)
        except Exception:
            tail = b""
        return parse_mov_header(data + tail)

    return {"error": "Unsupported or unknown video format"}