import os
import cv2
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


def get_video_properties(video_path: Union[str, Path], max_seconds: float = 0) -> VideoProperties:

    """
    Get video properties such as FPS, total frames, width, and height.
    Supports both local file paths and HTTP/HTTPS URLs (e.g. S3 presigned URLs).
    OpenCV uses FFmpeg internally, so URLs are opened directly without downloading.
    - For .ts files (local or remote), FPS is calculated manually using frame timestamps.
    - For other formats, use OpenCV's default implementation.

    Args:
        video_path: Local file path or HTTP/HTTPS URL (e.g. S3 presigned URL).
        max_seconds: Maximum number of seconds to analyse when frame-level scanning
            is required (e.g. ``.ts`` format).  A value ``<= 0`` means scan the
            entire video (default behaviour).  Positive values stop scanning after
            *max_seconds* seconds, which avoids downloading the full stream.
            For formats that rely on container metadata (non-``.ts``), this parameter
            has no effect unless ``CAP_PROP_FRAME_COUNT`` is unavailable (streams),
            in which case frames are counted up to *max_seconds*.

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

    # Open the video (OpenCV uses FFmpeg internally, supports both files and URLs)
    video = cv2.VideoCapture(video_source)
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
        total_frames = 0
        frame_timestamps = []  # Store frame timestamps to calculate FPS
        while True:
            ret, _ = video.read()
            if not ret:
                break
            total_frames += 1
            # Get the current frame's timestamp
            timestamp = video.get(cv2.CAP_PROP_POS_MSEC)  # Timestamp in milliseconds
            frame_timestamps.append(timestamp)
            # Stop early if max_seconds is set
            if limit_ms is not None and timestamp >= limit_ms:
                break

        # Manually calculate FPS using frame timestamps
        if len(frame_timestamps) > 1:
            time_diff = (frame_timestamps[-1] - frame_timestamps[0]) / 1000.0  # Time difference in seconds
            fps = total_frames / time_diff if time_diff > 0 else 0
        else:
            fps = 0

        # Round FPS to the nearest integer
        fps = round(fps)
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
    fps: int = None,
    frame_numbers: Optional[List[int]] = None
):
    """
    Process a video frame by frame, applying a callback to each frame.
    
    Args:
        video_input: Path to the input video.
        callback: Callback function that receives a frame (as Image), frame number, and total frame count.
        video_output: Path to the output video (optional).
        fps: FPS of the output video (if not specified, uses input FPS).
        frame_numbers: List of specific frame numbers to process (e.g., [0, 10, 20, 150]).
                      If None, processes all frames sequentially.
    
    Returns:
        None
    """
    # Open the input video
    input_video = cv2.VideoCapture(str(video_input))
    if not input_video.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_input}")
    
    # Get video properties
    input_fps, total_frames, width, height = get_video_properties(video_input)
    
    # Setup output video if specified
    if video_output:
        codec = _infer_codec_from_filename(video_output)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_fps = fps if fps is not None else input_fps
        out = cv2.VideoWriter(str(video_output), fourcc, output_fps, (width, height))
    else:
        out = None
    
    if frame_numbers is not None:
        # 🚀 MODALITÀ SELETTIVA: salta ai frame specifici
        frames_to_process = sorted(set(frame_numbers))
        
        for frame_number in frames_to_process:
            if frame_number >= total_frames:
                continue
                
            # Salta al frame desiderato
            input_video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = input_video.read()
            
            if not ret:
                continue
            
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

    Returns:
        The preprocessed frame as a NumPy array.

    Examples:
        >>> import cv2, numpy as np
        >>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> gray_small = preprocess_frame(frame, resize=(320, 240), gray=True)
        >>> gray_small.shape
        (240, 320)
    """
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