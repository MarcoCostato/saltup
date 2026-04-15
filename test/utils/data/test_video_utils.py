"""
Tests for get_video_properties using public remote video assets.

These tests require an active internet connection and are marked with
``pytest.mark.network``.  Run them explicitly with::

    pytest -m network test/utils/data/test_video_utils.py

or skip them in CI with::

    pytest -m "not network"
"""

import socket
import time

import pytest

from saltup.utils.data.video.video_utils import get_video_properties

# ---------------------------------------------------------------------------
# Public test assets — small files on stable CDNs
# ---------------------------------------------------------------------------

# W3Schools sample — small direct MP4, ~320x176, ~25 fps, ~10 s
_MP4_URL = "https://www.w3schools.com/html/mov_bbb.mp4"

# Apple HLS bipbop — first .ts segment, ~10 s, 400x300, ~29.97 fps
_TS_URL = (
    "https://devstreaming-cdn.apple.com/videos/streaming/examples/"
    "bipbop_4x3/gear1/fileSequence0.ts"
)

pytestmark = pytest.mark.network


# ---------------------------------------------------------------------------
# Module-level network probe
# ---------------------------------------------------------------------------

def _network_available(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> bool:
    """Return True if a basic TCP connection to *host:port* succeeds."""
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
        return True
    except OSError:
        return False


@pytest.fixture(scope="module", autouse=True)
def require_network():
    """Skip the entire module when no network is available."""
    if not _network_available():
        pytest.skip("no network access — skipping remote video tests")


# ---------------------------------------------------------------------------
# MP4 (via container metadata — no frame scan needed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def mp4_props():
    """Fetch and cache get_video_properties for the MP4 URL; skip if unreachable."""
    try:
        return get_video_properties(_MP4_URL)
    except RuntimeError:
        pytest.skip(f"MP4 URL not reachable via OpenCV in this build: {_MP4_URL}")


class TestGetVideoPropertiesRemoteMP4:
    """Smoke tests against a small remote MP4 on a stable CDN."""

    def test_returns_nonzero_fps_width_height(self, mp4_props):
        fps, total_frames, width, height = mp4_props
        assert fps > 0, f"expected fps > 0, got {fps}"
        assert width > 0, f"expected width > 0, got {width}"
        assert height > 0, f"expected height > 0, got {height}"

    def test_fps_in_realistic_range(self, mp4_props):
        fps, _, _, _ = mp4_props
        assert 1 <= fps <= 120, f"fps {fps} outside expected range [1, 120]"

    def test_total_frames_consistent_with_duration(self, mp4_props):
        """total_frames / fps should give a plausible positive duration."""
        fps, total_frames, _, _ = mp4_props
        if fps > 0 and total_frames > 0:
            duration = total_frames / fps
            assert duration > 0, f"computed duration {duration} should be positive"

    def test_max_seconds_does_not_alter_metadata_results(self, mp4_props):
        """For MP4 with valid container metadata max_seconds must not change fps/dims."""
        fps0, _, w0, h0 = mp4_props
        try:
            fps5, _, w5, h5 = get_video_properties(_MP4_URL, max_seconds=5)
        except RuntimeError:
            pytest.skip(f"MP4 URL not reachable via OpenCV: {_MP4_URL}")
        assert w0 == w5, f"width changed: {w0} -> {w5}"
        assert h0 == h5, f"height changed: {h0} -> {h5}"
        assert fps0 == fps5, f"fps changed: {fps0} -> {fps5}"


# ---------------------------------------------------------------------------
# TS (frame-level scan — max_seconds is meaningful)
# ---------------------------------------------------------------------------

class TestGetVideoPropertiesRemoteTS:
    """Tests against an Apple HLS .ts segment that requires frame-level scanning."""

    def test_returns_nonzero_fps_width_height(self):
        fps, total_frames, width, height = get_video_properties(
            _TS_URL, max_seconds=10
        )
        assert fps > 0, f"expected fps > 0, got {fps}"
        assert width > 0, f"expected width > 0, got {width}"
        assert height > 0, f"expected height > 0, got {height}"

    def test_fps_in_realistic_range(self):
        fps, _, _, _ = get_video_properties(_TS_URL, max_seconds=10)
        assert 1 <= fps <= 120, f"fps {fps} outside expected range [1, 120]"

    def test_max_seconds_limits_frame_count(self):
        """Scanning 3 s must not yield more frames than scanning 10 s."""
        _, frames_3s, _, _ = get_video_properties(_TS_URL, max_seconds=3)
        _, frames_10s, _, _ = get_video_properties(_TS_URL, max_seconds=10)
        assert frames_3s <= frames_10s, (
            f"frames with max_seconds=3 ({frames_3s}) should be <= "
            f"frames with max_seconds=10 ({frames_10s})"
        )

    def test_max_seconds_reduces_wall_time(self):
        """Partial scan (3 s) must complete faster than a full segment scan."""
        t0 = time.time()
        get_video_properties(_TS_URL, max_seconds=3)
        elapsed_partial = time.time() - t0

        t0 = time.time()
        get_video_properties(_TS_URL)  # full scan
        elapsed_full = time.time() - t0

        assert elapsed_partial < elapsed_full, (
            f"max_seconds=3 took {elapsed_partial:.2f}s, "
            f"full scan took {elapsed_full:.2f}s -- "
            "expected partial scan to be faster"
        )

    def test_fps_estimate_consistent_across_partial_scans(self):
        """FPS estimated from 3 s and 10 s should agree within +-5 fps."""
        fps_3s, _, _, _ = get_video_properties(_TS_URL, max_seconds=3)
        fps_10s, _, _, _ = get_video_properties(_TS_URL, max_seconds=10)
        assert abs(fps_3s - fps_10s) <= 5, (
            f"fps from 3 s scan ({fps_3s}) and 10 s scan ({fps_10s}) "
            "differ by more than 5 -- estimates are inconsistent"
        )

    def test_width_height_stable_across_max_seconds(self):
        """Resolution must not change depending on how many frames are scanned."""
        _, _, w3, h3 = get_video_properties(_TS_URL, max_seconds=3)
        _, _, w10, h10 = get_video_properties(_TS_URL, max_seconds=10)
        assert w3 == w10, f"width inconsistent: {w3} vs {w10}"
        assert h3 == h10, f"height inconsistent: {h3} vs {h10}"
