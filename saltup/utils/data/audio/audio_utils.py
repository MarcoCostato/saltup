"""
Audio loading utilities for the saltup library.

This module provides generic functions for loading audio data from files
and in-memory byte streams.  It supports both encoded formats (wav, mp3,
flac, …) via ``librosa`` and headerless raw PCM.

**Dependency note** — ``librosa`` is an *optional* dependency
(part of the ``saltup[full]`` extra).  Functions in this module raise
a clear ``ImportError`` at call time if ``librosa`` is not installed,
keeping the base ``saltup`` package lightweight.

Typical usage::

    from saltup.utils.data.audio.audio_utils import load_audio, load_audio_from_stream

    audio, sr = load_audio("recording.wav", sr=44100)
"""

from __future__ import annotations

import io
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy librosa import — fail gracefully at call time, not at module import.
# ---------------------------------------------------------------------------

_librosa = None


def _ensure_librosa():
    """Import librosa on first use and cache the reference. Reduce overhead for users who don't need it."""
    global _librosa
    if _librosa is None:
        try:
            import librosa
            _librosa = librosa
        except ImportError as exc:
            raise ImportError(
                "librosa is required for audio processing utilities. "
                "Install it with:  pip install saltup[full]  or  pip install librosa"
            ) from exc
    return _librosa


# =============================================================================
# Audio Loading
# =============================================================================

def load_audio(
    audio_path: str,
    sr: int = 44100,
    is_raw: bool = False,
    raw_format: str = "int16",
    n_channels: int = 1,
    downsample: bool = False,
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load an audio file and return the waveform as a 1-D float array.

    Supports two modes:

    * **Encoded files** (wav, mp3, flac, …) — decoded via ``librosa.load``.
    * **Raw headerless files** — read directly from disk as a flat
      binary blob with the specified ``raw_format``.

    In both cases the returned array is mono, float32-normalised to the
    range ``[-1, 1]``.

    Args:
        audio_path: Path to the audio file.
        sr: Target sample rate in Hz.  ``librosa.load`` will resample
            automatically.  For raw files this is the *assumed* original
            sample rate.  Defaults to 44 100.
        is_raw: If ``True``, treat the file as headerless raw PCM.
            Defaults to ``False``.
        raw_format: NumPy dtype string for raw files (e.g. ``"int16"``,
            ``"int32"``, ``"float32"``).  Ignored when ``is_raw`` is
            ``False``.  Defaults to ``"int16"``.
        n_channels: Number of interleaved channels in the raw file.
            Multi-channel data is averaged to mono.  Defaults to 1.
        downsample: If ``True``, resample the audio to *target_sr*
            after loading.  Defaults to ``False``.
        target_sr: Target sample rate when *downsample* is ``True``.
            Defaults to 16 000.

    Returns:
        A tuple ``(audio_data, sample_rate)`` where *audio_data* is a
        1-D ``float32`` NumPy array and *sample_rate* is the effective
        sample rate after any downsampling.

    Raises:
        IOError: If the file cannot be read or decoded.

    Examples:
        >>> y, sr = load_audio("speech.wav", sr=16000)
        >>> y.dtype
        dtype('float32')
    """
    librosa = _ensure_librosa()

    try:
        if is_raw:
            audio_data = np.fromfile(audio_path, dtype=raw_format)

            # De-interleave multi-channel data
            if n_channels > 1:
                audio_data = audio_data.reshape(-1, n_channels)

            # Normalise integer formats to [-1, 1]
            audio_data = _normalise_pcm(audio_data, raw_format)

            # Mix to mono
            if n_channels > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Optional downsampling
            if downsample and target_sr < sr:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
                logger.info("Downsampled raw audio to %d Hz", target_sr)

            logger.info(
                "Loaded raw audio: %d samples, %d channel(s) at %d Hz",
                len(audio_data), n_channels, sr,
            )
            return audio_data, sr

        else:
            # librosa handles mp3, wav, flac, ogg, …
            y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)

            if downsample and target_sr < sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
                logger.info("Downsampled audio to %d Hz", target_sr)

            logger.info("Loaded audio: %d samples at %d Hz", len(y), sr)
            return y, sr

    except Exception as exc:
        raise IOError(f"Cannot load audio file {audio_path}: {exc}") from exc


def load_audio_from_stream(
    stream: io.RawIOBase,
    sr: int = 44100,
    raw_format: str = "int16",
    n_channels: int = 1,
) -> Tuple[np.ndarray, int]:
    """Load raw audio from an in-memory byte stream (e.g. from MinIO / S3).

    The stream is consumed in a single ``read()`` call and interpreted as
    headerless PCM with the specified format and channel layout.

    Args:
        stream: A file-like or ``BytesIO`` object containing raw audio
            bytes.
        sr: Assumed sample rate of the raw data.  Defaults to 44 100.
        raw_format: NumPy dtype for the raw samples.  Defaults to
            ``"int16"``.
        n_channels: Number of interleaved channels.  Defaults to 1.

    Returns:
        ``(audio_data, sample_rate)`` — same convention as
        :func:`load_audio`.

    Raises:
        IOError: If the stream cannot be decoded.

    Examples:
        >>> import io
        >>> buf = io.BytesIO(b"\\x00" * 1000)
        >>> y, sr = load_audio_from_stream(buf, sr=16000)
    """
    try:
        audio_bytes = stream.read()
        audio_data = np.frombuffer(audio_bytes, dtype=raw_format)

        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels)

        audio_data = _normalise_pcm(audio_data, raw_format)

        if n_channels > 1:
            audio_data = np.mean(audio_data, axis=1)

        logger.info(
            "Loaded audio from stream: %d samples at %d Hz",
            len(audio_data), sr,
        )
        return audio_data, sr

    except Exception as exc:
        raise IOError(f"Cannot load audio from stream: {exc}") from exc


# =============================================================================
# Private Helpers
# =============================================================================

def _normalise_pcm(audio_data: np.ndarray, raw_format: str) -> np.ndarray:
    """Normalise integer PCM data to float32 in [-1, 1]."""
    if raw_format == "int16":
        return audio_data.astype(np.float32) / 32768.0
    elif raw_format == "int32":
        return audio_data.astype(np.float32) / 2147483648.0
    # float32 or other formats: return as-is
    return audio_data.astype(np.float32)

