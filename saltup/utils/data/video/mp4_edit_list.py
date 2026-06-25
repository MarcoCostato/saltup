"""
Mp4EditListPatcher — neutralize MP4 edit lists at serving time, header-only.

Why this exists
---------------
NVR/camera MP4 exports often carry an *edit list* (``moov/trak/edts/elst``) with
a non-zero ``media_time`` that trims an initial preroll. Chrome **honors** the
edit list; Firefox historically **ignores** it. The result: the same
``currentTime`` maps to a different frame in the two browsers (a constant offset
of several seconds), so a bounding box drawn in one browser lands on the wrong
frame in the other. Desktop players (VLC, GNOME) both honor it and agree.

There is no ``<video>`` API to control edit-list handling, and the golden copy
is immutable. The fix is applied to the bytes the browser receives:

1. **Remove the edit list** — rename every ``edts`` box to ``free`` (a no-op
   padding box). Both browsers then ignore it and agree on duration.
2. **Rebase composition timestamps** — with B-frames the first displayed frame
   has a composition delay (its PTS is e.g. 0.2s, not 0). Chrome honors that
   delay (starts at 0.2s), Firefox normalizes to 0 → a residual ~2-frame
   offset. We remove it by shifting every ``ctts`` offset down by the minimum
   PTS (converting ``ctts`` to version 1 / signed), so the first frame's PTS
   becomes 0 in both browsers. This is exactly what ``ffmpeg
   -movflags +negative_cts_offsets`` produces, verified byte-for-byte.

Crucially both edits touch **only the ``moov`` header**, keep every box the same
size, and never move the ``mdat``. The frame data is byte-identical, the file
size is unchanged, and ``stco``/``co64`` stay valid — so the patched ``moov``
splices into the existing byte-range proxy with a 1:1 mapping, no re-encode and
no full download.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Container boxes we recurse into while looking for the edit list.
_CONTAINER_TYPES = {b"moov", b"trak", b"edts", b"mdia", b"minf", b"stbl"}

# Safety bounds so a malformed file can never make us loop or read forever.
_MAX_TOPLEVEL_BOXES = 64
_BOX_HEADER_BYTES = 16  # enough for 64-bit size + type
# The minimum PTS always lives in the first GOP; cap the sample scan so a huge
# file can't make computing the shift expensive.
_MAX_SAMPLES_SCAN = 100_000
_HTTP_TIMEOUT = 15
# Cap the in-memory cache. Each entry is one moov (~100-200 KB typically).
_MAX_CACHE_ENTRIES = 512


@dataclass
class MoovPatch:
    """A cached, edit-list-stripped ``moov`` ready to splice into byte ranges.

    ``moov_start``/``moov_end`` are absolute file offsets; ``patched`` is the
    rewritten ``moov`` of exactly ``moov_end - moov_start`` bytes (same size as
    the original), so it overlays the original bytes 1:1.
    """
    moov_start: int
    moov_end: int
    patched: bytes


class Mp4EditListPatcher:
    """Locates and neutralizes MP4 edit lists, header-only, with an LRU cache.

    Results are cached by ``file_id`` (the golden copy is immutable, so the
    patch is stable). A cached value of ``None`` means "this file needs no
    patch" and is remembered too, so we never re-probe it.
    """

    def __init__(self) -> None:
        self._cache: "OrderedDict[object, Optional[MoovPatch]]" = OrderedDict()
        self._lock = threading.Lock()

    # -- public API ----------------------------------------------------------

    def get_patch(self, file_id: object, presigned_url: str) -> Optional[MoovPatch]:
        """Return a :class:`MoovPatch` for *file_id*, or ``None`` if not needed.

        Never raises: any failure (network, parsing, unexpected layout) is
        logged and returns ``None`` so streaming falls back to the original
        bytes untouched.
        """
        with self._lock:
            if file_id in self._cache:
                self._cache.move_to_end(file_id)
                return self._cache[file_id]

        try:
            patch = self._build_patch(presigned_url)
        except Exception as exc:  # fail-safe: never break streaming
            logger.warning("edit-list patch probe failed for file %s: %s", file_id, exc)
            patch = None

        with self._lock:
            self._cache[file_id] = patch
            self._cache.move_to_end(file_id)
            while len(self._cache) > _MAX_CACHE_ENTRIES:
                self._cache.popitem(last=False)
        return patch

    @classmethod
    def patch_bytes(cls, data: bytes) -> Optional[bytes]:
        """Return *data* with its ``moov`` edit list neutralized, or ``None``.

        Pure in-memory variant of :meth:`get_patch` for when the whole file is
        already local (tests, or callers that decode from a temp file). ``None``
        means the file had no problematic edit list — serve it verbatim.
        """
        buf = bytearray(data)
        located = cls._locate_moov_in_buffer(buf)
        if not located:
            return None
        start, end = located
        moov = bytearray(buf[start:end])
        if not cls._patch_moov(moov):
            return None
        buf[start:end] = moov
        return bytes(buf)

    # -- internals -----------------------------------------------------------

    @classmethod
    def _locate_moov_in_buffer(cls, buf: bytearray) -> Optional[tuple[int, int]]:
        """Find the ``moov`` span among top-level boxes of an in-memory file."""
        for box_type, box_start, _body, box_end in cls._iter_boxes(buf, 0, len(buf)):
            if box_type == b"moov":
                return box_start, box_end
        return None

    def _build_patch(self, presigned_url: str) -> Optional[MoovPatch]:
        located = self._locate_moov(presigned_url)
        if not located:
            return None
        moov_start, moov_end = located

        moov = bytearray(self._range_get(presigned_url, moov_start, moov_end - 1))
        if len(moov) != (moov_end - moov_start):
            logger.warning(
                "moov range short read (%d != %d); skipping patch",
                len(moov), moov_end - moov_start,
            )
            return None

        if not self._patch_moov(moov):
            return None  # nothing to fix -> serve original bytes

        return MoovPatch(moov_start=moov_start, moov_end=moov_end, patched=bytes(moov))

    def _locate_moov(self, presigned_url: str) -> Optional[tuple[int, int]]:
        """Walk top-level boxes via tiny Range reads to find the ``moov`` span."""
        offset = 0
        for _ in range(_MAX_TOPLEVEL_BOXES):
            hdr = self._range_get(presigned_url, offset, offset + _BOX_HEADER_BYTES - 1)
            if len(hdr) < 8:
                return None
            size = int.from_bytes(hdr[0:4], "big")
            box_type = hdr[4:8]
            header_size = 8
            if size == 1:
                if len(hdr) < 16:
                    return None
                size = int.from_bytes(hdr[8:16], "big")
                header_size = 16
            elif size == 0:
                # Box runs to EOF. Only useful if it's the moov itself.
                if box_type == b"moov":
                    total = self._total_size(presigned_url)
                    return (offset, total) if total else None
                return None
            if box_type == b"moov":
                return offset, offset + size
            if size < header_size:
                return None  # malformed
            offset += size
        return None

    @classmethod
    def _patch_moov(cls, moov: bytearray) -> bool:
        """Apply the edit-list + composition-offset + duration fix to every track.

        Returns ``True`` if anything was modified, ``False`` if the file had no
        problematic edit list and no composition delay (nothing to do).
        """
        changed = False
        for box_type, _bs, body, end in cls._iter_boxes(moov, 0, len(moov)):
            if box_type != b"moov":
                continue
            mvhd = cls._find_box(moov, body, end, b"mvhd", deep=False)
            movie_ts = cls._read_timescale(moov, mvhd) if mvhd else 0
            max_movie_dur = 0
            for t, _ts, tbody, tend in cls._iter_boxes(moov, body, end):
                if t != b"trak":
                    continue
                movie_dur = cls._patch_trak(moov, tbody, tend, movie_ts)
                if movie_dur is not None:
                    changed = True
                    max_movie_dur = max(max_movie_dur, movie_dur)
            # Keep the movie duration consistent with the realigned tracks.
            if changed and mvhd and max_movie_dur > 0:
                cls._set_header_duration(moov, mvhd, max_movie_dur, "mvhd")
        return changed

    @classmethod
    def _patch_trak(cls, moov: bytearray, start: int, end: int, movie_timescale: int):
        """Fix one track. Returns its movie-timescale duration if patched, else None.

        - ``edts`` (direct child of ``trak``) → renamed to ``free`` so neither
          browser applies it (Firefox ignores it anyway).
        - ``ctts`` (under ``mdia/minf/stbl``) → every offset shifted down by the
          track's minimum PTS, as signed (version 1) values, so the first frame
          presents at PTS 0 in every decoder.
        - ``mdhd``/``tkhd`` durations → realigned to the real sample count (sum of
          ``stts`` deltas). A stale ``mdhd`` makes Firefox (which reads it) count a
          few trailing phantom frames that Chrome (which reads the sample table)
          does not.
        """
        edts = cls._find_box(moov, start, end, b"edts", deep=False)
        stts = cls._find_box(moov, start, end, b"stts", deep=True)
        ctts = cls._find_box(moov, start, end, b"ctts", deep=True)
        mdhd = cls._find_box(moov, start, end, b"mdhd", deep=True)
        tkhd = cls._find_box(moov, start, end, b"tkhd", deep=False)

        shift = cls._min_pts(moov, stts, ctts) if (stts and ctts) else 0
        edit_nonzero = bool(edts and cls._edts_has_nonzero_media_time(moov, edts[1], edts[2]))
        if shift <= 0 and not edit_nonzero:
            return None  # this track is already clean

        if edts:
            moov[edts[0] + 4: edts[0] + 8] = b"free"  # neutralize the whole edit box
        if ctts and shift > 0:
            cls._rebase_ctts(moov, ctts, shift)

        movie_dur = 0
        if stts and mdhd:
            media_dur = cls._sum_stts(moov, stts)
            if media_dur > 0:
                cls._set_header_duration(moov, mdhd, media_dur, "mdhd")
                media_ts = cls._read_timescale(moov, mdhd)
                if movie_timescale and media_ts:
                    movie_dur = round(media_dur / media_ts * movie_timescale)
                    if tkhd:
                        cls._set_header_duration(moov, tkhd, movie_dur, "tkhd")
        return movie_dur

    # -- box parsing helpers -------------------------------------------------

    @staticmethod
    def _iter_boxes(moov: bytearray, start: int, end: int):
        """Yield ``(type, box_start, body_start, box_end)`` for boxes in a span."""
        pos = start
        while pos + 8 <= end:
            size = int.from_bytes(moov[pos:pos + 4], "big")
            box_type = bytes(moov[pos + 4:pos + 8])
            header = 8
            real = size
            if size == 1:
                real = int.from_bytes(moov[pos + 8:pos + 16], "big")
                header = 16
            elif size == 0:
                real = end - pos
            if real < header or pos + real > end:
                return
            yield box_type, pos, pos + header, pos + real
            pos += real

    @classmethod
    def _find_box(cls, moov: bytearray, start: int, end: int, target: bytes, deep: bool):
        """Find first *target* box; returns ``(box_start, body_start, box_end)``.

        When *deep*, recurses into container boxes; otherwise only direct children.
        """
        for box_type, box_start, body, box_end in cls._iter_boxes(moov, start, end):
            if box_type == target:
                return box_start, body, box_end
            if deep and box_type in _CONTAINER_TYPES:
                found = cls._find_box(moov, body, box_end, target, deep)
                if found:
                    return found
        return None

    # -- duration / timescale helpers (mvhd / mdhd / tkhd) -------------------
    # Field layouts after the 4-byte version+flags at *body*:
    #   mvhd/mdhd: v0 timescale@+12 duration@+16(4);  v1 timescale@+20 duration@+24(8)
    #   tkhd:                       duration@+20(4) (v0);          duration@+28(8) (v1)

    @staticmethod
    def _read_timescale(moov: bytearray, box) -> int:
        """Read the timescale field of an mvhd/mdhd box."""
        _start, body, _end = box
        off = body + (20 if moov[body] == 1 else 12)
        return int.from_bytes(moov[off:off + 4], "big")

    @staticmethod
    def _set_header_duration(moov: bytearray, box, value: int, kind: str) -> None:
        """Overwrite the duration field of an mvhd/mdhd/tkhd box (version-aware)."""
        _start, body, _end = box
        v1 = moov[body] == 1
        if kind == "tkhd":
            off, size = (body + 28, 8) if v1 else (body + 20, 4)
        else:  # mvhd / mdhd
            off, size = (body + 24, 8) if v1 else (body + 16, 4)
        moov[off:off + size] = int(value).to_bytes(size, "big")

    @staticmethod
    def _sum_stts(moov: bytearray, stts) -> int:
        """Sum of all sample durations (stts), i.e. the true media duration."""
        _start, body, end = stts
        count = int.from_bytes(moov[body + 4:body + 8], "big")
        ep = body + 8
        total = 0
        for _ in range(count):
            if ep + 8 > end:
                break
            run = int.from_bytes(moov[ep:ep + 4], "big")
            delta = int.from_bytes(moov[ep + 4:ep + 8], "big")
            total += run * delta
            ep += 8
        return total

    @staticmethod
    def _parse_runs(moov: bytearray, box, signed: bool):
        """Parse a run-length table (stts/ctts) into ``[(count, value), ...]``."""
        _start, body, end = box
        count = int.from_bytes(moov[body + 4:body + 8], "big")
        ep = body + 8
        runs = []
        for _ in range(count):
            if ep + 8 > end:
                break
            cnt = int.from_bytes(moov[ep:ep + 4], "big")
            val = int.from_bytes(moov[ep + 4:ep + 8], "big", signed=signed)
            runs.append((cnt, val))
            ep += 8
        return runs

    @classmethod
    def _min_pts(cls, moov: bytearray, stts, ctts) -> int:
        """Minimum presentation timestamp across the track, in media ticks.

        PTS(sample) = DTS(sample) + ctts_offset(sample). DTS comes from the
        stts deltas. Walked sample-by-sample via run-lengths (no full expansion).
        """
        stts_runs = cls._parse_runs(moov, stts, signed=False)
        if not stts_runs:
            return 0
        ctts_signed = moov[ctts[1]] == 1
        s_iter = iter(stts_runs)
        s_count, s_delta = next(s_iter)
        dts = 0
        min_pts = None
        scanned = 0
        for c_count, c_off in cls._parse_runs(moov, ctts, signed=ctts_signed):
            for _ in range(c_count):
                while s_count == 0:
                    try:
                        s_count, s_delta = next(s_iter)
                    except StopIteration:
                        break  # ran out of stts; reuse last delta
                pts = dts + c_off
                if min_pts is None or pts < min_pts:
                    min_pts = pts
                dts += s_delta
                if s_count > 0:
                    s_count -= 1
                scanned += 1
                if scanned >= _MAX_SAMPLES_SCAN:
                    return max(0, min_pts or 0)
        return max(0, min_pts or 0)

    @staticmethod
    def _rebase_ctts(moov: bytearray, ctts, shift: int) -> None:
        """Shift every ctts offset down by *shift*, as signed (version 1)."""
        _start, body, end = ctts
        version = moov[body]
        moov[body] = 1  # signed composition offsets
        count = int.from_bytes(moov[body + 4:body + 8], "big")
        ep = body + 8
        for _ in range(count):
            if ep + 8 > end:
                break
            off = int.from_bytes(moov[ep + 4:ep + 8], "big", signed=(version == 1))
            moov[ep + 4:ep + 8] = (off - shift).to_bytes(4, "big", signed=True)
            ep += 8

    @staticmethod
    def _edts_has_nonzero_media_time(moov: bytearray, start: int, end: int) -> bool:
        """True if the ``edts``'s ``elst`` has any entry with media_time != 0."""
        pos = start
        while pos + 8 <= end:
            size = int.from_bytes(moov[pos:pos + 4], "big")
            box_type = bytes(moov[pos + 4:pos + 8])
            if size < 8 or pos + size > end:
                return False
            if box_type == b"elst":
                body = pos + 8
                version = moov[body]
                count = int.from_bytes(moov[body + 4:body + 8], "big")
                ep = body + 8
                for _ in range(count):
                    if version == 1:
                        if ep + 16 > end:
                            break
                        media_time = int.from_bytes(moov[ep + 8:ep + 16], "big", signed=True)
                        ep += 20
                    else:
                        if ep + 8 > end:
                            break
                        media_time = int.from_bytes(moov[ep + 4:ep + 8], "big", signed=True)
                        ep += 12
                    if media_time != 0:
                        return True
                return False
            pos += size
        return False

    # -- HTTP helpers --------------------------------------------------------

    @staticmethod
    def _range_get(presigned_url: str, start: int, end: int) -> bytes:
        resp = requests.get(
            presigned_url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def _total_size(presigned_url: str) -> Optional[int]:
        resp = requests.get(
            presigned_url,
            headers={"Range": "bytes=0-0"},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        content_range = resp.headers.get("Content-Range", "")
        if "/" in content_range:
            try:
                return int(content_range.split("/")[-1])
            except ValueError:
                return None
        return None
