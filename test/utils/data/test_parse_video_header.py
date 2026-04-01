import saltup.utils.data.video.video_utils as video_utils
from saltup.utils.data.video.video_utils import (
    parse_avi_header,
    parse_flv_header,
    parse_mkv_header,
    parse_mov_header,
    parse_mp4_header,
    parse_video_header,
)


def test_parse_flv_header_valid_signature():
    header = b"FLV" + bytes([0x01, 0x05]) + (9).to_bytes(4, "big") + b"\x00\x00\x00\x00"
    result = parse_flv_header(header)

    assert result["format"] == "FLV"
    assert result["version"] == 1
    assert result["flags"] == 5
    assert result["data_offset"] == 9


def test_parse_avi_header_reads_basic_stream_metadata():
    avih_data = bytearray(40)
    avih_data[0:4] = (40_000).to_bytes(4, "little")
    avih_data[16:20] = (250).to_bytes(4, "little")
    avih_data[32:36] = (1280).to_bytes(4, "little")
    avih_data[36:40] = (720).to_bytes(4, "little")

    header = b"RIFF" + (100).to_bytes(4, "little") + b"AVI " + b"JUNK" + b"avih" + (40).to_bytes(4, "little") + bytes(avih_data)
    result = parse_avi_header(header)

    assert result["format"] == "AVI"
    assert result["width"] == 1280
    assert result["height"] == 720
    assert result["total_frames"] == 250
    assert result["fps"] == 25.0


def test_parse_mkv_header_detects_webm_doctype():
    header = b"\x1A\x45\xDF\xA3" + b"\x00" * 64 + b"webm"
    result = parse_mkv_header(header)

    assert result["format"] == "WEBM"
    assert result["width"] is None
    assert result["height"] is None


def test_parse_mp4_header_extracts_duration_and_dimensions():
    head = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"

    mvhd = bytearray(b"mvhd" + (b"\x00" * 40))
    mvhd[4] = 0
    mvhd[16:20] = (1000).to_bytes(4, "big")
    mvhd[20:24] = (5000).to_bytes(4, "big")

    tkhd = bytearray(b"tkhd" + (b"\x00" * 100))
    tkhd[4] = 0
    tkhd[80:84] = (1920 << 16).to_bytes(4, "big")
    tkhd[84:88] = (1080 << 16).to_bytes(4, "big")

    header = head + b"moov" + bytes(mvhd) + b"trak" + bytes(tkhd)
    result = parse_mp4_header(header)

    assert result["format"] == "MP4"
    assert result["width"] == 1920
    assert result["height"] == 1080
    assert result["duration"] == 5.0


def test_parse_mov_header_reuses_mp4_parsing_logic():
    header = b"\x00\x00\x00\x18ftypqt  " + (b"\x00" * 32)
    result = parse_mov_header(header)
    assert result["format"] == "MOV"


def test_parse_video_header_mp4_reads_moov_from_tail(tmp_path):
    mp4_path = tmp_path / "internet_style.mp4"
    head = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + (b"\x00" * 64)

    mvhd = bytearray(b"mvhd" + (b"\x00" * 40))
    mvhd[4] = 0
    mvhd[16:20] = (1000).to_bytes(4, "big")
    mvhd[20:24] = (5000).to_bytes(4, "big")

    tkhd = bytearray(b"tkhd" + (b"\x00" * 100))
    tkhd[4] = 0
    tkhd[80:84] = (1920 << 16).to_bytes(4, "big")
    tkhd[84:88] = (1080 << 16).to_bytes(4, "big")

    tail = b"moov" + bytes(mvhd) + b"trak" + bytes(tkhd)
    mp4_path.write_bytes(head + (b"\x00" * 256) + tail)

    result = parse_video_header(mp4_path)
    assert result["format"] == "MP4"
    assert result["width"] == 1920
    assert result["height"] == 1080
    assert result["duration"] == 5.0


def test_parse_video_header_dispatches_avi(tmp_path, monkeypatch):
    avi_path = tmp_path / "sample.avi"
    avi_path.write_bytes(b"RIFF" + (b"\x00" * 32))

    sentinel = {"format": "AVI", "width": 640, "height": 480, "fps": 30.0, "bit_depth": None}
    monkeypatch.setattr(video_utils, "parse_avi_header", lambda _data: sentinel)

    result = parse_video_header(avi_path)
    assert result == sentinel


def test_parse_video_header_unsupported_extension(tmp_path):
    file_path = tmp_path / "video.unknown"
    file_path.write_bytes(b"\x00\x01\x02\x03")
    result = parse_video_header(file_path)
    assert "error" in result
