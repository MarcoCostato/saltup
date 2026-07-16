import cv2
import numpy as np

from saltup.utils.data.image.image_utils import Image
from saltup.utils.data.video import video_utils


def test_motion_detection_crops_roi_before_resizing(monkeypatch):
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    metadata = video_utils.VideoProperties(
        fps=1,
        total_frames=2,
        width=frame.shape[1],
        height=frame.shape[0],
    )

    monkeypatch.setattr(video_utils, "get_video_properties", lambda *args, **kwargs: metadata)

    def fake_process_video(_path, callback, **kwargs):
        for frame_number in kwargs["frame_numbers"]:
            callback(Image(frame.copy()), frame_number, metadata.total_frames, metadata)
        return metadata

    monkeypatch.setattr(video_utils, "process_video", fake_process_video)

    resize_input_shapes = []
    original_resize = cv2.resize

    def recording_resize(image, *args, **kwargs):
        resize_input_shapes.append(image.shape)
        return original_resize(image, *args, **kwargs)

    monkeypatch.setattr(video_utils.cv2, "resize", recording_resize)

    config = video_utils.MotionDetectionOptions(
        roi=(0.25, 0.25, 0.75, 0.75),
        resize_width=4,
        store=False,
        verbose=False,
    )

    video_utils.motion_detection("unused.mp4", config=config)

    assert resize_input_shapes == [(4, 6, 3), (4, 6, 3)]
