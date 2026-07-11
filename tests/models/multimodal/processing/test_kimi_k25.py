# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
from PIL import Image

from vllm.assets.video import VideoAsset
from vllm.model_executor.models.kimi_k25 import (
    _format_video_timestamp,
    _frames_to_pil_images,
    _split_video_chunks,
)

pytestmark = pytest.mark.skip_global_cleanup


class FakeImageProcessor:
    media_proc_cfg = {
        "temporal_merge_kernel_size": 2,
        "timestamp_mode": "hh:mm:ss.fff",
    }


def test_format_video_timestamp() -> None:
    assert _format_video_timestamp(0) == "00:00:00.000"
    assert _format_video_timestamp(65.25, "mm:ss.fff") == "01:05.250"
    assert _format_video_timestamp(65.25, "mm:ss") == "01:05"
    assert _format_video_timestamp(90061.5) == "25:01:01.500"
    assert _format_video_timestamp(-1) == "00:00:00.000"

    with pytest.raises(ValueError, match="Invalid Kimi video timestamp mode"):
        _format_video_timestamp(1, "seconds")


def test_frames_to_pil_images_accepts_uint8_thwc() -> None:
    thwc_frames = np.full((2, 5, 6, 3), 127, dtype=np.uint8)

    thwc_images = _frames_to_pil_images(thwc_frames)

    assert len(thwc_images) == 2
    assert all(image.mode == "RGB" for image in thwc_images)
    assert thwc_images[0].size == (6, 5)
    assert thwc_images[0].getpixel((0, 0)) == (127, 127, 127)


def test_frames_to_pil_images_accepts_frame_lists() -> None:
    image = Image.new("RGBA", (2, 3))
    frame = np.full((3, 2, 3), 255, dtype=np.uint8)

    images = _frames_to_pil_images([image, frame])

    assert len(images) == 2
    assert all(image.mode == "RGB" for image in images)
    assert images[0].size == (2, 3)
    assert images[1].size == (2, 3)
    assert images[1].getpixel((0, 0)) == (255, 255, 255)


def test_frames_to_pil_images_rejects_unexpected_frame_dtype() -> None:
    with pytest.raises(ValueError, match="must be uint8"):
        _frames_to_pil_images(np.ones((1, 2, 2, 3), dtype=np.float32))


def test_split_video_chunks_uses_video_metadata_for_timestamps() -> None:
    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)

    chunks = _split_video_chunks(
        (
            frames,
            {
                "fps": 10.0,
                "frames_indices": [0, 10, 20],
            },
        ),
        FakeImageProcessor(),
    )

    assert len(chunks) == 2
    assert chunks[0]["type"] == "video_chunk"
    assert len(chunks[0]["video_chunk"]) == 2
    assert chunks[0]["prompt"].startswith("00:00:00.000")
    assert chunks[1]["prompt"].startswith("00:00:02.000")


def test_split_video_chunks_accepts_existing_video_asset() -> None:
    video_asset = VideoAsset(name="baby_reading", num_frames=4)
    metadata = dict(video_asset.metadata)
    metadata["fps"] = 2.0
    metadata["frames_indices"] = [0, 1, 2, 3]

    chunks = _split_video_chunks(
        (video_asset.np_ndarrays, metadata),
        FakeImageProcessor(),
    )

    assert len(chunks) == 2
    assert [chunk["type"] for chunk in chunks] == ["video_chunk", "video_chunk"]
    assert chunks[0]["prompt"].startswith("00:00:00.000")
    assert chunks[1]["prompt"].startswith("00:00:01.000")
    assert all(
        isinstance(frame, Image.Image)
        for chunk in chunks
        for frame in chunk["video_chunk"]
    )


def test_split_video_chunks_rejects_empty_frames() -> None:
    frames = np.zeros((0, 2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="decoded to zero frames"):
        _split_video_chunks(frames, FakeImageProcessor())


def test_split_video_chunks_rejects_invalid_metadata() -> None:
    frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="frames_indices length"):
        _split_video_chunks(
            (frames, {"fps": 1.0, "frames_indices": [0]}),
            FakeImageProcessor(),
        )

    with pytest.raises(ValueError, match="fps must be positive"):
        _split_video_chunks((frames, {"fps": 0}), FakeImageProcessor())
