# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clef-Omni decodes media as its joint_schema_model.py does."""

from io import BytesIO

import av
import numpy as np
import pytest

from vllm.entrypoints.generate.structured_decisions.media import (
    DecodedMediaIO,
    decode_audio_clip,
    decode_video,
)


def encode_video(seconds: float, fps: int, size: tuple[int, int]) -> bytes:
    buffer = BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for _ in range(round(seconds * fps)):
            frame = np.zeros((size[1], size[0], 3), np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame)):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def encode_wav(seconds: float, rate: int) -> bytes:
    samples = (np.sin(np.arange(round(seconds * rate)) / 10) * 1e4).astype(np.int16)
    buffer = BytesIO()
    with av.open(buffer, "w", format="wav") as container:
        stream = container.add_stream("pcm_s16le", rate=rate, layout="stereo")
        frame = av.AudioFrame.from_ndarray(
            np.repeat(samples, 2).reshape(1, -1), format="s16", layout="stereo"
        )
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def test_video_is_sampled_at_two_frames_a_second():
    frames, soundtrack = decode_video(encode_video(2.3, fps=10, size=(640, 480)))
    # Frames at 0, 0.5, 1, 1.5 and 2 seconds, then the last again for an even
    # count. 640x480 shrinks to at most 256 * 32 * 32 pixels, in even sizes.
    assert frames.shape == (6, 444, 592, 3)
    np.testing.assert_array_equal(frames[-1], frames[-2])
    assert soundtrack is None


def test_audio_is_16khz_mono():
    samples = decode_audio_clip(encode_wav(1.5, rate=22050))
    assert samples.dtype == np.float32 and samples.ndim == 1
    assert abs(samples.size - 24000) <= 16


def test_bad_media_is_a_request_error():
    with pytest.raises(ValueError, match="invalid or corrupted media"):
        DecodedMediaIO(decode_video).load_bytes(b"not a video")
