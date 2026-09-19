# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
from PIL import Image

from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataParser,
    VideoProcessorItems,
)

H, W = 480, 640


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", (W, H)),
        # HWC, e.g. from np.array(PIL.Image)
        np.zeros((H, W, 3), dtype=np.uint8),
        torch.zeros((H, W, 3), dtype=torch.uint8),
        # CHW, standard PyTorch / numpy convention
        np.zeros((3, H, W), dtype=np.uint8),
        torch.zeros((3, H, W), dtype=torch.uint8),
    ],
)
def test_image_size_hwc_chw(image):
    """Image sizes must be channel-layout agnostic.

    `get_image_size` determines the multimodal placeholder count; reading an
    HWC array (the layout `np.array(PIL.Image)` produces) as CHW yields a
    bogus size and a placeholder/embedding count mismatch at inference time.
    """
    items = ImageProcessorItems([image])

    assert items.get_image_size(0) == (W, H)


@pytest.mark.parametrize(
    "frame",
    [
        Image.new("RGB", (W, H)),
        np.zeros((H, W, 3), dtype=np.uint8),
        torch.zeros((H, W, 3), dtype=torch.uint8),
        np.zeros((3, H, W), dtype=np.uint8),
        torch.zeros((3, H, W), dtype=torch.uint8),
    ],
)
def test_frame_size_hwc_chw(frame):
    """`get_frame_size` must stay consistent with `get_image_size`."""
    items = VideoProcessorItems([[frame]])

    assert items.get_frame_size(0) == (W, H)


def test_video_with_metadata_tensor_passthrough():
    """Tensor frames pass through unchanged regardless of device: HF video
    processors accept tensors, and device-resident frames (e.g. NVDEC-decoded)
    must not be copied back to host."""
    frames = torch.zeros((4, H, W, 3), dtype=torch.uint8)
    video, metadata = MultiModalDataParser()._get_video_with_metadata(frames)

    assert video is frames
    assert metadata is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_video_with_metadata_keeps_device_tensor():
    """Device-resident frames (e.g. NVDEC-decoded) pass through as tensors,
    so a device-side HF processor can consume them without a D2H copy."""
    frames = torch.zeros((4, H, W, 3), dtype=torch.uint8, device="cuda")
    video, metadata = MultiModalDataParser()._get_video_with_metadata(frames)

    assert video is frames
    assert metadata is None


@pytest.mark.parametrize(
    "frames",
    [
        [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(2)],
        [torch.zeros((H, W, 3), dtype=torch.uint8) for _ in range(2)],
    ],
)
def test_parse_video_frame_list_as_single_video(frames):
    """A list of decoded frames must represent one video item."""
    items = MultiModalDataParser().parse_mm_data({"video": frames})["video"]

    assert items.get_count() == 1
    video = items.get(0)
    assert isinstance(video, np.ndarray)
    np.testing.assert_array_equal(video, np.stack([np.asarray(f) for f in frames]))


@pytest.mark.parametrize(
    "modality,processor_cls",
    [
        ("audio", AudioProcessorItems),
        ("image", ImageProcessorItems),
        ("video", VideoProcessorItems),
    ],
)
def test_parse_mm_data_accepts_none_cached_item(modality, processor_cls):
    mm_items = MultiModalDataParser().parse_mm_data({modality: [None]})
    items = mm_items[modality]
    assert isinstance(items, processor_cls)
    assert len(items) == 1
    assert items.get(0) is None


def test_cached_audio_items_preserve_positions_during_resampling():
    waveform = np.arange(16, dtype=np.float32)
    parser = MultiModalDataParser(
        target_sr=16000, target_channels=1, audio_resample_method="scipy"
    )
    items = parser.parse_mm_data(
        {"audio": [None, (waveform, 8000), None, (waveform, 16000)]}
    )["audio"]

    assert len(items) == 4
    assert items.get(0) is None
    assert items.get(2) is None
    assert len(items.get(1)) == 32
    np.testing.assert_array_equal(items.get(3), waveform)


class _CountingDecoder:
    """Decoder that records how many times it ran."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


def test_parse_audio_lazy_item_resamples_on_decode():
    """A lazy audio item parses without decoding; resampling and channel
    normalization are stacked onto the decode."""
    from vllm.multimodal.media import LazyMedia

    waveform = np.arange(16, dtype=np.float32)
    parser = MultiModalDataParser(
        target_sr=16000, target_channels=1, audio_resample_method="scipy"
    )
    decoder = _CountingDecoder((waveform, 8000))
    items = parser.parse_mm_data({"audio": [LazyMedia(decoder, b"audio-bytes")]})[
        "audio"
    ]

    assert decoder.calls == 0
    audio = items.get(0)
    assert decoder.calls == 1
    assert len(audio) == 32


def test_parse_video_lazy_item_unpacks_frames_on_decode():
    """A lazy video item parses without decoding; the frames/metadata tuple
    is unpacked when the decode runs."""
    from vllm.multimodal.media import LazyMedia, MediaWithBytes

    frames = np.zeros((2, H, W, 3), dtype=np.uint8)
    metadata = {"total_num_frames": 2, "fps": 2.0, "duration": 1.0}
    inner = MediaWithBytes((frames, metadata), b"video-bytes")

    decoder = _CountingDecoder(inner)
    lazy = LazyMedia(decoder, b"video-bytes")
    items = MultiModalDataParser().parse_mm_data({"video": [lazy]})["video"]

    assert decoder.calls == 0
    np.testing.assert_array_equal(items.get(0), frames)
    assert decoder.calls == 1

    # video_needs_metadata defers its validation to decode time and yields
    # the (frames, metadata) tuple once decoded.
    parser = MultiModalDataParser(video_needs_metadata=True)
    decoder = _CountingDecoder(inner)
    lazy = LazyMedia(decoder, b"video-bytes")
    items = parser.parse_mm_data({"video": [lazy]})["video"]  # no error yet
    assert decoder.calls == 0
    video, out_metadata = items.get(0)
    np.testing.assert_array_equal(video, frames)
    assert out_metadata == metadata


def test_parse_video_lazy_missing_metadata_raises_on_decode():
    """With video_needs_metadata, a lazy video whose decode yields no
    metadata fails at decode time, not at parse time."""
    from vllm.multimodal.media import LazyMedia

    frames = np.zeros((2, H, W, 3), dtype=np.uint8)
    parser = MultiModalDataParser(video_needs_metadata=True)
    lazy = LazyMedia(lambda: frames, b"video-bytes")
    items = parser.parse_mm_data({"video": [lazy]})["video"]

    with pytest.raises(ValueError, match="metadata is required"):
        items.get(0)
