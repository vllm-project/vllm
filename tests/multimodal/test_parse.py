# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
from PIL import Image

from vllm.multimodal.media import MediaRef
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataParser,
    VideoProcessorItems,
)

H, W = 480, 640


class AudioMetadataParser(MultiModalDataParser):
    embedding_fields = {
        "audio": {"audio_embeds": "values", "audio_num_tokens": "metadata"},
    }


@pytest.mark.parametrize("allow_missing", [False, True])
def test_audio_metadata_requires_ec_consumer(allow_missing):
    parser = AudioMetadataParser(allow_missing_mm_embeddings=allow_missing)
    data = {"audio_num_tokens": torch.tensor([[3], [5]])}
    if not allow_missing:
        with pytest.raises(ValueError, match="audio_embeds"):
            parser.parse_mm_data({"audio": data})
    else:
        items = parser.parse_mm_data({"audio": data})["audio"]
        assert len(items) == 2
        assert items.get(1)["audio_num_tokens"].item() == 5
        assert items.get_processor_data() == {}


@pytest.mark.parametrize("counts", [[0], [-1], [1.5], [True], [[1, 2]]])
def test_audio_metadata_rejects_invalid_token_counts(counts):
    parser = AudioMetadataParser(allow_missing_mm_embeddings=True)
    with pytest.raises(ValueError, match="positive integer"):
        parser.parse_mm_data({"audio": {"audio_num_tokens": torch.tensor(counts)}})


def test_audio_metadata_checks_supplied_embedding_lengths():
    parser = AudioMetadataParser()
    data = {
        "audio_num_tokens": torch.tensor([3, 5]),
        "audio_embeds": [torch.zeros(3, 8), torch.zeros(5, 8)],
    }
    assert len(parser.parse_mm_data({"audio": data})["audio"]) == 2
    data["audio_num_tokens"] = torch.tensor([3, 4])
    with pytest.raises(ValueError, match="does not match"):
        parser.parse_mm_data({"audio": data})


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
    waveform = np.arange(16, dtype=np.float32)
    parser = MultiModalDataParser(
        target_sr=16000, target_channels=1, audio_resample_method="scipy"
    )
    decoder = _CountingDecoder((waveform, 8000))
    items = parser.parse_mm_data({"audio": [MediaRef(decoder, b"audio-bytes")]})[
        "audio"
    ]

    assert decoder.calls == 0
    audio = items.get(0)
    assert decoder.calls == 1
    assert len(audio) == 32


def test_parse_audio_lazy_item_key_covers_resample_settings():
    """The resampling settings join the item's cache key, so a different
    target rate cannot reuse a stale processor cache entry."""

    def parsed_key(target_sr, target_channels=None):
        parser = MultiModalDataParser(
            target_sr=target_sr,
            target_channels=target_channels,
            audio_resample_method="scipy",
        )
        items = parser.parse_mm_data(
            {"audio": [MediaRef(lambda: None, b"audio-bytes")]}
        )["audio"]
        ref = items.get_raw(0)
        assert isinstance(ref, MediaRef)
        return ref.key

    assert parsed_key(16000) == parsed_key(16000)
    assert parsed_key(16000) != parsed_key(8000)
    assert parsed_key(16000) != parsed_key(16000, target_channels=1)


def test_parse_video_lazy_item_unpacks_frames_on_decode():
    """A lazy video item parses without decoding; the frames/metadata tuple
    is unpacked when the decode runs."""
    frames = np.zeros((2, H, W, 3), dtype=np.uint8)
    metadata = {"total_num_frames": 2, "fps": 2.0, "duration": 1.0}

    decoder = _CountingDecoder((frames, metadata))
    lazy = MediaRef(decoder, b"video-bytes")
    items = MultiModalDataParser().parse_mm_data({"video": [lazy]})["video"]

    assert decoder.calls == 0
    np.testing.assert_array_equal(items.get(0), frames)
    assert decoder.calls == 1

    # video_needs_metadata defers its validation to decode time and yields
    # the (frames, metadata) tuple once decoded.
    parser = MultiModalDataParser(video_needs_metadata=True)
    decoder = _CountingDecoder((frames, metadata))
    lazy = MediaRef(decoder, b"video-bytes")
    items = parser.parse_mm_data({"video": [lazy]})["video"]  # no error yet
    assert decoder.calls == 0
    video, out_metadata = items.get(0)
    np.testing.assert_array_equal(video, frames)
    assert out_metadata == metadata


def test_parse_video_lazy_missing_metadata_raises_on_decode():
    """With video_needs_metadata, a lazy video whose decode yields no
    metadata fails at decode time, not at parse time."""
    frames = np.zeros((2, H, W, 3), dtype=np.uint8)
    parser = MultiModalDataParser(video_needs_metadata=True)
    lazy = MediaRef(lambda: frames, b"video-bytes")
    items = parser.parse_mm_data({"video": [lazy]})["video"]

    with pytest.raises(ValueError, match="metadata is required"):
        items.get(0)


@pytest.mark.parametrize("video_needs_metadata", [False, True])
@pytest.mark.parametrize("single_item", [False, True])
def test_lazy_video_metadata_available_before_frames(video_needs_metadata, single_item):
    """Audio extraction can read metadata without first requesting frames."""
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    metadata = {"original_video_bytes": b"video-bytes", "fps": 2.0}
    decoder = _CountingDecoder((frames, metadata))
    ref = MediaRef(decoder, b"video-bytes")
    parser = MultiModalDataParser(video_needs_metadata=video_needs_metadata)
    data = ref if single_item else [None, (frames, {"fps": 1.0}), ref]
    items = parser.parse_mm_data({"video": data})["video"]

    assert decoder.calls == 0
    index = 0 if single_item else 2
    assert isinstance(items, VideoProcessorItems)
    raw = items.get_raw(index)
    assert isinstance(raw, MediaRef)
    assert raw.key
    assert decoder.calls == 0
    expected = [metadata] if single_item else [None, {"fps": 1.0}, metadata]
    assert items.metadata == expected
    assert items.metadata == expected
    video = items.get(index)
    assert video is not None
    np.testing.assert_array_equal(video[0] if video_needs_metadata else video, frames)
    assert decoder.calls == 1


@pytest.mark.parametrize("video_needs_metadata", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_reparse_video_preserves_metadata(video_needs_metadata, lazy):
    """Cache miss reparsing preserves metadata and the processor's frames view."""
    from vllm.multimodal.media import MediaRef

    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    metadata = {"fps": 2.0, "original_video_bytes": b"video-bytes"}
    decoder = _CountingDecoder((frames, metadata))
    video = MediaRef(decoder, b"video-bytes") if lazy else (frames, metadata)
    parser = MultiModalDataParser(video_needs_metadata=video_needs_metadata)
    items = parser.parse_mm_data({"video": [None, video]})["video"]
    reparsed = parser.parse_mm_data({"video": items.get_all_raw()})["video"]

    assert decoder.calls == 0
    assert isinstance(items, VideoProcessorItems)
    assert isinstance(reparsed, VideoProcessorItems)
    if lazy:
        original_ref = items.get_raw(1)
        assert isinstance(original_ref, MediaRef)
        reparsed_ref = reparsed.get_raw(1)
        assert isinstance(reparsed_ref, MediaRef)
        assert reparsed_ref.key == original_ref.key
    assert reparsed.metadata == items.metadata == [None, metadata]
    result = reparsed.get(1)
    assert result is not None
    np.testing.assert_array_equal(result[0] if video_needs_metadata else result, frames)
    assert decoder.calls == int(lazy)


@pytest.mark.parametrize("video_needs_metadata", [False, True])
def test_select_video_preserves_lazy_refs_and_metadata(video_needs_metadata):
    """Selecting misses keeps raw refs and resolves only selected metadata."""
    frames = np.zeros((3, 8, 12, 3), dtype=np.uint8)
    metadata = {"fps": 2.0}
    decoders = [_CountingDecoder((frames, metadata)) for _ in range(2)]
    refs = [MediaRef(decoder, bytes([index])) for index, decoder in enumerate(decoders)]
    items = MultiModalDataParser(
        video_needs_metadata=video_needs_metadata
    ).parse_mm_data({"video": [None, *refs]})["video"]

    selected = items.select([2, 1]).select([0])
    assert isinstance(selected, VideoProcessorItems)
    assert selected.get_original_index(0) == 2
    assert selected.get_raw(0) is items.get_raw(2)
    assert all(decoder.calls == 0 for decoder in decoders)
    assert selected.get_metadata(0) == metadata
    assert decoders[0].calls == 0
    assert decoders[1].calls == 1
    assert selected.get_num_frames(0) == 3
    assert selected.get_frame_size(0) == (12, 8)
    np.testing.assert_array_equal(selected.get_frames(0), frames)
    result = selected.get(0)
    assert result is not None
    np.testing.assert_array_equal(result[0] if video_needs_metadata else result, frames)


def test_select_audio_metadata_preserves_passthrough_fields():
    """Selecting dictionary embeddings keeps aligned model input fields."""
    parser = AudioMetadataParser(allow_missing_mm_embeddings=True)
    items = parser.parse_mm_data(
        {"audio": {"audio_num_tokens": torch.tensor([[3], [5]])}}
    )["audio"]
    selected = items.select([1])

    assert selected.get_count() == 1
    assert selected.get_original_index(0) == 1
    assert selected.get(0)["audio_num_tokens"].item() == 5
    token_counts = selected.get_passthrough_data()["audio_num_tokens"]
    assert isinstance(token_counts, torch.Tensor)
    assert token_counts.numel() == 1


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize("indices", [[], [2, 0]])
def test_select_embeddings_preserves_representation(as_tensor, indices):
    """Selection keeps batched tensors and lists usable as model inputs."""
    data = torch.arange(24).reshape(3, 2, 4)
    items = MultiModalDataParser().parse_mm_data(
        {"image": data if as_tensor else list(data)}
    )["image"]
    selected = items.select([2, 1, 0]).select(indices)
    values = selected.get_passthrough_data()["image_embeds"]
    assert isinstance(values, torch.Tensor if as_tensor else list)
    assert selected.get_count() == len(indices)
    for index, original in enumerate([2, 1, 0][i] for i in indices):
        assert selected.get_original_index(index) == original
        torch.testing.assert_close(selected.get(index), data[original])
    if as_tensor:
        assert isinstance(values, torch.Tensor)
        assert values.shape == (len(indices), 2, 4)
