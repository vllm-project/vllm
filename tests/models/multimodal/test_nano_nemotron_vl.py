# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from vllm import envs
from vllm.model_executor.models.nano_nemotron_vl import (
    NanoNemotronVLMultiModalProcessor,
    NemotronH_Nano_VL_V2,
)
from vllm.model_executor.models.vision import FusedInputNorm
from vllm.multimodal.parse import (
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)
from vllm.transformers_utils.processors.nano_nemotron_vl import (
    NanoNemotronVLProcessor,
    _bicubic_resize_and_normalize,
)


def test_bicubic_resize_preserves_uint8_for_device_normalization():
    pixels = torch.randint(0, 256, (1, 31, 47, 3), dtype=torch.uint8)
    size = (48, 64)

    output = _bicubic_resize_and_normalize(
        pixels,
        size=size,
        norm_mean=None,
        norm_std=None,
        do_cpu_normalize=False,
    )
    expected = torch.nn.functional.interpolate(
        pixels.permute(0, 3, 1, 2),
        size=size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).contiguous()

    assert output.dtype == torch.uint8
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_nemotron_dynamic_device_normalization_matches_cpu_reference():
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    shape = (1, 7, 3 * 16 * 16)
    pixels = torch.randint(0, 256, shape, dtype=torch.uint8)

    model = object.__new__(NemotronH_Nano_VL_V2)
    nn.Module.__init__(model)
    model.input_norm = FusedInputNorm(image_mean, image_std, 1.0 / 255.0)
    model.llm_dtype = torch.bfloat16

    output = model._normalize_pixel_values(pixels)
    batch, patches, _ = pixels.shape
    expected = pixels.to(torch.float32).view(batch, patches, 3, -1)
    expected = (
        expected / 255.0 - torch.tensor(image_mean).view(1, 1, 3, 1)
    ) / torch.tensor(image_std).view(1, 1, 3, 1)
    expected = expected.view(shape)

    torch.testing.assert_close(output, expected.to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("modality", ["image", "video"])
def test_nemotron_processor_defers_normalization_to_device(modality: str):
    config = SimpleNamespace(
        force_image_size=16,
        patch_size=4,
        downsample_ratio=0.5,
        use_thumbnail=False,
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        dtype=torch.bfloat16,
        vision_config=SimpleNamespace(args={}),
        sound_config=None,
    )
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1])
    common_kwargs = {
        "config": config,
        "tokenizer": tokenizer,
        "max_model_len": 1024,
    }
    cpu_processor = NanoNemotronVLProcessor(**common_kwargs)
    device_processor = NanoNemotronVLProcessor(
        **common_kwargs, do_rescale=False, do_normalize=False
    )

    pixels = np.random.default_rng(0).integers(0, 256, (4, 19, 23, 3), dtype=np.uint8)
    if modality == "image":
        image = Image.fromarray(pixels[0])
        cpu_values = cpu_processor._images_to_pixel_values_lst([image], 1)[0]
        raw_values = device_processor._images_to_pixel_values_lst([image], 1)[0]
    else:
        cpu_values = cpu_processor._videos_to_pixel_values_lst(
            [pixels], dtype=torch.bfloat16
        )[0]
        raw_values = device_processor._videos_to_pixel_values_lst([pixels])[0]
    assert raw_values.dtype == torch.uint8

    model = object.__new__(NemotronH_Nano_VL_V2)
    nn.Module.__init__(model)
    model.input_norm = FusedInputNorm(config.norm_mean, config.norm_std, 1.0 / 255.0)
    model.llm_dtype = torch.bfloat16
    normalized = model._normalize_pixel_values(raw_values)
    torch.testing.assert_close(normalized, cpu_values, rtol=0, atol=0)


def test_nemotron_processor_rejects_partial_normalization():
    with pytest.raises(
        ValueError, match="requires do_rescale and do_normalize to have the same value"
    ):
        NanoNemotronVLProcessor(
            config=None,
            tokenizer=None,
            max_model_len=1024,
            do_rescale=False,
            do_normalize=True,
        )


@pytest.mark.parametrize("input_key", ["image_embeds", "video_embeds"])
def test_precomputed_multimodal_embeddings(input_key: str):
    model = object.__new__(NemotronH_Nano_VL_V2)
    embeds = torch.randn(2, 4, 8)

    outputs = model.embed_multimodal(**{input_key: embeds})

    assert len(outputs) == len(embeds)
    assert all(torch.equal(output, embed) for output, embed in zip(outputs, embeds))


class _TextOnlyMultiModalConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        return 0


class _ImageOnlyMultiModalConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        return 1 if modality == "image" else 0


class _ModelConfig:
    multimodal_config = _TextOnlyMultiModalConfig()


class _ImageOnlyModelConfig:
    multimodal_config = _ImageOnlyMultiModalConfig()


class _LanguageModel:
    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


class _MissingMultiModalModule:
    def named_parameters(self):
        raise AssertionError("multimodal weights should not be inspected")

    def load_weights(self, weights):
        raise AssertionError("multimodal weights should not be loaded")


class _AdapterModule:
    def named_parameters(self):
        return []


class _VisionModel:
    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


class _FakeTensor:
    """Sentinel stand-in for torch.Tensor in load_weights tests. Supports the
    .detach().clone() chain used by load_weights for buffered mm weights;
    both methods return self so identity (and the existing equality
    assertions) are preserved through cloning."""

    def detach(self):
        return self

    def clone(self):
        return self


def test_nano_nemotron_vl_skips_multimodal_weights_in_text_only_mode():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    object.__setattr__(model, "model_config", _ModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", _MissingMultiModalModule())
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("mlp1.0.weight", object()),
            ("vision_model.radio_model.encoder.weight", object()),
            ("sound_encoder.encoder.weight", object()),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]


def test_nano_nemotron_vl_loads_vision_weights_without_sound_encoder():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    vision_weight = _FakeTensor()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("vision_model.radio_model.encoder.weight", vision_weight),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]
    assert vision_model.loaded_weights == [
        ("radio_model.encoder.weight", vision_weight)
    ]


def test_nano_nemotron_vl_requires_sound_encoder_for_sound_weights():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "sound_encoder", None)

    with pytest.raises(AssertionError):
        model.load_weights([("sound_encoder.encoder.weight", object())])


def _make_mm_items_with_video_bytes(
    video_bytes: bytes,
) -> MultiModalDataItems:
    """Build a minimal MultiModalDataItems with one video entry."""
    items = MultiModalDataItems()
    items["video"] = VideoProcessorItems(
        data=[None],
        metadata=[{"original_video_bytes": video_bytes}],
    )
    return items


def test_extract_audio_from_videos_passes_max_duration():
    """_extract_audio_from_videos must forward VLLM_MAX_AUDIO_DECODE_DURATION_S
    to load_audio_pyav so decompression-bomb audio is rejected."""
    dummy_audio = (np.zeros(16000, dtype=np.float32), 16000.0)
    mm_items = _make_mm_items_with_video_bytes(b"\x00" * 64)

    processor = object.__new__(NanoNemotronVLMultiModalProcessor)
    processor.data_parser = MultiModalDataParser(
        target_sr=dummy_audio[1], target_channels=1
    )

    target = "vllm.model_executor.models.nano_nemotron_vl.load_audio_pyav"
    with patch(target, return_value=dummy_audio) as mock_load:
        processor._extract_audio_from_videos(mm_items)

    mock_load.assert_called_once()
    _, kwargs = mock_load.call_args
    assert kwargs["max_duration_s"] == envs.VLLM_MAX_AUDIO_DECODE_DURATION_S


def test_extract_audio_from_videos_rejects_oversized_audio():
    """When load_audio_pyav raises due to duration limit the video is
    marked as having no audio instead of crashing the server."""
    mm_items = _make_mm_items_with_video_bytes(b"\x00" * 64)

    processor = object.__new__(NanoNemotronVLMultiModalProcessor)

    target = "vllm.model_executor.models.nano_nemotron_vl.load_audio_pyav"
    with patch(
        target,
        side_effect=ValueError("Audio exceeds maximum allowed duration"),
    ):
        _, audio_items, has_audio = processor._extract_audio_from_videos(mm_items)

    assert audio_items == []
    assert has_audio == [False]
