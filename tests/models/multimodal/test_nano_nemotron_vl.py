# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.model_executor.models.nano_nemotron_vl import (
    NanoNemotronVLMultiModalProcessor,
    NemotronH_Nano_VL_V2,
)
from vllm.multimodal.parse import (
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)


def _uninitialized_model() -> NemotronH_Nano_VL_V2:
    return object.__new__(NemotronH_Nano_VL_V2)  # type: ignore[type-abstract]


@pytest.mark.parametrize("input_key", ["image_embeds", "video_embeds"])
def test_precomputed_multimodal_embeddings(input_key: str):
    model = _uninitialized_model()
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


def test_nano_nemotron_vl_skips_multimodal_weights_in_text_only_mode():
    model = _uninitialized_model()
    language_model = _LanguageModel()
    object.__setattr__(model, "model_config", _ModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", _MissingMultiModalModule())
    object.__setattr__(model, "vision_final_layernorm", _MissingMultiModalModule())
    object.__setattr__(model, "sound_encoder", None)

    language_weight = torch.tensor(1.0)
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("mlp1.0.weight", torch.empty(0)),
            ("vision_model.radio_model.encoder.weight", torch.empty(0)),
            ("sound_encoder.encoder.weight", torch.empty(0)),
            ("vision_projector.vision_final_layernorm.weight", torch.empty(0)),
            ("vision_projector.vision_final_layernorm.bias", torch.empty(0)),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]


def test_nano_nemotron_vl_loads_vision_weights_without_sound_encoder():
    model = _uninitialized_model()
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "vision_final_layernorm", None)
    object.__setattr__(model, "sound_encoder", None)

    language_weight = torch.tensor(1.0)
    vision_weight = torch.tensor([2.0, 3.0])
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("vision_model.radio_model.encoder.weight", vision_weight),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]
    assert len(vision_model.loaded_weights) == 1
    name, loaded_weight = vision_model.loaded_weights[0]
    assert name == "radio_model.encoder.weight"
    assert isinstance(loaded_weight, torch.Tensor)
    torch.testing.assert_close(loaded_weight, vision_weight)
    assert loaded_weight is not vision_weight


def test_nano_nemotron_vl_requires_sound_encoder_for_sound_weights():
    model = _uninitialized_model()
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "vision_final_layernorm", None)
    object.__setattr__(model, "sound_encoder", None)

    with pytest.raises(AssertionError):
        model.load_weights([("sound_encoder.encoder.weight", torch.empty(0))])


def _vision_norm_model(use_layernorm=True):
    model = _uninitialized_model()
    torch.nn.Module.__init__(model)
    model.model_config = _ImageOnlyModelConfig()
    model.language_model = _LanguageModel()
    model.vision_model = _VisionModel()
    model.sound_encoder = None
    model.mlp1 = torch.nn.Identity()
    model.vision_final_layernorm = (
        torch.nn.LayerNorm(4, eps=1e-6, dtype=torch.bfloat16) if use_layernorm else None
    )
    model.patch_size = 2
    model.downsample_ratio = 0.5
    model.ps_version = "v2"
    model.video_temporal_patch_size = 2
    return model


@pytest.mark.parametrize("mode", ["image", "dynamic", "video"])
@pytest.mark.parametrize("use_layernorm", [True, False])
def test_vision_layernorm_is_loaded_and_applied_before_pixel_shuffle(
    mode, use_layernorm
):
    model = _vision_norm_model(use_layernorm)
    weight = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.bfloat16)
    bias = torch.tensor([0.5, -0.5, 1.0, -1.0], dtype=torch.bfloat16)
    model.load_weights(
        [
            ("vision_projector.vision_final_layernorm.weight", weight),
            ("vision_projector.vision_final_layernorm.bias", bias),
        ]
        if use_layernorm
        else []
    )
    features = torch.arange(16, dtype=torch.bfloat16).reshape(1, 4, 4)
    expected = (
        torch.nn.functional.layer_norm(features, (4,), weight, bias, eps=1e-6)
        if use_layernorm
        else features
    )
    expected = model.pixel_shuffle(expected.reshape(1, 2, 2, 4)).reshape(1, 1, 16)
    model.vision_model = lambda *args, **kwargs: (None, features)
    pixels = torch.zeros(2 if mode == "video" else 1, 3, 4, 4)
    if mode == "dynamic":
        output = model.extract_feature_dynamic(pixels, imgs_sizes=[(4, 4)])
    else:
        output = model.extract_feature(
            pixels, num_frames=2 if mode == "video" else None
        )
    torch.testing.assert_close(output, expected)


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
