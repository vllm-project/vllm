# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.nano_nemotron_vl import (
    NanoNemotronVLMultiModalProcessor,
    NemotronH_Nano_VL_V2,
)
from vllm.model_executor.models.parakeet import ProjectedParakeet
from vllm.multimodal.parse import (
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
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
    """Text-only mode must not inspect or load multimodal weights."""
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
    """Vision weights must load when the model has no sound encoder."""
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
    """Sound weights must fail when the model was configured without audio."""
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


def _make_lora_count_model() -> NemotronH_Nano_VL_V2:
    model = object.__new__(NemotronH_Nano_VL_V2)
    object.__setattr__(model, "downsample_ratio", 0.5)
    object.__setattr__(model, "num_image_token", 256)
    object.__setattr__(model, "patch_size", 16)
    object.__setattr__(model, "video_temporal_patch_size", 2)
    patch_generator = SimpleNamespace(num_skip=3)
    vision_model = SimpleNamespace(
        model=SimpleNamespace(patch_generator=patch_generator)
    )
    object.__setattr__(model, "vision_model", vision_model)
    sound_encoder = SimpleNamespace(
        encoder=SimpleNamespace(
            _get_subsampling_output_length=lambda lengths: torch.div(
                lengths + 3, 4, rounding_mode="floor"
            )
        )
    )
    object.__setattr__(model, "sound_encoder", sound_encoder)
    return model


def test_nano_nemotron_vl_lora_mapping_includes_vision_and_audio():
    """Tower and connector mappings must cover both supported modalities."""
    model = _make_lora_count_model()
    mapping = model.get_mm_mapping()

    assert mapping.language_model == ["language_model"]
    assert mapping.tower_model == ["vision_model", "sound_encoder.encoder"]
    assert mapping.connector == ["mlp1", "sound_encoder.projection"]


def test_nano_nemotron_vl_maps_radio_lora_names():
    """RADIO checkpoint names must map to the vLLM vision encoder paths."""
    mapper = NemotronH_Nano_VL_V2.hf_to_vllm_mapper.get_rename_mapper()
    name = (
        "base_model.model.vision_model.radio_model.model.blocks.0.attn.qkv."
        "lora_A.weight"
    )

    assert parse_fine_tuned_lora_name(name, mapper) == (
        "vision_model.model.encoder.layers.0.attn.qkv",
        True,
    )


def test_nano_nemotron_vl_maps_audio_projection_lora_names():
    """Audio projection checkpoint names must map to their runtime paths."""
    mapper = NemotronH_Nano_VL_V2.hf_to_vllm_mapper.get_rename_mapper()
    name = "base_model.model.sound_projection.linear1.lora_A.weight"

    assert parse_fine_tuned_lora_name(name, mapper) == (
        "sound_encoder.projection.linear1",
        True,
    )


def test_parakeet_replaces_only_same_length_encoder_linears():
    """Only Parakeet linears using the common audio sequence can receive LoRA."""

    class _FakeReplicatedLinear(torch.nn.Linear):
        def __init__(
            self,
            input_size,
            output_size,
            *,
            bias,
            return_bias,
            prefix,
        ):
            del return_bias, prefix
            super().__init__(input_size, output_size, bias=bias)

    encoder = torch.nn.Module()
    encoder.subsampling = torch.nn.Module()
    encoder.subsampling.linear = torch.nn.Linear(4, 8)
    encoder.layer = torch.nn.Module()
    encoder.layer.q_proj = torch.nn.Linear(8, 8, bias=False)
    encoder.layer.relative_k_proj = torch.nn.Linear(8, 8, bias=False)

    model = object.__new__(ProjectedParakeet)
    torch.nn.Module.__init__(model)
    model.encoder = encoder

    target = "vllm.model_executor.models.parakeet.ReplicatedLinear"
    with patch(target, _FakeReplicatedLinear):
        model._replace_encoder_linears(prefix="sound_encoder.encoder")

    assert isinstance(model.encoder.subsampling.linear, _FakeReplicatedLinear)
    assert isinstance(model.encoder.layer.q_proj, _FakeReplicatedLinear)
    assert type(model.encoder.layer.relative_k_proj) is torch.nn.Linear


def test_nano_nemotron_vl_fixed_image_lora_token_counts():
    """Fixed-resolution image counts include RADIO skips and spatial merging."""
    model = _make_lora_count_model()
    mm_kwargs = {"pixel_values_flat": SimpleNamespace(data=torch.empty(2, 3, 32, 32))}

    assert model.get_mm_lora_token_counts(
        modality="image", mm_kwargs=mm_kwargs, num_mm_embeds=2
    ) == (14, 2)


def test_nano_nemotron_vl_dynamic_image_lora_token_counts():
    """Dynamic images must derive LoRA counts from each image's dimensions."""
    model = _make_lora_count_model()
    mm_kwargs = {
        "pixel_values_flat": SimpleNamespace(data=torch.empty(3, 32, 64)),
        "imgs_sizes": SimpleNamespace(data=(32, 64)),
    }

    assert model.get_mm_lora_token_counts(
        modality="image", mm_kwargs=mm_kwargs, num_mm_embeds=2
    ) == (11, 2)


def test_nano_nemotron_vl_video_lora_counts_precede_pruning():
    """Video tower counts must use temporal tubelets before EVS pruning."""
    model = _make_lora_count_model()
    mm_kwargs = {
        "pixel_values_flat_video": SimpleNamespace(data=torch.empty(3, 3, 32, 32))
    }

    assert model.get_mm_lora_token_counts(
        modality="video", mm_kwargs=mm_kwargs, num_mm_embeds=1
    ) == (14, 2)


def test_nano_nemotron_vl_audio_lora_token_counts():
    """Audio counts must use padded Parakeet output length for both stages."""
    model = _make_lora_count_model()
    mm_kwargs = {"input_audio_features": SimpleNamespace(data=torch.empty(2, 17, 80))}

    assert model.get_mm_lora_token_counts(
        modality="audio", mm_kwargs=None, num_mm_embeds=128
    ) == (128, 128)
    assert model.get_mm_lora_token_counts(
        modality="audio", mm_kwargs=mm_kwargs, num_mm_embeds=8
    ) == (10, 10)


def test_nano_nemotron_vl_precomputed_and_unknown_lora_token_counts():
    """Precomputed embeds bypass LoRA stages and unknown modalities fail closed."""
    model = _make_lora_count_model()

    assert model.get_mm_lora_token_counts(
        modality="image",
        mm_kwargs={"image_embeds": SimpleNamespace(data=torch.empty(2, 8))},
        num_mm_embeds=2,
    ) == (0, 0)
    with pytest.raises(ValueError, match="Unsupported modality"):
        model.get_mm_lora_token_counts(
            modality="unknown", mm_kwargs=None, num_mm_embeds=1
        )


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
    """Oversized audio must be treated as absent instead of crashing extraction."""
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
