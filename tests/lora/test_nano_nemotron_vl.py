# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
from vllm.model_executor.models.parakeet import ProjectedParakeet


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
            quant_config=None,
            return_bias,
            prefix,
        ):
            del quant_config, return_bias, prefix
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

    # _replace_encoder_linears delegates to recursive_replace_linear, which
    # constructs the replacement class in transformers.utils, not parakeet.
    target = "vllm.model_executor.models.transformers.utils.ReplicatedLinear"
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
