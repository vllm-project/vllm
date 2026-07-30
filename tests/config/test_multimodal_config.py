# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
from transformers import PretrainedConfig

from vllm.config.model import ModelConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_mm_encoder_attn_backend_str_conversion():
    config = MultiModalConfig(mm_encoder_attn_backend="FLASH_ATTN")
    assert config.mm_encoder_attn_backend == AttentionBackendEnum.FLASH_ATTN


def test_mm_encoder_attn_backend_invalid():
    with pytest.raises(ValueError):
        MultiModalConfig(mm_encoder_attn_backend="not_a_backend")


def test_mm_encoder_attn_backend_hash_updates():
    base_hash = MultiModalConfig().compute_hash()
    overridden_hash = MultiModalConfig(
        mm_encoder_attn_backend=AttentionBackendEnum.FLASH_ATTN
    ).compute_hash()
    assert base_hash != overridden_hash


def test_language_model_only_does_not_affect_mm_hash():
    """language_model_only does not affect the ViT computation graph,
    so it should not change the multimodal config hash."""
    base_hash = MultiModalConfig().compute_hash()
    lm_only_hash = MultiModalConfig(language_model_only=True).compute_hash()
    assert base_hash == lm_only_hash


def test_language_model_only_affects_model_hash():
    """language_model_only affects the LM computation graph,
    so it should change the model config hash."""
    model = "llava-hf/llava-1.5-7b-hf"
    base_hash = ModelConfig(model).compute_hash()
    lm_only_hash = ModelConfig(model, language_model_only=True).compute_hash()
    assert base_hash != lm_only_hash


@pytest.mark.parametrize("backend_arg", ["video_backend", "backend"])
def test_use_gpu_video_backend_from_media_io_kwargs(backend_arg: str):
    config = MultiModalConfig(
        media_io_kwargs={"video": {backend_arg: "pynvvideocodec"}}
    )

    assert config.use_gpu_video_backend()


def test_mm_encoder_fp8_scale_path_requires_fp8():
    with pytest.raises(ValueError, match="mm_encoder_attn_dtype"):
        MultiModalConfig(mm_encoder_fp8_scale_path="/tmp/scales.json")


def test_mm_encoder_attn_dtype_hash_updates(tmp_path):
    scale_file = tmp_path / "scales.json"
    scale_file.write_text("{}")
    base_hash = MultiModalConfig().compute_hash()
    fp8_hash = MultiModalConfig(mm_encoder_attn_dtype="fp8").compute_hash()
    fp8_static_hash = MultiModalConfig(
        mm_encoder_attn_dtype="fp8",
        mm_encoder_fp8_scale_path=str(scale_file),
    ).compute_hash()
    assert base_hash != fp8_hash
    assert fp8_hash != fp8_static_hash


def _make_mm_prefix_model_config(
    *,
    language_model_only: bool = False,
) -> ModelConfig:
    model_config = MagicMock(spec=ModelConfig)
    model_config.multimodal_config = MultiModalConfig(
        language_model_only=language_model_only
    )
    # Bind real helper methods onto the mock.
    model_config._supports_multimodal_for_mm_prefix = (
        ModelConfig._supports_multimodal_for_mm_prefix.__get__(
            model_config, ModelConfig
        )
    )
    return model_config


@pytest.mark.parametrize("supports_mm", [True, False])
def test_supports_multimodal_for_mm_prefix_uses_registry(supports_mm: bool):
    model_config = _make_mm_prefix_model_config()

    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.supports_multimodal_inputs",
        return_value=supports_mm,
    ) as mocked:
        assert model_config._supports_multimodal_for_mm_prefix() is supports_mm
        mocked.assert_called_once_with(model_config)

    # Sticky cache — registry must not be consulted again.
    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.supports_multimodal_inputs",
        side_effect=AssertionError("should use cache"),
    ):
        assert model_config._supports_multimodal_for_mm_prefix() is supports_mm


def test_supports_multimodal_for_mm_prefix_before_multimodal_config():
    model_config = _make_mm_prefix_model_config()
    model_config.multimodal_config = None

    assert model_config._supports_multimodal_for_mm_prefix() is True
    assert not hasattr(model_config, "_supports_multimodal_inputs_cached")


def test_language_model_only_disables_via_supports_multimodal_inputs():
    """language_model_only zeros all limits, so registry reports text-only."""
    model_config = _make_mm_prefix_model_config(language_model_only=True)

    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.supports_multimodal_inputs",
        return_value=False,
    ):
        assert model_config._supports_multimodal_for_mm_prefix() is False


def test_convertor_clears_mm_prefix_when_multimodal_disabled():
    hf_config = PretrainedConfig(
        model_type="gemma3",
        architectures=["Gemma3ForConditionalGeneration"],
    )
    hf_config.is_mm_prefix_lm = True
    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)

    assert convertor.is_mm_prefix_lm(supports_multimodal=True) is True
    assert convertor.is_mm_prefix_lm(supports_multimodal=False) is False

    enabled = convertor.convert(supports_multimodal=True)
    disabled = convertor.convert(supports_multimodal=False)
    assert enabled.is_mm_prefix_lm is True
    assert disabled.is_mm_prefix_lm is False


def test_sticky_cache_survives_text_subconfig_regeneration():
    """with_hf_config deepcopies the cached decision onto text submodules."""
    model_config = _make_mm_prefix_model_config()
    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.supports_multimodal_inputs",
        return_value=False,
    ):
        assert model_config._supports_multimodal_for_mm_prefix() is False

    # Simulate deepcopy onto a Gemma4ForCausalLM-like config that would
    # otherwise fail registry lookup / return False incorrectly.
    text_config = _make_mm_prefix_model_config()
    text_config._supports_multimodal_inputs_cached = (
        model_config._supports_multimodal_inputs_cached
    )
    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.supports_multimodal_inputs",
        side_effect=AssertionError("must not re-query registry"),
    ):
        assert text_config._supports_multimodal_for_mm_prefix() is False
