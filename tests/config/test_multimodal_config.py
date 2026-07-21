# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.config.model import ModelConfig
from vllm.config.model_arch import ModelArchitectureConfig
from vllm.config.multimodal import MultiModalConfig
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
    is_mm_prefix_lm: bool = True,
) -> ModelConfig:
    model_config = MagicMock(spec=ModelConfig)
    model_config.multimodal_config = MultiModalConfig(
        language_model_only=language_model_only
    )
    model_config.model_arch_config = ModelArchitectureConfig(
        architectures=["PrefixLMForConditionalGeneration"],
        model_type="prefix_lm",
        text_model_type=None,
        hidden_size=1,
        total_num_hidden_layers=1,
        total_num_attention_heads=1,
        head_size=1,
        vocab_size=1,
        total_num_kv_heads=1,
        num_experts=0,
        quantization_config=None,
        is_deepseek_mla=False,
        is_mm_prefix_lm=is_mm_prefix_lm,
        rswa_window=None,
        derived_max_model_len_and_key=(8192.0, "max_position_embeddings"),
    )
    return model_config


@pytest.mark.parametrize(
    ("supported_limits", "allowed_limits", "expected"),
    [
        ({"image": None, "video": None}, {"image": 1, "video": 1}, True),
        ({"image": None, "video": None}, {"image": 0, "video": 0}, False),
        ({"image": None}, {"image": 0}, False),
        (
            {"image": None, "video": None, "audio": None},
            {"image": 0, "video": 1, "audio": 1},
            True,
        ),
        ({"audio": None}, {"audio": 0}, True),
    ],
)
def test_mm_prefix_lm_respects_vision_limits(
    supported_limits: dict[str, int | None],
    allowed_limits: dict[str, int],
    expected: bool,
):
    model_config = _make_mm_prefix_model_config()
    info = SimpleNamespace(
        supported_mm_limits=supported_limits,
        allowed_mm_limits=allowed_limits,
    )

    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.get_processing_info",
        return_value=info,
    ):
        result = ModelConfig._apply_mm_prefix_lm_limits(
            model_config, model_config.model_arch_config
        )

    assert result.is_mm_prefix_lm is expected


def test_language_model_only_disables_mm_prefix_lm():
    model_config = _make_mm_prefix_model_config(language_model_only=True)

    result = ModelConfig._apply_mm_prefix_lm_limits(
        model_config, model_config.model_arch_config
    )

    assert not result.is_mm_prefix_lm


def test_mm_prefix_limits_noop_for_causal_model():
    model_config = _make_mm_prefix_model_config(is_mm_prefix_lm=False)
    original_arch = model_config.model_arch_config

    result = ModelConfig._apply_mm_prefix_lm_limits(model_config, original_arch)

    assert result is original_arch


def test_mm_prefix_limits_noop_before_multimodal_config():
    model_config = _make_mm_prefix_model_config()
    model_config.multimodal_config = None
    original_arch = model_config.model_arch_config

    result = ModelConfig._apply_mm_prefix_lm_limits(model_config, original_arch)

    assert result is original_arch


def test_mm_prefix_limits_sticky_across_text_subconfig():
    model_config = _make_mm_prefix_model_config(language_model_only=True)
    first = ModelConfig._apply_mm_prefix_lm_limits(
        model_config, model_config.model_arch_config
    )
    assert not first.is_mm_prefix_lm
    assert model_config._mm_prefix_lm_disabled is True

    # Simulate with_hf_config regenerating a text-only arch with the flag True.
    text_arch = replace(model_config.model_arch_config, is_mm_prefix_lm=True)
    second = ModelConfig._apply_mm_prefix_lm_limits(model_config, text_arch)
    assert not second.is_mm_prefix_lm


def test_mm_prefix_limits_skips_registry_for_text_architecture():
    model_config = _make_mm_prefix_model_config()
    original_arch = model_config.model_arch_config

    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.get_processing_info",
        side_effect=ValueError("no multimodal processor"),
    ):
        result = ModelConfig._apply_mm_prefix_lm_limits(model_config, original_arch)

    assert result is original_arch
    assert not getattr(model_config, "_mm_prefix_lm_disabled", False)
