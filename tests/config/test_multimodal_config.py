# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.model import ModelConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.config.utils import hash_factors
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_mm_encoder_attn_backend_str_conversion():
    config = MultiModalConfig(mm_encoder_attn_backend="FLASH_ATTN")
    assert config.mm_encoder_attn_backend == AttentionBackendEnum.FLASH_ATTN


def test_mm_encoder_attn_backend_invalid():
    with pytest.raises(ValueError):
        MultiModalConfig(mm_encoder_attn_backend="not_a_backend")


def test_mm_encoder_attn_backend_hash_updates():
    base_compile_signature = MultiModalConfig().compile_factors()
    overridden_compile_signature = MultiModalConfig(
        mm_encoder_attn_backend=AttentionBackendEnum.FLASH_ATTN
    ).compile_factors()
    assert base_compile_signature != overridden_compile_signature


def test_language_model_only_does_not_affect_mm_hash():
    """language_model_only does not affect the ViT computation graph,
    so it should not change the multimodal config hash."""
    base_hash = hash_factors(MultiModalConfig().compile_factors())
    lm_only_hash = hash_factors(
        MultiModalConfig(language_model_only=True).compile_factors()
    )
    assert base_hash == lm_only_hash


def test_language_model_only_affects_model_hash():
    """language_model_only affects the LM computation graph,
    so it should change the model config hash."""
    model = "llava-hf/llava-1.5-7b-hf"
    base_hash = hash_factors(ModelConfig(model).compile_factors())
    lm_only_hash = hash_factors(
        ModelConfig(model, language_model_only=True).compile_factors()
    )
    assert base_hash != lm_only_hash


def test_mm_encoder_fp8_scale_path_requires_fp8():
    with pytest.raises(ValueError, match="mm_encoder_attn_dtype"):
        MultiModalConfig(mm_encoder_fp8_scale_path="/tmp/scales.json")


def test_mm_encoder_attn_dtype_hash_updates(tmp_path):
    scale_file = tmp_path / "scales.json"
    scale_file.write_text("{}")
    base_hash = hash_factors(MultiModalConfig().compile_factors())
    fp8_hash = hash_factors(
        MultiModalConfig(mm_encoder_attn_dtype="fp8").compile_factors()
    )
    fp8_static_hash = hash_factors(
        MultiModalConfig(
            mm_encoder_attn_dtype="fp8",
            mm_encoder_fp8_scale_path=str(scale_file),
        ).compile_factors()
    )
    assert base_hash != fp8_hash
    assert fp8_hash != fp8_static_hash
