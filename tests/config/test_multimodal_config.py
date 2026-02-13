# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

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


def test_language_model_only_does_not_affect_hash():
    """language_model_only does not affect the ViT computation graph,
    so it should not change the hash."""
    base_hash = MultiModalConfig().compute_hash()
    lm_only_hash = MultiModalConfig(language_model_only=True).compute_hash()
    assert base_hash == lm_only_hash
