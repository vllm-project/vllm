# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon attention layers use the KV cache dtype from --kv-cache-dtype."""

import pytest
from transformers import FalconConfig

from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.models.falcon import FalconAttention
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="needs a CUDA attention backend"
)
@pytest.mark.parametrize("alibi", [False, True], ids=["rotary", "alibi"])
def test_falcon_attention_honors_kv_cache_dtype(dist_init, alibi):
    cache_config = CacheConfig(cache_dtype="bfloat16")
    config = FalconConfig(hidden_size=256, num_attention_heads=4, alibi=alibi)
    with set_current_vllm_config(VllmConfig(cache_config=cache_config)):
        layer = FalconAttention(config, cache_config=cache_config, prefix="h.0")
    assert layer.attn.kv_cache_dtype == "bfloat16"
