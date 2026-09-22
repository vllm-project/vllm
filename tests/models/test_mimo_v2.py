# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo decoder cache policy and hybrid attention windows."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16"])
def test_mimo_decoder_preserves_cache_policy_and_windows(dist_init, cache_dtype):
    from vllm.model_executor.models.mimo_v2 import MiMoV2FlashDecoderLayer

    config = VllmConfig()
    config.cache_config = CacheConfig(
        block_size=16, cache_dtype=cache_dtype, sliding_window=128
    )
    config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN_DIFFKV
    config.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        is_mm_prefix_lm=False,
        hf_text_config=SimpleNamespace(
            hidden_size=64,
            intermediate_size=128,
            hidden_act="silu",
            layernorm_epsilon=1e-6,
            hybrid_layer_pattern=[0, 1],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=192,
            v_head_dim=128,
            swa_num_attention_heads=2,
            swa_num_key_value_heads=1,
            swa_head_dim=192,
            swa_v_head_dim=128,
            sliding_window_size=128,
            attention_bias=False,
            max_position_embeddings=256,
        ),
    )
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        layers = [
            MiMoV2FlashDecoderLayer(config, prefix=f"model.layers.{i}")
            for i in range(2)
        ]
        full, sliding = [layer.self_attn.attn for layer in layers]
        assert full.sliding_window is None
        assert sliding.sliding_window == 128
        assert config.cache_config.sliding_window == 128
        assert full.kv_cache_dtype == sliding.kv_cache_dtype == cache_dtype
        assert type(full.get_kv_cache_spec(config)) is FullAttentionSpec
        assert type(sliding.get_kv_cache_spec(config)) is SlidingWindowSpec
        assert full.get_kv_cache_spec(config).dtype == torch.bfloat16
        assert sliding.get_kv_cache_spec(config).dtype == torch.bfloat16
