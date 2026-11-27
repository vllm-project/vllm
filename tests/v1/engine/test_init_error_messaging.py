# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.core.utils import full_layer, get_kv_cache_configs
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.v1.kv_cache_interface import AttentionSpec


def _make_config(
    *,
    enable_extensible_kv_cache: bool,
    gpu_memory_utilization: float | None = None,
) -> VllmConfig:
    config = VllmConfig(
        model_config=ModelConfig(max_model_len=2048),
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=gpu_memory_utilization,
        ),
    )
    config.cache_config.enable_extensible_kv_cache = enable_extensible_kv_cache
    config.cache_config.kv_cache_layout = "LBNHC"
    return config


def test_kv_cache_oom_no_memory():
    config = _make_config(enable_extensible_kv_cache=False)
    spec = {
        "layer_0": full_layer(
            AttentionSpec(
                block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
            )
        )
    }

    with pytest.raises(ValueError, match="No available memory for the cache blocks"):
        get_kv_cache_configs(config, [spec], [0])


def test_kv_cache_oom_insufficient_memory():
    config = _make_config(
        enable_extensible_kv_cache=True,
        gpu_memory_utilization=0.9,
    )
    attn_spec = AttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
    )
    spec = {"layer_0": full_layer(attn_spec)}

    # An explicit utilization below 1.0, not the device, limits the cache.
    # 2048 tokens needs 128 blocks; 4 blocks leaves 3 usable after null block.
    with pytest.raises(ValueError, match=r"within `gpu_memory_utilization`=0\.9"):
        get_kv_cache_configs(config, [spec], [attn_spec.page_size_bytes * 4])
