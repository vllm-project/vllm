# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_utils import check_enough_kv_cache_memory
from vllm.v1.kv_cache_interface import FullAttentionSpec


def test_kv_cache_oom_no_memory():
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
    )

    spec = {
        "layer_0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
        )
    }

    with pytest.raises(ValueError, match="No available memory for the cache blocks"):
        check_enough_kv_cache_memory(config, spec, 0)


def test_kv_cache_oom_insufficient_memory():
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
    )

    spec = {
        "layer_0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
        )
    }

    with pytest.raises(
        ValueError, match="To serve at least one request with the model's max seq len"
    ):
        check_enough_kv_cache_memory(config, spec, 64 * spec["layer_0"].page_size_bytes)
    assert config.model_config.max_model_len == 2048
