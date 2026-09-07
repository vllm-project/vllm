# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``fp8_ds_mla`` is the packed DeepSeek-MLA cache layout; a regular attention
layer built under it (a GQA drafter next to a DS-MLA target) must keep the
model dtype instead of failing backend selection."""

import os
import tempfile

import pytest
import torch

from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@pytest.fixture
def single_rank_dist():
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)
            yield
        cleanup_dist_env_and_memory()
    finally:
        try:
            os.unlink(temp_file)
        except OSError:
            pass


def _regular_attention_backend():
    if current_platform.is_cpu():
        return AttentionBackendEnum.CPU_ATTN.get_class()
    return AttentionBackendEnum.FLASH_ATTN.get_class()


@pytest.mark.parametrize(
    ("cache_dtype", "expected"),
    [
        ("fp8_ds_mla", "auto"),
        ("auto", "auto"),
        ("fp8", "fp8"),
    ],
)
def test_regular_attention_layer_kv_cache_dtype(
    single_rank_dist, cache_dtype: str, expected: str
):
    vllm_config = VllmConfig(model_config=ModelConfig(dtype="bfloat16"))
    with set_current_vllm_config(vllm_config):
        attn = Attention(
            num_heads=8,
            head_size=128,
            scale=128**-0.5,
            num_kv_heads=4,
            cache_config=CacheConfig(cache_dtype=cache_dtype),
            prefix="draft.model.layers.0.self_attn.attn",
            attn_backend=_regular_attention_backend(),
        )
    assert attn.kv_cache_dtype == expected
    if expected == "auto":
        # The model dtype, never a byte-packed cache dtype.
        assert attn.kv_cache_torch_dtype == torch.bfloat16
