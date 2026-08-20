# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for weight loading in `QKVParallelLinear`."""

import contextlib
import os
import tempfile

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import QKVParallelLinear

HIDDEN_SIZE = 64
HEAD_SIZE = 8


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
        # FileStore may already have removed the file it was given.
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


def interleave(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_kv_heads: int):
    """Fuse [Q|K|V] into the per-KV-group interleaved layout found on disk."""
    trailing = q.shape[1:]
    groups = [t.reshape(num_kv_heads, -1, HEAD_SIZE, *trailing) for t in (q, k, v)]
    return torch.cat(groups, dim=1).reshape(-1, *trailing)


@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (8, 2), (8, 1)])
@pytest.mark.parametrize("is_bias", [False, True])
def test_fused_qkv_interleaved_weight_loader(
    single_rank_dist, num_heads: int, num_kv_heads: int, is_bias: bool
):
    """An interleaved fused checkpoint loads to the same [Q|K|V] as split weights."""
    shape = () if is_bias else (HIDDEN_SIZE,)
    q = torch.randn(num_heads * HEAD_SIZE, *shape)
    k = torch.randn(num_kv_heads * HEAD_SIZE, *shape)
    v = torch.randn(num_kv_heads * HEAD_SIZE, *shape)

    layer = QKVParallelLinear(
        HIDDEN_SIZE,
        HEAD_SIZE,
        num_heads,
        num_kv_heads,
        bias=True,
        params_dtype=torch.float32,
        fused_qkv_interleaved=True,
    )
    param = layer.bias if is_bias else layer.weight
    layer.weight_loader(param, interleave(q, k, v, num_kv_heads))

    assert torch.equal(param.data, torch.cat((q, k, v)))
