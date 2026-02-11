# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for multi-group HMA support in uniform KV cache allocation.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheTensor,
    SlidingWindowSpec,
)
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
)
from vllm.v1.worker.utils import AttentionGroup

pytestmark = pytest.mark.cpu_test

MODULE = "vllm.v1.worker.kv_connector_model_runner_mixin"

BLOCK_SIZE = 16
NUM_KV_HEADS = 4
HEAD_SIZE = 64


class _MockBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str=None
    ):
        return (num_blocks, 2, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)


def _make_group(
    group_id=0, spec_cls=FullAttentionSpec, layer_names=None, num_kv_heads=NUM_KV_HEADS
):
    kwargs = dict(
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )
    if spec_cls is SlidingWindowSpec:
        kwargs["sliding_window"] = 128
    return [
        AttentionGroup(
            backend=_MockBackend,
            layer_names=layer_names or [],
            kv_cache_spec=spec_cls(**kwargs),
            kv_cache_group_id=group_id,
        )
    ]


def _use_uniform(attn_groups):
    mock = MagicMock()
    mock.prefer_cross_layer_blocks = True
    with (
        patch(f"{MODULE}.has_kv_transfer_group", return_value=True),
        patch(f"{MODULE}.get_kv_transfer_group", return_value=mock),
    ):
        return KVConnectorModelRunnerMixin.use_uniform_kv_cache(attn_groups, "auto")


def test_multi_group_compatible():
    """Two groups (full + sliding window) with same shape are compatible."""
    assert _use_uniform(
        [
            _make_group(group_id=0, spec_cls=FullAttentionSpec),
            _make_group(group_id=1, spec_cls=SlidingWindowSpec),
        ]
    )


def test_multi_group_incompatible():
    """Groups with different num_kv_heads are rejected."""
    assert not _use_uniform(
        [
            _make_group(num_kv_heads=4, group_id=0),
            _make_group(num_kv_heads=8, group_id=1),
        ]
    )


def test_allocate_multi_group_shared_tensors():
    """Allocation shares memory across groups at each position."""
    num_positions = 4
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * 4, shared_by=[f"full.{i}", f"sw.{i}"]
            )
            for i in range(num_positions)
        ],
        kv_cache_groups=[],
    )

    kv_caches, cross_layer, _ = KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
        kv_cache_config=kv_cache_config,
        attn_groups=[
            _make_group(group_id=0, layer_names=[f"full.{i}" for i in range(4)]),
            _make_group(group_id=1, layer_names=[f"sw.{i}" for i in range(4)]),
        ],
        cache_dtype="auto",
        device=torch.device("cpu"),
        kernel_block_sizes=[BLOCK_SIZE, BLOCK_SIZE],
    )

    assert len(kv_caches) == 8
    for i in range(num_positions):
        assert kv_caches[f"full.{i}"].data_ptr() == kv_caches[f"sw.{i}"].data_ptr()
    assert cross_layer.shape[0] == num_positions


def test_allocate_rejects_mismatched_kernel_block_sizes():
    """Different kernel_block_sizes across groups are rejected."""
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )

    with pytest.raises(AssertionError, match="same kernel block size"):
        KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
            kv_cache_config=KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(size=spec.page_size_bytes * 4, shared_by=["l0"])
                ],
                kv_cache_groups=[],
            ),
            attn_groups=[_make_group()],
            cache_dtype="auto",
            device=torch.device("cpu"),
            kernel_block_sizes=[16, 32],
        )
