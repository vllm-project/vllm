# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for uniform cross-layer KV cache allocation."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheTensor,
    MambaSpec,
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


class _MockMambaBackend:
    @staticmethod
    def get_kv_cache_shape(*a, **kw):
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_stride_order(*a, **kw):
        raise NotImplementedError


def _make_group(
    group_id=0,
    spec_cls=FullAttentionSpec,
    layer_names=None,
    num_kv_heads=NUM_KV_HEADS,
    backend=_MockBackend,
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
            backend=backend,
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


def test_different_page_sizes_accepted():
    """Groups with different page_size_bytes are accepted (separate groups)."""
    assert _use_uniform(
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

    kv_caches, cross_layer_groups = (
        KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
            kv_cache_config=kv_cache_config,
            attn_groups=[
                _make_group(group_id=0, layer_names=[f"full.{i}" for i in range(4)]),
                _make_group(
                    group_id=1,
                    spec_cls=SlidingWindowSpec,
                    layer_names=[f"sw.{i}" for i in range(4)],
                ),
            ],
            cache_dtype="auto",
            device=torch.device("cpu"),
            kernel_block_sizes=[BLOCK_SIZE, BLOCK_SIZE],
        )
    )

    assert len(kv_caches) == 8
    assert len(cross_layer_groups) == 1
    assert not cross_layer_groups[0].tp_layout
    for i in range(num_positions):
        assert kv_caches[f"full.{i}"].data_ptr() == kv_caches[f"sw.{i}"].data_ptr()


def test_mamba_allocation():
    """Mamba layers produce list[Tensor] views with data isolation."""
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((4, 2), (8,)),
        dtypes=(torch.float32, torch.float32),
    )
    nb = 2

    kv, groups = KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
        kv_cache_config=KVCacheConfig(
            num_blocks=nb,
            kv_cache_tensors=[
                KVCacheTensor(size=spec.page_size_bytes * nb, shared_by=[f"m.{i}"])
                for i in range(2)
            ],
            kv_cache_groups=[],
        ),
        attn_groups=[
            [
                AttentionGroup(
                    backend=_MockMambaBackend,
                    layer_names=["m.0", "m.1"],
                    kv_cache_spec=spec,
                    kv_cache_group_id=0,
                )
            ]
        ],
        cache_dtype="auto",
        device=torch.device("cpu"),
        kernel_block_sizes=[BLOCK_SIZE],
    )

    assert len(groups) == 1
    assert not groups[0].tp_layout
    for n in ["m.0", "m.1"]:
        assert isinstance(kv[n], list) and len(kv[n]) == 2
        assert kv[n][0].shape == (nb, 4, 2)
        assert kv[n][1].shape == (nb, 8)

    # Data isolation: writing to one layer shouldn't affect the other
    kv["m.0"][0][0].fill_(42.0)
    kv["m.1"][0][1].fill_(99.0)
    assert torch.all(kv["m.0"][0][0] == 42.0)
    assert torch.all(kv["m.1"][0][1] == 99.0)
    assert torch.all(kv["m.1"][0][0] == 0.0)
    assert torch.all(kv["m.0"][0][1] == 0.0)


def test_tp_layout_shape():
    """With tp_size > 1, the backing tensor uses (NB, H, layers, per_head_page)."""
    num_layers = 3
    nb = 4
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )

    kv_caches, groups = KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
        kv_cache_config=KVCacheConfig(
            num_blocks=nb,
            kv_cache_tensors=[
                KVCacheTensor(size=spec.page_size_bytes * nb, shared_by=[f"layer.{i}"])
                for i in range(num_layers)
            ],
            kv_cache_groups=[],
        ),
        attn_groups=[
            _make_group(
                group_id=0,
                layer_names=[f"layer.{i}" for i in range(num_layers)],
            )
        ],
        cache_dtype="auto",
        device=torch.device("cpu"),
        kernel_block_sizes=[BLOCK_SIZE],
        tp=True,
    )

    assert len(groups) == 1
    group = groups[0]
    assert group.tp_layout

    per_head_page = spec.page_size_bytes // NUM_KV_HEADS
    assert group.tensor.shape == (nb, NUM_KV_HEADS, num_layers, per_head_page)

    # Per-layer views should match the backend's logical shape
    # _MockBackend: (num_blocks, 2, num_kv_heads, block_size, head_size)
    expected = (nb, 2, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
    for i in range(num_layers):
        assert kv_caches[f"layer.{i}"].shape == expected


def test_tp_layout_head_contiguity():
    """Slicing a subset of heads from TP-layout tensor is contiguous."""
    nb = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )

    _, groups = KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
        kv_cache_config=KVCacheConfig(
            num_blocks=nb,
            kv_cache_tensors=[
                KVCacheTensor(size=spec.page_size_bytes * nb, shared_by=[f"l.{i}"])
                for i in range(num_layers)
            ],
            kv_cache_groups=[],
        ),
        attn_groups=[
            _make_group(
                group_id=0,
                layer_names=[f"l.{i}" for i in range(num_layers)],
            )
        ],
        cache_dtype="auto",
        device=torch.device("cpu"),
        kernel_block_sizes=[BLOCK_SIZE],
        tp=True,
    )

    # One block's per-head data (all layers) should be contiguous
    group = groups[0]
    block_head = group.tensor[0, 0]  # (layers, per_head_page)
    assert block_head.is_contiguous()
    # And head dim comes before layers in memory (H varies slower)
    assert group.tensor.stride(1) > group.tensor.stride(2)


def test_tp_size_1_default_layout():
    """With tp_size=1, the tensor uses the default (NB, layers, page) layout."""
    nb = 4
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )

    _, groups = KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
        kv_cache_config=KVCacheConfig(
            num_blocks=nb,
            kv_cache_tensors=[
                KVCacheTensor(size=spec.page_size_bytes * nb, shared_by=[f"l.{i}"])
                for i in range(2)
            ],
            kv_cache_groups=[],
        ),
        attn_groups=[_make_group(group_id=0, layer_names=["l.0", "l.1"])],
        cache_dtype="auto",
        device=torch.device("cpu"),
        kernel_block_sizes=[BLOCK_SIZE],
        tp=False,
    )

    assert len(groups) == 1
    assert not groups[0].tp_layout
    assert groups[0].tensor.shape == (nb, 2, spec.page_size_bytes)
