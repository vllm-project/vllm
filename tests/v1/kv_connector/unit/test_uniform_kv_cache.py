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
    """NHD backend: layers dim sits right after blocks."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str=None
    ):
        return (num_blocks, 2, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if include_num_layers_dimension:
            # logical_with_layers: (L, B, 2, H, bs, d)
            # physical:            (B, L, 2, H, bs, d)
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)


class _MockHNDBackend(_MockBackend):
    """HND backend: heads come before layers in physical order."""

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if include_num_layers_dimension:
            # logical_with_layers: (L, B, 2, H, bs, d)
            # physical:            (B, H, L, 2, bs, d)
            return (1, 3, 0, 2, 4, 5)
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


def _allocate(
    num_blocks,
    num_layers,
    backend=_MockBackend,
    prefix="l",
    kernel_block_sizes=None,
    attn_groups=None,
):
    """Shorthand for allocate_hybrid_kv_caches with FullAttentionSpec."""
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )
    names = [f"{prefix}.{i}" for i in range(num_layers)]
    if attn_groups is None:
        attn_groups = [_make_group(group_id=0, layer_names=names, backend=backend)]
    if kernel_block_sizes is None:
        kernel_block_sizes = [BLOCK_SIZE] * len(attn_groups)
    return KVConnectorModelRunnerMixin.allocate_hybrid_kv_caches(
        kv_cache_config=KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=spec.page_size_bytes * num_blocks,
                    shared_by=[n],
                )
                for n in names
            ],
            kv_cache_groups=[],
        ),
        attn_groups=attn_groups,
        cache_dtype="auto",
        device=torch.device("cpu"),
        kernel_block_sizes=kernel_block_sizes,
    )


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
        KVConnectorModelRunnerMixin.allocate_hybrid_kv_caches(
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
    # NHD backend -- default layout (blocks, layers, page_size)
    assert cross_layer_groups[0].tensor.ndim == 3
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

    kv, groups = KVConnectorModelRunnerMixin.allocate_hybrid_kv_caches(
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
    # Mamba -- default layout (blocks, layers, page_size)
    assert groups[0].tensor.ndim == 3
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


def test_hnd_backend_extracts_heads():
    """HND backend: heads before layers in physical order."""
    nb, num_layers = 4, 3
    kv_caches, groups = _allocate(
        nb, num_layers, backend=_MockHNDBackend, prefix="layer"
    )

    assert len(groups) == 1
    group = groups[0]

    # HND backend -- heads-first layout (blocks, heads, layers, per_head_page)
    spec = group.spec
    per_head_page = spec.page_size_bytes // NUM_KV_HEADS
    assert group.tensor.shape == (nb, NUM_KV_HEADS, num_layers, per_head_page)

    expected = (nb, 2, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
    for i in range(num_layers):
        assert kv_caches[f"layer.{i}"].shape == expected


def test_hnd_head_contiguity():
    """One block + one head across all layers is contiguous in HND layout."""
    _, groups = _allocate(4, 2, backend=_MockHNDBackend)

    group = groups[0]
    block_head = group.tensor[0, 0]  # (layers, per_head_page)
    assert block_head.is_contiguous()
    # H varies slower than layers
    assert group.tensor.stride(1) > group.tensor.stride(2)


def test_nhd_backend_uses_default_layout():
    """NHD backend places layers right after blocks -- default layout."""
    nb, num_layers = 4, 2
    kv_caches, groups = _allocate(nb, num_layers, backend=_MockBackend)

    assert len(groups) == 1
    group = groups[0]
    # NHD backend -- default layout (blocks, layers, page_size)
    assert group.tensor.shape == (nb, num_layers, group.spec.page_size_bytes)

    expected = (nb, 2, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
    for i in range(num_layers):
        assert kv_caches[f"l.{i}"].shape == expected
