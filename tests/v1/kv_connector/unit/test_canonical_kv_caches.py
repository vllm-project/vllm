# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CanonicalKVCaches abstraction."""

from unittest.mock import patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.v1.kv_cache_interface import (
    CanonicalKVCaches,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
)
from vllm.v1.worker.utils import AttentionGroup

# ---------------------------------------------------------------------------
# Mock backends and connectors
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16
NUM_KV_HEADS = 4
HEAD_SIZE = 8
NUM_BLOCKS = 10
DTYPE = torch.float16


class MockFlashAttnBackend:
    """Mimics FlashAttention NHD layout."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def get_kv_cache_block_dim(
        cls, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        _S = 1234567
        shape = cls.get_kv_cache_shape(
            _S, block_size, num_kv_heads, head_size, cache_dtype_str
        )
        return shape.index(_S)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if include_num_layers_dimension:
            return (2, 0, 1, 3, 4, 5)
        return (0, 1, 2, 3, 4)


class MockNoStrideOrderBackend:
    """Backend that does not support stride order."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        raise NotImplementedError


class MockConnector(SupportsHMA):
    prefer_cross_layer_blocks = True

    def request_finished_all_groups(self, request, block_ids):
        return False, None


class MockConnectorNoHMA:
    prefer_cross_layer_blocks = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_full_attn_spec():
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )


def _make_sw_spec(sliding_window=128):
    return SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
        sliding_window=sliding_window,
    )


def _make_hma_kv_cache_config():
    """HMA config: 3 groups, group_size=2, 2 KVCacheTensors."""
    full_spec = _make_full_attn_spec()
    sw_spec = _make_sw_spec()
    page_size = full_spec.page_size_bytes

    groups = [
        KVCacheGroupSpec(["full.0", "full.1"], full_spec),
        KVCacheGroupSpec(["sw.0", "sw.2"], sw_spec),
        KVCacheGroupSpec(["sw.1", "sw.3"], sw_spec),
    ]
    size = page_size * NUM_BLOCKS
    tensors = [
        KVCacheTensor(size=size, shared_by=["full.0", "sw.0", "sw.1"]),
        KVCacheTensor(size=size, shared_by=["full.1", "sw.2", "sw.3"]),
    ]
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def _make_attn_groups(backend_cls, kv_cache_config):
    attn_groups = []
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        attn_groups.append(
            [
                AttentionGroup(
                    backend=backend_cls,
                    layer_names=group.layer_names,
                    kv_cache_spec=group.kv_cache_spec,
                    kv_cache_group_id=gid,
                )
            ]
        )
    return attn_groups


def _patch_connector(connector):
    return (
        patch(
            "vllm.v1.worker.kv_connector_model_runner_mixin.has_kv_transfer_group",
            return_value=True,
        ),
        patch(
            "vllm.v1.worker.kv_connector_model_runner_mixin.get_kv_transfer_group",
            return_value=connector,
        ),
    )


def _use_canonical(config, attn_groups):
    return KVConnectorModelRunnerMixin.use_canonical_kv_caches(
        config, attn_groups, "auto"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.cpu_test
def test_use_canonical_kv_caches_happy_path():
    """Should return True for a valid HMA model with compatible connector."""
    config = _make_hma_kv_cache_config()
    attn_groups = _make_attn_groups(MockFlashAttnBackend, config)
    p1, p2 = _patch_connector(MockConnector())
    with p1, p2:
        assert _use_canonical(config, attn_groups) is True


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "description,config_fn,backend,connector_fn,patch_no_connector",
    [
        (
            "single_group",
            lambda: KVCacheConfig(
                num_blocks=NUM_BLOCKS,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_make_full_attn_spec().page_size_bytes * NUM_BLOCKS,
                        shared_by=["layer0"],
                    )
                ],
                kv_cache_groups=[KVCacheGroupSpec(["layer0"], _make_full_attn_spec())],
            ),
            MockFlashAttnBackend,
            MockConnector,
            False,
        ),
        (
            "no_connector",
            _make_hma_kv_cache_config,
            MockFlashAttnBackend,
            None,
            True,
        ),
        (
            "no_hma_support",
            _make_hma_kv_cache_config,
            MockFlashAttnBackend,
            MockConnectorNoHMA,
            False,
        ),
        (
            "mamba_group",
            lambda: KVCacheConfig(
                num_blocks=NUM_BLOCKS,
                kv_cache_tensors=[],
                kv_cache_groups=[
                    KVCacheGroupSpec(["attn.0"], _make_full_attn_spec()),
                    KVCacheGroupSpec(
                        ["mamba.0"],
                        MambaSpec(
                            block_size=BLOCK_SIZE,
                            shapes=((16,), (16,)),
                            dtypes=(DTYPE,),
                        ),
                    ),
                ],
            ),
            MockFlashAttnBackend,
            MockConnector,
            False,
        ),
        (
            "no_stride_order",
            _make_hma_kv_cache_config,
            MockNoStrideOrderBackend,
            MockConnector,
            False,
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_use_canonical_kv_caches_returns_false(
    description, config_fn, backend, connector_fn, patch_no_connector
):
    """Should return False when any precondition is not met."""
    config = config_fn()
    attn_groups = _make_attn_groups(backend, config)

    if patch_no_connector:
        with patch(
            "vllm.v1.worker.kv_connector_model_runner_mixin.has_kv_transfer_group",
            return_value=False,
        ):
            assert _use_canonical(config, attn_groups) is False
    else:
        p1, p2 = _patch_connector(connector_fn())
        with p1, p2:
            assert _use_canonical(config, attn_groups) is False


@pytest.mark.cpu_test
def test_allocate_canonical_kv_caches():
    """Allocation should produce correct kv_caches dict and
    CanonicalKVCaches with contiguous per-block data."""
    config = _make_hma_kv_cache_config()
    attn_groups = _make_attn_groups(MockFlashAttnBackend, config)

    kv_caches, canonical = KVConnectorModelRunnerMixin.allocate_canonical_kv_caches(
        config, attn_groups, "auto", torch.device("cpu"), [BLOCK_SIZE]
    )

    assert isinstance(canonical, CanonicalKVCaches)

    # -- kv_caches dict: all 6 layers present with correct shapes
    expected_shape = (2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert len(kv_caches) == 6
    for name in ["full.0", "full.1", "sw.0", "sw.1", "sw.2", "sw.3"]:
        assert kv_caches[name].shape == expected_shape

    # layers sharing a position point to the same memory
    assert kv_caches["full.0"].data_ptr() == kv_caches["sw.0"].data_ptr()
    assert kv_caches["full.1"].data_ptr() == kv_caches["sw.2"].data_ptr()

    # -- block tensors: 2 positions * 2 splits (K/V) = 4
    assert len(canonical.tensors) == 4
    for bt in canonical.tensors:
        assert bt.tensor.shape == (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)

    # contiguity: V0 starts right after K0, K1 starts right after V0
    k_block_bytes = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * DTYPE.itemsize
    ptrs = [bt.tensor.data_ptr() for bt in canonical.tensors]
    assert ptrs[1] - ptrs[0] == k_block_bytes  # K0 -> V0
    assert ptrs[2] - ptrs[1] == k_block_bytes  # V0 -> K1

    # -- group_data_refs: 3 groups, each with 2 layers * 2 splits = 4 refs
    assert len(canonical.group_data_refs) == 3
    for refs in canonical.group_data_refs:
        assert len(refs) == 4
        assert [r.tensor_idx for r in refs] == [0, 1, 2, 3]

    # ref page_size = spec page_size // num_splits
    full_page = config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
    for refs in canonical.group_data_refs:
        for ref in refs:
            assert ref.page_size_bytes == full_page // 2
