# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for reshape_kv_cache and _validate_layout_compatibility."""

import pytest
import torch

from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.v1.core.kv_cache_utils import _validate_layout_compatibility
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
    compute_kv_cache_shape,
    reshape_kv_cache,
)

NUM_BLOCKS = 4
BLOCK_SIZE = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 8
DTYPE = torch.bfloat16


def _make_attn_spec(**overrides) -> FullAttentionSpec:
    defaults = dict(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    defaults.update(overrides)
    return FullAttentionSpec(**defaults)


def _alloc_raw_buffer(
    spec: FullAttentionSpec, num_blocks: int, num_slots: int
) -> torch.Tensor:
    total_bytes = spec.page_size_bytes * num_blocks * num_slots
    return torch.zeros(total_bytes, dtype=torch.int8, device="cuda")


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_reshape_shape_correctness(layout):
    spec = _make_attn_spec()
    num_slots = 2
    raw = _alloc_raw_buffer(spec, NUM_BLOCKS, num_slots)
    views = reshape_kv_cache(raw, spec, NUM_BLOCKS, num_slots, layout)

    expected_4d = compute_kv_cache_shape(spec, NUM_BLOCKS)
    assert len(views) == num_slots
    for v in views:
        assert v.shape == expected_4d


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_reshape_data_roundtrip(layout):
    spec = _make_attn_spec()
    num_slots = 1
    raw = _alloc_raw_buffer(spec, NUM_BLOCKS, num_slots)
    views = reshape_kv_cache(raw, spec, NUM_BLOCKS, num_slots, layout)
    view = views[0]

    view.fill_(0)
    block_idx, head_idx = 1, 0
    view[block_idx, head_idx, :, :] = 42.0

    recovered = reshape_kv_cache(raw, spec, NUM_BLOCKS, num_slots, layout)
    assert torch.all(recovered[0][block_idx, head_idx] == 42.0)
    assert recovered[0][0, 1].abs().sum() == 0


def test_reshape_mamba_spec():
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((4,), (8,)),
        dtypes=(torch.float32, torch.float32),
    )
    num_slots = 2
    slot_bytes = mamba_spec.page_size_bytes * NUM_BLOCKS
    raw = torch.arange(slot_bytes * num_slots, dtype=torch.int8, device="cuda")
    views = reshape_kv_cache(
        raw, mamba_spec, NUM_BLOCKS, num_slots, KVCacheLayout.LBHNC
    )

    assert len(views) == num_slots
    for v in views:
        assert v.shape == (slot_bytes,)
    assert torch.equal(views[0], raw[:slot_bytes])
    assert torch.equal(views[1], raw[slot_bytes:])


@pytest.fixture(autouse=True)
def _reset_layout():
    yield
    set_kv_cache_layout(None)


def _make_group(layer_names, **spec_overrides):
    spec = _make_attn_spec(**spec_overrides)
    return KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)


def _make_tensor(shared_by, size=1024):
    return KVCacheTensor(size=size, shared_by=shared_by)


def test_validate_non_block_contiguous_mismatched_raises():
    set_kv_cache_layout("LBNHC")
    group_a = _make_group(["layer.0"], num_kv_heads=4, head_size=64)
    group_b = _make_group(["layer.1"], num_kv_heads=8, head_size=32)
    tensor = _make_tensor([["layer.0"], ["layer.1"]])
    with pytest.raises(ValueError, match="different"):
        _validate_layout_compatibility([tensor], [group_a, group_b])
