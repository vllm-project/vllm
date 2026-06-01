# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for reshape_kv_cache."""

import pytest
import torch

from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheLayout,
    MambaSpec,
    compute_layer_kv_cache_shape_bytes,
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

    byte_4d = compute_layer_kv_cache_shape_bytes(spec, NUM_BLOCKS)
    dtype_size = torch.tensor([], dtype=spec.dtype).element_size()
    expected_4d = (*byte_4d[:3], byte_4d[3] // dtype_size)
    assert len(views) == num_slots
    for v in views:
        assert v.shape == expected_4d
        assert v.dtype == spec.dtype


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
    expected_shape = (
        NUM_BLOCKS,
        mamba_spec.num_heads,
        1,
        mamba_spec.state_content_size_bytes,
    )
    for v in views:
        assert v.shape == expected_shape
        assert v.dtype == torch.int8


@pytest.fixture(autouse=True)
def _reset_layout():
    yield
    set_kv_cache_layout(None)
