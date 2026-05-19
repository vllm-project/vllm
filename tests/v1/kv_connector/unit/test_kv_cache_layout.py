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


# -- reshape_kv_cache: shape correctness -------------------------------------


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


# -- reshape_kv_cache: data round-trip ---------------------------------------


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


# -- reshape_kv_cache: slot independence --------------------------------------


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_reshape_slot_independence(layout):
    spec = _make_attn_spec()
    num_slots = 3
    raw = _alloc_raw_buffer(spec, NUM_BLOCKS, num_slots)
    views = reshape_kv_cache(raw, spec, NUM_BLOCKS, num_slots, layout)

    for i, v in enumerate(views):
        v.fill_(float(i + 1))

    views2 = reshape_kv_cache(raw, spec, NUM_BLOCKS, num_slots, layout)
    for i, v in enumerate(views2):
        assert torch.all(v == float(i + 1)), f"Slot {i} corrupted: expected {i + 1}"


# -- reshape_kv_cache: non-attention (MambaSpec) path -------------------------


def test_reshape_mamba_spec():
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((4,), (8,)),
        dtypes=(torch.float32, torch.float32),
    )
    num_slots = 2
    slot_bytes = mamba_spec.page_size_bytes * NUM_BLOCKS
    raw = torch.arange(slot_bytes * num_slots, dtype=torch.int8, device="cuda")
    views = reshape_kv_cache(raw, mamba_spec, NUM_BLOCKS, num_slots, KVCacheLayout.HNC)

    assert len(views) == num_slots
    for v in views:
        assert v.shape == (slot_bytes,)
    assert torch.equal(views[0], raw[:slot_bytes])
    assert torch.equal(views[1], raw[slot_bytes:])


# -- reshape_kv_cache: page_size_padded (as_strided path) --------------------


def test_reshape_page_size_padded():
    spec = _make_attn_spec()
    padded_page = spec.page_size_bytes + 64
    padded_spec = _make_attn_spec(page_size_padded=padded_page)
    num_slots = 1
    raw = torch.zeros(
        padded_page * NUM_BLOCKS * num_slots, dtype=torch.int8, device="cuda"
    )

    views = reshape_kv_cache(raw, padded_spec, NUM_BLOCKS, num_slots, KVCacheLayout.HNC)

    expected_4d = compute_kv_cache_shape(padded_spec, NUM_BLOCKS)
    assert views[0].shape == expected_4d

    views[0][0, 0, 0, 0] = 99.0
    assert views[0][0, 0, 0, 0].item() == 99.0


# -- reshape_kv_cache: layouts produce identical logical data -----------------


def test_reshape_all_layouts_same_logical_data():
    spec = _make_attn_spec()
    num_slots = 2
    shape_4d = compute_kv_cache_shape(spec, NUM_BLOCKS)
    reference = torch.randn(num_slots, *shape_4d, dtype=DTYPE, device="cuda")

    for layout in KVCacheLayout:
        raw = _alloc_raw_buffer(spec, NUM_BLOCKS, num_slots)
        views = reshape_kv_cache(raw, spec, NUM_BLOCKS, num_slots, layout)
        for i in range(num_slots):
            views[i].copy_(reference[i])

        views2 = reshape_kv_cache(raw, spec, NUM_BLOCKS, num_slots, layout)
        for i in range(num_slots):
            assert torch.equal(views2[i], reference[i]), (
                f"Layout {layout.name} slot {i} data mismatch"
            )


# -- _validate_layout_compatibility ------------------------------------------


@pytest.fixture(autouse=True)
def _reset_layout():
    yield
    set_kv_cache_layout(None)


def _make_group(layer_names, **spec_overrides):
    spec = _make_attn_spec(**spec_overrides)
    return KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)


def _make_tensor(shared_by, size=1024):
    return KVCacheTensor(size=size, shared_by=shared_by)


@pytest.mark.parametrize("layout_name", ["HNC", "BLHNC"])
def test_validate_block_contiguous_always_passes(layout_name):
    set_kv_cache_layout(layout_name)
    group_a = _make_group(["layer.0"], num_kv_heads=4, head_size=64)
    group_b = _make_group(["layer.1"], num_kv_heads=8, head_size=32)
    tensor = _make_tensor([["layer.0"], ["layer.1"]])
    _validate_layout_compatibility([tensor], [group_a, group_b])


@pytest.mark.parametrize("layout_name", ["NHC", "BHLNC"])
def test_validate_non_block_contiguous_uniform_passes(layout_name):
    set_kv_cache_layout(layout_name)
    group_a = _make_group(["layer.0"], num_kv_heads=4, head_size=64)
    group_b = _make_group(["layer.1"], num_kv_heads=4, head_size=64)
    tensor = _make_tensor([["layer.0"], ["layer.1"]])
    _validate_layout_compatibility([tensor], [group_a, group_b])


@pytest.mark.parametrize("layout_name", ["NHC", "BHLNC"])
def test_validate_non_block_contiguous_mismatched_raises(layout_name):
    set_kv_cache_layout(layout_name)
    group_a = _make_group(["layer.0"], num_kv_heads=4, head_size=64)
    group_b = _make_group(["layer.1"], num_kv_heads=8, head_size=32)
    tensor = _make_tensor([["layer.0"], ["layer.1"]])
    with pytest.raises(ValueError, match="different"):
        _validate_layout_compatibility([tensor], [group_a, group_b])


def test_validate_single_layer_always_passes():
    set_kv_cache_layout("NHC")
    group = _make_group(["layer.0"], num_kv_heads=4, head_size=64)
    tensor = _make_tensor([["layer.0"]])
    _validate_layout_compatibility([tensor], [group])


def test_validate_mismatched_head_size_only():
    set_kv_cache_layout("NHC")
    group_a = _make_group(["layer.0"], num_kv_heads=4, head_size=64)
    group_b = _make_group(["layer.1"], num_kv_heads=4, head_size=128)
    tensor = _make_tensor([["layer.0"], ["layer.1"]])
    with pytest.raises(ValueError, match="different"):
        _validate_layout_compatibility([tensor], [group_a, group_b])
