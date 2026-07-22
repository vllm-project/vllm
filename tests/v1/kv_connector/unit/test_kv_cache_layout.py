# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for reshape_kv_cache."""

import pytest
import torch

from vllm.v1.attention.backends.utils import (
    get_flashinfer_layout_string,
    set_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheLayout,
    compute_layer_kv_cache_shape_bytes,
    reshape_kv_cache,
)

NUM_BLOCKS = 4
BLOCK_SIZE = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 8
DTYPE = torch.bfloat16


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ("LBHNC", "HND"),
        ("LBNHC", "NHD"),
        ("BLHNC", "HND"),
        ("BLNHC", "NHD"),
        ("BHLNC", "HND"),
    ],
)
def test_flashinfer_layout_string(layout: str, expected: str):
    set_kv_cache_layout(layout)
    try:
        assert get_flashinfer_layout_string() == expected
    finally:
        set_kv_cache_layout(None)


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_reshape_kv_cache(layout):
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    num_slots = 2
    total_bytes = spec.page_size_bytes * NUM_BLOCKS * num_slots
    raw = torch.zeros(total_bytes, dtype=torch.int8, device="cuda")
    views = reshape_kv_cache(raw, spec, NUM_BLOCKS, num_slots, layout)

    byte_4d = compute_layer_kv_cache_shape_bytes(spec, NUM_BLOCKS)
    dtype_size = torch.tensor([], dtype=spec.dtype).element_size()
    expected_shape = (*byte_4d[:3], byte_4d[3] // dtype_size)

    assert len(views) == num_slots
    for v in views:
        assert v.shape == expected_shape
        assert v.dtype == spec.dtype

    # The per-layer view preserves the physical order of B, H, N, and C
    # after the layer dimension is selected. Dimensions later in that order
    # have smaller strides, including when layer interleaving creates gaps.
    stride_order = layout.layer_view_order
    strides = views[0].stride()
    for i in range(3):
        for j in range(i + 1, 4):
            if stride_order[i] < stride_order[j]:
                assert strides[i] >= strides[j], (
                    f"layout {layout.name}: dim {i} (physical pos "
                    f"{stride_order[i]}) should have >= stride than "
                    f"dim {j} (physical pos {stride_order[j]}), got "
                    f"strides={strides}"
                )
