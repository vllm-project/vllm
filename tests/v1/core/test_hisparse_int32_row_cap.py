# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HiSparse must not hand AITER a KV row index it cannot address.

ROCm's AITER sparse MLA decode kernel computes its KV row byte offset in int32.
Above ``(2**31 - 1) // row_width`` it wraps and answers with near-zero
attention output -- no fault, no NaN, just wrong numbers, which surfaces as a
silent accuracy collapse (measured: GSM8K 0.905 -> 0.0 on GLM-5.3). HiSparse is
the only caller that gets near the limit, because it points attention at a view
over the whole multi-layer KV slab rather than at one layer's paged cache.

``_cap_num_blocks_for_int32_rows`` is the guard. These tests pin the arithmetic
against the layout measured on GLM-5.3 TP4/gfx950, where the cap must land on
58254 blocks, and pin that it is inert when it should be.
"""

import torch

from vllm.v1.hisparse.layout import HiSparseLayout, _cap_num_blocks_for_int32_rows
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)

# GLM-5.3 TP4: 576-wide bf16 rows, 16 tokens per block, 4 layers packed into
# each block of the shared slab -- so one block spans 64 attention rows.
BLOCK_SIZE = 16
ROW_WIDTH = 576
ROW_BYTES = ROW_WIDTH * 2
PAGE_SIZE = ROW_BYTES * BLOCK_SIZE
BYTES_PER_BLOCK = PAGE_SIZE * 4

MAX_ROWS = (2**31 - 1) // ROW_WIDTH  # 3728270
MAX_BLOCKS = MAX_ROWS // 64  # 58254


def _layout(*, page_size: int = PAGE_SIZE) -> HiSparseLayout:
    source_spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=ROW_WIDTH,
        dtype=torch.bfloat16,
    )
    hot_spec = HiSparseHotSpec(
        block_size=BLOCK_SIZE,
        page_size=page_size,
        blocks_per_request=256,
    )
    return HiSparseLayout(
        source_group=KVCacheGroupSpec(["layer.0"], source_spec),
        device_groups=[KVCacheGroupSpec(["layer.0.hisparse_hot"], hot_spec)],
        host_num_blocks=0,
        host_block_stride=0,
        shared_host_pool=False,
    )


def test_caps_an_overflowing_block_count(monkeypatch):
    """A pool the memory budget allows but the kernel cannot address is cut."""
    monkeypatch.setattr(
        "vllm.platforms.current_platform.is_rocm", lambda: True, raising=False
    )
    capped = _cap_num_blocks_for_int32_rows(1_160_787, _layout(), BYTES_PER_BLOCK)
    assert capped == MAX_BLOCKS
    assert capped * 64 * ROW_WIDTH <= 2**31 - 1


def test_leaves_a_safe_block_count_alone(monkeypatch):
    """The cap must not shrink pools that were already addressable."""
    monkeypatch.setattr(
        "vllm.platforms.current_platform.is_rocm", lambda: True, raising=False
    )
    assert _cap_num_blocks_for_int32_rows(1000, _layout(), BYTES_PER_BLOCK) == 1000
    assert (
        _cap_num_blocks_for_int32_rows(MAX_BLOCKS, _layout(), BYTES_PER_BLOCK)
        == MAX_BLOCKS
    )


def test_inert_off_rocm(monkeypatch):
    """CUDA's kernels address rows in 64-bit; capping there would cost cache."""
    monkeypatch.setattr(
        "vllm.platforms.current_platform.is_rocm", lambda: False, raising=False
    )
    assert (
        _cap_num_blocks_for_int32_rows(1_160_787, _layout(), BYTES_PER_BLOCK)
        == 1_160_787
    )
