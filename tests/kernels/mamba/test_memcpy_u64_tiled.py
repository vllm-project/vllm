# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for ``_memcpy_u64_tiled``.

Guards the head/body/tail alignment handling and the u64 body's tile
partitioning used by ``postprocess_mamba_fused_kernel`` and
``precopy_mamba_align_fused_kernel``. The device function must be
byte-identical to a ``dst[dst_off:dst_off+copy_size] = src[src_off:src_off+
copy_size]`` slice copy for every combination of:

* ``copy_size``: the degenerate 0-byte case, sub-8B (head-only), 8B-aligned
  bodies, and a multi-tile body that spans several ``COPY_BLOCK_SIZE``
  iterations.
* ``(src_off, dst_off)`` pairs: aligned, shared sub-8B misalignment (fast
  path exercising head/tail masking), and mismatched src/dst alignment
  (byte-fallback path). The kernel branches on ``(src ^ dst) & 7``, so a
  full 5x5 product would re-exercise the same two codepaths.
* ``NUM_TILES``: 1 (SD conv callsite, single-CTA memcpy) and
  ``_TEMPORAL_TILES`` (temporal callsite, u64 range partitioned across
  CTAs).
* ``COPY_BLOCK_SIZE``: the production value 1024 (single-tile at these
  sizes) and a small value 8 so the existing ``copy_size`` cases actually
  cross tile boundaries under ``NUM_TILES=_TEMPORAL_TILES``. The
  partitioning math is COPY_BLOCK_SIZE-agnostic, so testing at a small
  value covers boundary arithmetic for all values.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.mamba_utils import _TEMPORAL_TILES, _memcpy_u64_tiled

try:
    import pytest

    pytestmark = pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="_memcpy_u64_tiled needs CUDA/Triton",
    )
    _parametrize = pytest.mark.parametrize
except ModuleNotFoundError:  # allow running directly as ``python <thisfile>``
    pytest = None

    def _parametrize(_name, _values):
        def _deco(fn):
            return fn

        return _deco


@triton.jit
def _memcpy_wrapper_kernel(
    src_ptr,
    dst_ptr,
    copy_size,
    src_off,
    dst_off,
    COPY_BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    """Thin harness: one CTA per tile, exercising the device function."""
    tile_idx = tl.program_id(0)
    src_addr = src_ptr.to(tl.int64) + src_off
    dst_addr = dst_ptr.to(tl.int64) + dst_off
    _memcpy_u64_tiled(
        src_addr,
        dst_addr,
        copy_size,
        tile_idx,
        COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
        NUM_TILES=NUM_TILES,
    )


# Copy sizes: 0/1/7 (all-head), 8 (all-body when dst-aligned), 15 (body+tail,
# tail-mask path), 16 (small tiled body), 1024 and 4 KiB (spans all 16 tiles
# at COPY_BLOCK_SIZE=8, with 4 KiB adding multi-iteration-per-tile coverage;
# both single-tile at COPY_BLOCK_SIZE=1024).
_COPY_SIZES = [0, 1, 7, 8, 15, 16, 1024, 4 * 1024]
# (src_off, dst_off) pairs. Torch tensors are 256B-aligned at data_ptr, so
# slicing by these bytes yields a controlled sub-8B alignment. The kernel
# branches on ``(src ^ dst) & 7``, so we cover both sides plus the aligned
# baseline; the full 5x5 product added no coverage.
_ALIGN_PAIRS = [
    (0, 0),  # fully aligned; head/tail masked out
    (3, 3),  # shared misalignment; fast path with head_bytes=5
    (7, 7),  # shared, extreme; head_bytes=1 with non-empty body
    (1, 3),  # mismatched; byte load/store fallback
    (3, 1),  # mismatched, opposite direction
]
_MAX_ALIGN_OFF = max(o for pair in _ALIGN_PAIRS for o in pair)
# (NUM_TILES, COPY_BLOCK_SIZE) configs. NUM_TILES=1 is the SD conv callsite
# (single-CTA memcpy) where tile partitioning collapses, so COPY_BLOCK_SIZE
# is not observable — one value suffices. NUM_TILES=_TEMPORAL_TILES is the
# temporal callsite: 1024 matches the production launch; 8 shrinks per-tile
# rounding so ``_COPY_SIZES``' 1024/4096 cases cross tile boundaries
# end-to-end (partitioning math is COPY_BLOCK_SIZE-agnostic, so a small
# value proves it for all values).
_TILE_BLOCK_CONFIGS = [
    (1, 1024),
    (_TEMPORAL_TILES, 1024),
    (_TEMPORAL_TILES, 8),
]


@_parametrize("copy_size", _COPY_SIZES)
@_parametrize("src_off,dst_off", _ALIGN_PAIRS)
@_parametrize("num_tiles,copy_block_size", _TILE_BLOCK_CONFIGS)
def test_memcpy_u64_tiled_matches_slice_copy(
    copy_size, src_off, dst_off, num_tiles, copy_block_size
):
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Pad by the max possible offset so the reference slice is in-bounds.
    slack = _MAX_ALIGN_OFF + 8
    src = torch.randint(0, 256, (copy_size + slack,), dtype=torch.uint8, device=device)
    # Start dst from a distinct random pattern so the "unchanged region"
    # check catches any accidental out-of-range writes.
    dst = torch.randint(0, 256, (copy_size + slack,), dtype=torch.uint8, device=device)
    ref = dst.clone()
    ref[dst_off : dst_off + copy_size] = src[src_off : src_off + copy_size]

    _memcpy_wrapper_kernel[(num_tiles,)](
        src,
        dst,
        copy_size,
        src_off,
        dst_off,
        COPY_BLOCK_SIZE=copy_block_size,
        NUM_TILES=num_tiles,
    )
    torch.accelerator.synchronize()

    # Byte-identical: the memcpy is dtype-agnostic; any diff is a bug.
    torch.testing.assert_close(dst, ref, rtol=0, atol=0)


if __name__ == "__main__":
    for cs in _COPY_SIZES:
        for so, do in _ALIGN_PAIRS:
            for nt, cbs in _TILE_BLOCK_CONFIGS:
                test_memcpy_u64_tiled_matches_slice_copy(cs, so, do, nt, cbs)
    print("ok")
