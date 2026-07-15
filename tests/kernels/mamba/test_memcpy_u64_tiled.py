# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for ``_memcpy_u64_tiled``.

Guards the head/body/tail alignment handling and the u64 body's tile
partitioning used by ``postprocess_mamba_fused_kernel`` and
``precopy_mamba_align_fused_kernel``. The device function must be
byte-identical to a ``dst[dst_off:dst_off+copy_size] = src[src_off:src_off+
copy_size]`` slice copy for every combination of:

* ``copy_size`` including the degenerate 0-byte case, sub-8B (head-only),
  8B-aligned bodies, and large multi-MiB copies matching real temporal
  states.
* Sub-8B alignment offsets on ``src`` and ``dst`` (independent), covering
  head=0/1/3/5/7 bytes and every src/dst sub-alignment mismatch.
* ``NUM_TILES`` values used at the callsites (1 for the SD conv path;
  ``_TEMPORAL_TILES`` for the temporal path) and a few extras that stress
  the tile boundary rounding.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.mamba_utils import _memcpy_u64_tiled

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


# Copy sizes: 0/1/7 (all-head), 8 (all-body when dst-aligned), 15 (head+body
# or body+tail), 16/17 (small tiled body), 1024 (one COPY_BLOCK_SIZE),
# 80 KiB (Nemotron conv per-block), 4 MiB (Nemotron temporal per-block).
_COPY_SIZES = [0, 1, 7, 8, 15, 16, 17, 1024, 80 * 1024, 4 * 1024 * 1024]
# Sub-8B alignment offsets. torch tensors are 256B-aligned at data_ptr, so
# slicing by these values yields a controlled sub-8B alignment.
_ALIGN_OFFS = [0, 1, 3, 5, 7]
# NUM_TILES: 1 matches the SD conv callsite, 8 matches _TEMPORAL_TILES, 16
# stresses tile-boundary rounding when body_u64 is small.
_NUM_TILES = [1, 2, 4, 8, 16]


@_parametrize("copy_size", _COPY_SIZES)
@_parametrize("dst_off", _ALIGN_OFFS)
@_parametrize("src_off", _ALIGN_OFFS)
@_parametrize("num_tiles", _NUM_TILES)
def test_memcpy_u64_tiled_matches_slice_copy(copy_size, dst_off, src_off, num_tiles):
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Pad by the max possible offset so the reference slice is in-bounds.
    slack = max(_ALIGN_OFFS) + 8
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
        COPY_BLOCK_SIZE=1024,
        NUM_TILES=num_tiles,
    )
    torch.accelerator.synchronize()

    # Byte-identical: the memcpy is dtype-agnostic; any diff is a bug.
    torch.testing.assert_close(dst, ref, rtol=0, atol=0)


if __name__ == "__main__":
    for cs in _COPY_SIZES:
        for do in _ALIGN_OFFS:
            for so in _ALIGN_OFFS:
                for nt in _NUM_TILES:
                    test_memcpy_u64_tiled_matches_slice_copy(cs, do, so, nt)
    print("ok")
