# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoders for CSF-compressed NVFP4 block scales.

ModelOpt checkpoints declare ``weight_scale_encoding`` per layer and store each
E4M3 block-scale matrix as several ``<name>.weight_scale.<stream>`` tensors.
Every decoder rebuilds the exact scale bytes of the uncompressed checkpoint.

- ``csf``: ``nvfp4_csf_fixed`` (uint8, ``[rows // 16, 16 * (1 + cols // 2)]``)
  holds, per 16-row slab, one base byte per row followed by the rows' 4-bit
  codes, two per byte, low nibble first; a scale byte is ``base + code``.
  ``nvfp4_csf_exceptions`` (uint32) holds sorted words whose low 24 bits are a
  flat ``row * cols + col`` position and whose high 8 bits replace that byte.
- ``csf_32x2_v1``: ``csf32_bases`` (uint8, one per row) and ``csf32_codes``
  (uint8, 4-bit codes) cover 32-row x 2-column blocks, grouped into 128 x 4
  tiles, 32 tiles to a page. ``csf32_exceptions`` (uint16) holds the in-block
  position in the low byte and the scale byte in the high byte;
  ``csf32_page_bases`` (uint32, byte offset of each page's first exception)
  and ``csf32_offsets`` (uint16, per page a 0 then each block's exception end)
  index them.
"""

import functools
from collections.abc import Callable

import numpy as np
import torch

# numpy keeps these small ops off torch's intra-op thread pool, whose per-op
# sync dominates when weight loading runs with many threads.


def _decode_csf(fixed: torch.Tensor, exceptions: torch.Tensor) -> np.ndarray:
    if fixed.dtype != torch.uint8 or fixed.dim() != 2 or fixed.shape[1] % 16:
        raise ValueError(f"Invalid CSF fixed stream {fixed.dtype} {fixed.shape}")
    if exceptions.dtype != torch.uint32:
        raise ValueError(f"Invalid CSF exceptions dtype {exceptions.dtype}")
    rows = fixed.shape[0] * 16
    cols = (fixed.shape[1] // 16 - 1) * 2
    if rows * cols > 1 << 24:
        raise ValueError("CSF scale exceeds 24-bit exception positions")
    raw = fixed.numpy()
    bases = raw[:, :16].reshape(rows, 1)
    if (bases > 240).any():
        raise ValueError("CSF row bases must be in 0..240")
    packed = raw[:, 16:].reshape(rows, cols // 2)
    codes = np.stack((packed & 0xF, packed >> 4), axis=-1).reshape(rows, cols)
    out = (bases + codes).reshape(-1)
    words = exceptions.reshape(-1).numpy()
    positions = words & 0xFFFFFF
    if positions.size and (
        positions[-1] >= rows * cols or (positions[1:] <= positions[:-1]).any()
    ):
        raise ValueError("CSF exception positions must be increasing and in range")
    out[positions] = words >> 24
    return out.reshape(rows, cols)


@functools.cache
def _csf32_blocks(rows: int, cols: int):
    """Origin, width and page of each 32x2 block in storage order, and the flat
    ``[rows, cols]`` position of each code."""
    tile_r, tile_c, pair, block = np.meshgrid(
        np.arange(-(-rows // 128)),
        np.arange(-(-cols // 4)),
        np.arange(2),
        np.arange(4),
        indexing="ij",
    )
    r0 = (128 * tile_r + 32 * block).ravel()
    c0 = (4 * tile_c + 2 * pair).ravel()
    page = (tile_r * tile_c.shape[1] + tile_c).ravel() // 32
    keep = (r0 < rows) & (c0 < cols)
    r0, c0, page = r0[keep], c0[keep], page[keep]
    width = np.minimum(cols - c0, 2)
    i, j = np.meshgrid(np.arange(32), np.arange(2), indexing="ij")
    cells = (r0[:, None, None] + i) * cols + c0[:, None, None] + j
    return r0, c0, width, page, cells[j < width[:, None, None]]


def _decode_csf_32x2(
    bases: torch.Tensor,
    codes: torch.Tensor,
    page_bases: torch.Tensor,
    offsets: torch.Tensor,
    exceptions: torch.Tensor,
) -> np.ndarray:
    dtypes = (torch.uint8, torch.uint8, torch.uint32, torch.uint16, torch.uint16)
    streams = (bases, codes, page_bases, offsets, exceptions)
    rows = bases.numel()
    cols = 2 * codes.numel() // max(rows, 1)
    if (
        any(t.dtype != d for t, d in zip(streams, dtypes))
        or rows % 32
        or rows * cols != 2 * codes.numel()
        or cols == 0
    ):
        raise ValueError("Invalid csf_32x2_v1 streams")
    r0, c0, width, page, cells = _csf32_blocks(rows, cols)
    num_pages = int(page[-1]) + 1
    base, packed, origins, index, words = (t.reshape(-1).numpy() for t in streams)
    if origins.size != num_pages or index.size != r0.size + num_pages:
        raise ValueError("Invalid csf_32x2_v1 index sizes")
    if (base > 240).any():
        raise ValueError("CSF row bases must be in 0..240")
    out = np.empty(rows * cols, dtype=np.uint8)
    out[cells] = np.stack((packed & 0xF, packed >> 4), -1).ravel() + base[cells // cols]

    first = np.flatnonzero(np.diff(page, prepend=-1))
    ends = origins[page].astype(np.int64) // 2 + index[np.arange(r0.size) + page + 1]
    starts = np.concatenate(([0], ends[:-1]))
    if (
        (index[first + np.arange(num_pages)] != 0).any()
        or (origins != 2 * starts[first]).any()
        or (ends < starts).any()
        or (ends - starts > 32 * width).any()
        or ends[-1] != words.size
    ):
        raise ValueError("Invalid csf_32x2_v1 exception index")
    block = np.repeat(np.arange(r0.size), ends - starts)
    pos = (words & 0xFF).astype(np.int64)
    if (
        (pos >= 64).any()
        or (pos % 2 >= width[block]).any()
        or ((pos[1:] <= pos[:-1]) & (block[1:] == block[:-1])).any()
    ):
        raise ValueError("CSF exception positions must be increasing and in range")
    out[(r0[block] + pos // 2) * cols + c0[block] + pos % 2] = words >> 8
    return out.reshape(rows, cols)


# weight_scale_encoding -> (stream suffixes in decoder argument order, decoder)
CSF_SCALE_ENCODINGS: dict[str, tuple[tuple[str, ...], Callable[..., np.ndarray]]] = {
    "csf": (("nvfp4_csf_fixed", "nvfp4_csf_exceptions"), _decode_csf),
    "csf_32x2_v1": (
        tuple(
            f"csf32_{name}"
            for name in ("bases", "codes", "page_bases", "offsets", "exceptions")
        ),
        _decode_csf_32x2,
    ),
}


def decode_csf_scale(encoding: str, streams: dict[str, torch.Tensor]) -> torch.Tensor:
    """Rebuild one ``[rows, cols]`` float8_e4m3fn block-scale matrix."""
    names, decode = CSF_SCALE_ENCODINGS[encoding]
    out = decode(*(streams[name].cpu() for name in names))
    return torch.from_numpy(out).view(torch.float8_e4m3fn)
