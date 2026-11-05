# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page geometry of the MiniMax M3 MXFP4 K/V and index-K caches on ROCm.

One 128-token page of one head holds, for K (or V, or the index key), the
128-wide rows as e2m1 data (two elements a byte, the low nibble the even one)
followed by one e8m0 scale per 32 elements::

    [128 tokens x 64 B data][128 tokens x 4 B scale]  = 8704 B = 128 x 68 B

There is no global scale. This is the page AITER's ``fused_qknorm_idxrqknorm``
writes and its FlyDSL scorers and ``pa_decode_mxfp4`` read.
"""

import torch

PAGE_TOKENS = 128
HEAD_DIM = 128
DATA_BYTES = HEAD_DIM // 2
SCALE_BYTES = HEAD_DIM // 32
# Bytes per token and per page of one head.
TOKEN_BYTES = DATA_BYTES + SCALE_BYTES
PAGE_BYTES = PAGE_TOKENS * TOKEN_BYTES


def mxfp4_page_views(pages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(data, scale) views of uint8 pages ``[..., PAGE_BYTES]``.

    The views are ``[..., 128, 64]`` and ``[..., 128, 4]`` and keep the leading
    dims and their strides, so ``[pages, heads, 8704]`` becomes the
    ``[pages, heads, 128, 64]`` / ``[pages, heads, 128, 4]`` pair
    ``pa_decode_mxfp4`` takes and ``[pages, 8704]`` the ``[pages, 128, 64]`` /
    ``[pages, 128, 4]`` pair the index scorers take.
    """
    if pages.dtype != torch.uint8 or pages.shape[-1] != PAGE_BYTES:
        raise ValueError(
            f"MXFP4 pages are uint8 [..., {PAGE_BYTES}], got {pages.dtype} "
            f"{tuple(pages.shape)}"
        )
    if pages.stride(-1) != 1:
        raise ValueError("MXFP4 pages need a contiguous byte axis")
    lead, lead_strides = tuple(pages.shape[:-1]), tuple(pages.stride()[:-1])
    offset = pages.storage_offset()
    data = pages.as_strided(
        (*lead, PAGE_TOKENS, DATA_BYTES), (*lead_strides, DATA_BYTES, 1), offset
    )
    scale = pages.as_strided(
        (*lead, PAGE_TOKENS, SCALE_BYTES),
        (*lead_strides, SCALE_BYTES, 1),
        offset + PAGE_TOKENS * DATA_BYTES,
    )
    return data, scale
