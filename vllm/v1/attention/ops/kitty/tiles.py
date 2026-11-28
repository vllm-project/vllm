# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kitty page layout and the Triton tile loaders shared by the read kernels.

A packed page holds one 128-token block of one KV head. Segment byte offsets
(D = head size, DB = boosted K channels):

    K low plane   0             32*D    uint8 [D, 32], 2-bit codes along tokens
    V data        32*D          32*D    uint8 [128, D/4], 2-bit codes along D
    K high plane  64*D          32*DB   uint8 [DB, 32], row r = rank-r channel
    K scale/min   64*D + 32*DB  4*D     fp16 [D, 2]
    V scale/min   + 4*D         512     fp16 [128, 2]
    K rank table  + 512         D       uint8 [D], DB means not boosted

The residual block of a (request, KV head) is a [640, D] array: K sink rows
0-127, K tail buffer 128-255 (row 128 + p % 128), V sink 256-383, V tail buffer
384-639 (row 384 + p % 256).
"""

import numpy as np

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

KITTY_PAGE_SIZE = 128
KITTY_RESIDUAL_ROWS = 640
# Residual rows hold one K and one V row per slot, so 320 slots give 640 rows.
KITTY_RESIDUAL_SLOTS = KITTY_RESIDUAL_ROWS // 2
K_SINK_ROW = 0
K_TAIL_ROW = 128
V_SINK_ROW = 256
V_TAIL_ROW = 384


def kitty_d_boost(head_size: int, boost_ratio: float) -> int:
    """DB: number of K channels per page boosted from 2 to 4 bits.

    DB = floor(D * boost_ratio), with 0 <= boost_ratio <= 1, so 0 <= DB <= D.
    For example, D = 128 and ratio 0.25 give DB = 32. The DB channels with the
    largest mean |K| in a page are boosted (see `kitty_boost_ranks`).
    """
    return int(head_size * boost_ratio)


def kitty_page_bytes(head_size: int, d_boost: int) -> int:
    """Bytes of one packed page per (block, KV head).

    A page holds T = 128 tokens. Summing the segments in the module docstring:

        K low plane   D * T * 2 bits / 8   = 32 * D
        V data        T * D * 2 bits / 8   = 32 * D
        K high plane  DB * T * 2 bits / 8  = 32 * DB
        K scale/min   D * 2 * 2 bytes      = 4 * D
        V scale/min   T * 2 * 2 bytes      = 512
        K rank table  D * 1 byte           = D

    Total = (32 + 32 + 4 + 1) * D + 32 * DB + 512 = 69 * D + 32 * DB + 512.
    A boosted channel's 4-bit code is its 2 low bits plus 2 high bits from the
    high plane, so boosting costs 32 extra bytes per boosted channel.
    """
    return 69 * head_size + 32 * d_boost + 512


def kitty_state_content_bytes(head_size: int, d_boost: int) -> int:
    """C: bytes per token slot of a packed block, so one page is 128 * C.

    C = ceil(page_bytes / 128) = ceil((69 * D + 32 * DB + 512) / 128). The
    rounding pads the page to a whole number of token slots, so 128 * C >=
    page_bytes. For D = 128 and DB = 32 this is exact: 10368 / 128 = 81.
    """
    return cdiv(kitty_page_bytes(head_size, d_boost), KITTY_PAGE_SIZE)


def kitty_first_unsealed_k(n):
    """S_k(n): first K block not sealed at cache length n (int or ndarray).

    Block b holds positions [128 * b, 128 * (b + 1)). Block 0 is the sink and
    is never sealed. Any other block b is sealed once it is full, which happens
    when n >= 128 * (b + 1). The sealed blocks are 1, ..., floor(n / 128) - 1, so

        S_k(n) = max(floor(n / 128), 1).

    Positions p >= 128 * S_k(n) stay in the K tail, which holds
    n - 128 * S_k(n) = n mod 128 <= 127 tokens once n >= 128. A step from
    length n_old to n_new seals the K blocks [S_k(n_old), S_k(n_new)).
    """
    return np.maximum(n // KITTY_PAGE_SIZE, 1)


def kitty_first_unsealed_v(n):
    """S_v(n): first V block not sealed at cache length n (int or ndarray).

    Like `kitty_first_unsealed_k`, except that V seals one page later. A
    block b >= 1 is sealed only when n >= 128 * (b + 2), so the most recent full
    block stays unquantized. The sealed blocks are 1, ..., floor(n / 128) - 2,
    so

        S_v(n) = max(floor(n / 128) - 1, 1).

    Once n >= 256, the V tail holds n - 128 * S_v(n) = 128 + n mod 128 tokens,
    which is between 128 and 255. A step from length n_old to n_new seals the
    V blocks [S_v(n_old), S_v(n_new)).
    """
    return np.maximum(n // KITTY_PAGE_SIZE - 1, 1)


@triton.jit
def kitty_dequant_k_page(page_ptr, offs_t, offs_d, D: tl.constexpr, DB: tl.constexpr):
    """Reconstruct a packed K page as an fp32 `[D, 128]` tile.

    Read 2-bit codes for all channels and 2 extra bits for boosted channels,
    then reconstruct values with each channel's `code * scale + minimum`.
    """
    off_khi: tl.constexpr = 64 * D
    off_ksm: tl.constexpr = 64 * D + 32 * DB
    off_rank: tl.constexpr = off_ksm + 4 * D + 512
    byte = offs_t // 4
    shift = (offs_t % 4) * 2
    low = tl.load(page_ptr + offs_d[:, None] * 32 + byte[None, :]).to(tl.int32)
    low = (low >> shift[None, :]) & 3
    rank = tl.load(page_ptr + off_rank + offs_d).to(tl.int32)
    boosted = rank < DB
    high = tl.load(
        page_ptr + off_khi + rank[:, None] * 32 + byte[None, :],
        mask=boosted[:, None],
        other=0,
    ).to(tl.int32)
    high = (high >> shift[None, :]) & 3
    code = low | (high << 2)
    sm_ptr = (page_ptr + off_ksm).to(tl.pointer_type(tl.float16))
    scale = tl.load(sm_ptr + 2 * offs_d).to(tl.float32)
    mn = tl.load(sm_ptr + 2 * offs_d + 1).to(tl.float32)
    return code.to(tl.float32) * scale[:, None] + mn[:, None]


@triton.jit
def kitty_dequant_v_page(page_ptr, offs_t, offs_d, D: tl.constexpr, DB: tl.constexpr):
    """Reconstruct a packed V page as an fp32 `[128, D]` tile.

    Read 2-bit codes packed along channels, then reconstruct values with
    each token's `code * scale + minimum`.
    """
    off_v: tl.constexpr = 32 * D
    off_vsm: tl.constexpr = 68 * D + 32 * DB
    byte = offs_d // 4
    shift = (offs_d % 4) * 2
    data = tl.load(page_ptr + off_v + offs_t[:, None] * (D // 4) + byte[None, :])
    code = (data.to(tl.int32) >> shift[None, :]) & 3
    sm_ptr = (page_ptr + off_vsm).to(tl.pointer_type(tl.float16))
    scale = tl.load(sm_ptr + 2 * offs_t).to(tl.float32)
    mn = tl.load(sm_ptr + 2 * offs_t + 1).to(tl.float32)
    return code.to(tl.float32) * scale[:, None] + mn[:, None]


@triton.jit
def kitty_load_k_tile(
    page_ptr,
    res_ptr,
    b,
    old_len,
    s_k,
    offs_t,
    offs_d,
    D: tl.constexpr,
    DB: tl.constexpr,
):
    """K of position block b as a `[D, 128]` tile in the residual's dtype.

    `page_ptr` is packed block `block_table[r, b]` of this KV head and
    `res_ptr` the request's `[640, D]` residual region.

    Blocks `0 < b < s_k` are dequantized from `page_ptr`. K seals each non-sink
    block as soon as it is full. Block 0 uses residual sink rows 0-127; the
    unfinished block uses tail rows 128-255. Positions >= `old_len` load as 0.
    """
    if (b > 0) & (b < s_k):
        k = kitty_dequant_k_page(page_ptr, offs_t, offs_d, D, DB)
        k = k.to(res_ptr.dtype.element_ty)
    else:
        row0 = tl.where(b == 0, 0, 128)
        valid = b * 128 + offs_t < old_len
        k = tl.load(
            res_ptr + (row0 + offs_t)[None, :] * D + offs_d[:, None],
            mask=valid[None, :],
            other=0.0,
        )
    return k


@triton.jit
def kitty_load_v_tile(
    page_ptr,
    res_ptr,
    b,
    old_len,
    s_v,
    offs_t,
    offs_d,
    D: tl.constexpr,
    DB: tl.constexpr,
):
    """V of position block b as a `[128, D]` tile in the residual's dtype.

    Blocks `0 < b < s_v` are dequantized from `page_ptr`. V seals one block
    later than K, keeping the most recent full block unquantized. Block 0
    uses residual sink rows 256-383; unsealed blocks use the two-block tail
    at rows 384-639, selected by `b % 2`. Positions >= `old_len` load as 0.
    """
    if (b > 0) & (b < s_v):
        v = kitty_dequant_v_page(page_ptr, offs_t, offs_d, D, DB)
        v = v.to(res_ptr.dtype.element_ty)
    else:
        row0 = tl.where(b == 0, 256, 384 + (b % 2) * 128)
        valid = b * 128 + offs_t < old_len
        v = tl.load(
            res_ptr + (row0 + offs_t)[:, None] * D + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
    return v
