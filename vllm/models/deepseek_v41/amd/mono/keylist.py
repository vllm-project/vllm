# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2's key lists without K1: the KT and KLEN that K1's ``stage_kt`` writes.

On an index layer (2, 8, 14, 20, 24, 28, 32 and 36 of DeepSeek-V4.1-Flash)
vLLM runs the attention front: the projections, the KV insert, the compressor
and the indexer, which writes the layer's top-512. K2 can then run the rest of
the layer on vLLM's q, but it reads each token's keys from KT and KLEN, which
only K1 wrote so far. ``launch`` writes them with one Triton program a token,
from the same inputs ``stage_kt`` reads, in the same format:

    KT[t * KEYS + k] = the compressed slot | 1 << 31 of top-k entry k, for
                       k < ntopk (-1 where the entry is -1), then the window
                       slots, then -1 up to KEYS
    KLEN[t] = ntopk + the window length, KLEN[S + t] = ntopk

ntopk is min((position + 1) // ratio, TOPK), as in ``stage_kt``, and a token
whose slot is -1 has no keys. tests/kernels/test_dsv41_mono_gfx942.py
compares the result with a reference built in torch.
"""

import torch

from vllm.triton_utils import tl, triton

from .attention.plan import KEYS, TOPK, WINDOW


@triton.jit
def _keylist_kernel(
    slot_ptr,
    pos_ptr,
    swa_lens_ptr,
    swa_idx_ptr,
    topk_ptr,
    t2r_ptr,
    bt_ptr,
    bt_stride,
    comp_block,
    kt_ptr,
    klen_ptr,
    S,
    RATIO: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    KEYS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    live = tl.load(slot_ptr + t) >= 0
    nswa = tl.where(live, tl.load(swa_lens_ptr + t), 0)
    if RATIO > 0:
        pos = tl.load(pos_ptr + t).to(tl.int32)
        ntopk = tl.where(live, tl.minimum((pos + 1) // RATIO, TOPK), 0)
    else:
        ntopk = 0
    kv_len = ntopk + nswa
    # The window part: entry k - ntopk of the token's dense window row.
    sidx = tl.minimum(tl.maximum(k - ntopk, 0), WINDOW - 1)
    s_slot = tl.load(swa_idx_ptr + t * WINDOW + sidx)
    val = tl.where(k < kv_len, s_slot, -1)
    if RATIO > 0:
        # The top-k part: a local compressed row through the request's block
        # table, with bit 31 set so that K2 reads it from the compressed
        # cache. The loads are masked where stage_kt discards the value.
        head = k < ntopk
        local = tl.load(topk_ptr + t * TOPK + k, mask=head, other=-1)
        lc = tl.maximum(local, 0)
        req = tl.load(t2r_ptr + t)
        blk = tl.load(bt_ptr + req * bt_stride + lc // comp_block, mask=head, other=0)
        c_slot = (blk * comp_block + lc % comp_block) | -2147483648
        val = tl.where(head, tl.where(local >= 0, c_slot, -1), val)
    tl.store(kt_ptr + t * KEYS + k, val, mask=k < KEYS)
    tl.store(klen_ptr + t, kv_len)
    tl.store(klen_ptr + S + t, ntopk)


def launch(
    kt: torch.Tensor,  # [>= S * KEYS] int32
    klen: torch.Tensor,  # [>= 2 S] int32
    slot_mapping: torch.Tensor,  # [S] int64
    positions: torch.Tensor,  # [S] int64
    swa_indices: torch.Tensor,  # [>= S, 1, WINDOW] int32, dense window rows
    swa_lens: torch.Tensor,  # [>= S] int32
    token_to_req: torch.Tensor,  # [>= S] int32
    ratio: int,
    topk: torch.Tensor | None = None,  # [>= S, TOPK] int32, local rows
    comp_block_table: torch.Tensor | None = None,  # [reqs, blocks] int32
    comp_block: int = 1,  # the compressed cache's rows a block
) -> None:
    S = slot_mapping.shape[0]
    assert swa_indices.stride(0) == WINDOW, swa_indices.stride()
    if ratio:
        assert topk is not None and comp_block_table is not None
        assert topk.stride(0) == TOPK, topk.stride()
        bt, bt_stride = comp_block_table, comp_block_table.stride(0)
    else:
        topk, bt, bt_stride = swa_lens, swa_lens, 0
    _keylist_kernel[(S,)](
        slot_mapping,
        positions,
        swa_lens,
        swa_indices,
        topk,
        token_to_req,
        bt,
        bt_stride,
        comp_block,
        kt,
        klen,
        S,
        RATIO=ratio,
        WINDOW=WINDOW,
        TOPK=TOPK,
        KEYS=KEYS,
        BLOCK=triton.next_power_of_2(KEYS),
        num_warps=4,
    )
