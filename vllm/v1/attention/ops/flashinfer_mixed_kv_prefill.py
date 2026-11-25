# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded NVFP4 V staging for the opt-in mixed FlashInfer prefill policy."""

from collections.abc import Callable

import torch

from vllm.triton_utils import tl, triton

_MAX_STAGING_BYTES = 64 << 20


@triton.jit
def _unpack_v(
    V,
    SF,
    TABLE,
    LENS,
    OUT,
    MAP,
    vs: tl.constexpr,
    ss: tl.constexpr,
    table_stride: tl.constexpr,
    W: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    X: tl.constexpr,
):
    ref, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row, col = ref // W, ref % W
    live = tl.load(LENS + row) - col * P
    page = tl.load(TABLE + row * table_stride + col, live > 0, 0).to(tl.int64)
    if head == 0 and tile == 0:
        tl.store(MAP + 2 * row * W + col, page)
        tl.store(MAP + (2 * row + 1) * W + col, tl.where(live > 0, ref + 1, 0))
    if live <= 0:
        return
    x = tile * X + tl.arange(0, X)
    t, quartet = x // (D // 4), x % (D // 4)
    group = quartet // 4
    st = t // 4 * 4 + group // (D // 64)
    sg = group % (D // 64) * 4 + t % 4
    valid = (t < P) & (t < live)
    a = tl.load(V + page * vs[0] + head * vs[1] + t * vs[2] + quartet, valid, 0)
    sf = tl.load(SF + page * ss[0] + head * ss[1] + st * ss[2] + sg, valid, 0)
    v = tl.inline_asm_elementwise(
        "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 $0, $1, $2;",
        "=r,h,r",
        [a, sf.to(tl.uint32) * 0x01010101],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    # PV MMA can read a whole physical page. Write zero in unwritten tail rows.
    tl.store(OUT + ((ref.to(tl.int64) + 1) * H + head) * (P * D // 4) + x, v, t < P)


def maybe_staged_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sf: torch.Tensor,
    table: torch.Tensor,
    lens: torch.Tensor,
    ptr: torch.Tensor,
    out: torch.Tensor,
    *,
    max_q_len: int,
    num_decodes: int,
    num_decode_tokens: int,
    qk_scale: float,
    v_scale: float,
    get_workspace: Callable[[], torch.Tensor],
) -> bool:
    """Return False without writing output for unsupported layouts or large batches."""
    nd, nt = num_decodes, num_decode_tokens
    b, w = table.shape[0] - nd, table.shape[1]
    _, h, p, d = k.shape
    if (
        not b
        or max_q_len <= 1
        or p != 64
        or d not in (128, 256)
        or k.stride(2) != d
        or v.stride(2) != d // 2
        or sf.stride(2) != d // 16
        or (1 + b * w) * h * p * d + 8 * b * w > _MAX_STAGING_BYTES
    ):
        return False
    from flashinfer.attention.mixed_kv import paged_fp8_k_nvfp4_v_fmha
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache

    if nt:
        paged_fp8_k_nvfp4_v_fmha(
            q[:nt],
            k,
            v,
            sf,
            table[:nd],
            lens[:nd],
            ptr[: nd + 1],
            max_q_len=1,
            out=out[:nt],
            qk_scale=qk_scale,
            v_scale=v_scale,
        )
    # Rebase inside the captured call. No host read of lengths or device offsets.
    qptr = ptr[nd:] - nt if nd else ptr
    tab, lens = table[nd:], lens[nd:]
    kvptr = torch.empty(b + 1, device=q.device, dtype=torch.int32)
    kvptr[:1].zero_()
    torch.cumsum(lens, dim=0, dtype=torch.int32, out=kvptr[1:])
    expanded = torch.empty((1 + b * w, h, p, d), device=q.device, dtype=q.dtype)
    expanded[0].zero_()
    maps = torch.empty((b, 2, w), device=q.device, dtype=torch.int32)
    v16 = v.view(torch.uint16)
    _unpack_v[(b * w, h, triton.cdiv(p * d // 4, 256))](
        v16,
        sf.view(torch.uint8),
        tab,
        lens,
        expanded.view(torch.uint32),
        maps,
        v16.stride(),
        sf.stride(),
        tab.stride(0),
        w,
        h,
        p,
        d,
        256,
        ptx_version=94,
    )
    trtllm_batch_context_with_kv_cache(
        query=q[nt:],
        kv_cache=(k, expanded),
        workspace_buffer=get_workspace(),
        block_tables=maps,
        seq_lens=lens,
        out=out[nt:],
        out_dtype=out.dtype,
        kv_layout="HND",
        bmm1_scale=qk_scale,
        bmm2_scale=v_scale,
        max_q_len=max_q_len,
        max_kv_len=p * w,
        batch_size=b,
        cum_seq_lens_q=qptr,
        cum_seq_lens_kv=kvptr,
        causal=True,
        uses_shared_paged_kv_idx=False,
    )
    return True
