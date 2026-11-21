# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from SGLang's compact A4 MoE kernels
# (python/sglang/kernels/ops/moe/mxfp4_compact_a4_moe.py, Apache-2.0).
"""Compact MXFP4 experts for small Muse Spark decode calls (ROCm gfx950).

For the `[T, top_k]` routes of T MXFP4-quantized rows over this rank's local
experts, in three launches:

1. `_route_quant_kernel`: per local expert, its routes in token order, placed in
   `BM`-row blocks after the lower experts' blocks (unused blocks get expert -1),
   and the per-row MXFP4 quant of the T rows (the rounding of `sort_quant`);
2. `_expert_kernel` (gate/up): `dot_scaled` gate and up GEMMs, SiLU-and-mul and
   the MXFP4 quant of each route's product (the bf16 rounding and the quant of
   the bf16-output FlyDSL stage 1 followed by the stage-2 input quant);
3. `_expert_kernel` (down): one unweighted BF16 row per local (token, slot).

Every program reads one N tile of its block's expert weights once, streamed
past the caches. Weights and scales are in AITER's preshuffled MXFP4 layout, so
the AITER expert stages and this path share one copy.
"""

import functools
from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton

# AITER's Triton MXFP4 quant (private); without it this path is off.
try:
    from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

    if not {"SCALING_MODE", "USE_ASM"} <= set(
        getattr(_mxfp4_quant_op, "arg_names", ())
    ):
        raise ImportError
except ImportError:
    _mxfp4_quant_op = None

# Largest call (tokens) that takes this path.
MAX_TOKENS = 448
TOP_K = 8


class TileConfig(NamedTuple):
    """Rows per expert block, and per stage: N tile, K tile (elements), warps,
    pipeline stages."""

    block_m: int
    bn1: int
    bk1: int
    warps1: int
    stages1: int
    bn2: int
    bk2: int
    warps2: int
    stages2: int


# By the largest token count each row covers (gfx950, Muse Spark expert
# shapes). With few routes per touched expert, 16-row blocks and narrow N tiles
# spread each expert's weights over more programs; with more routes, 32-row
# blocks keep an expert's weights from being read twice.
_TILE_TABLE: tuple[tuple[int, TileConfig], ...] = (
    (64, TileConfig(16, 32, 1024, 2, 2, 32, 1024, 2, 2)),
    (MAX_TOKENS, TileConfig(32, 64, 512, 4, 2, 64, 512, 4, 2)),
)


def tile_config(num_tokens: int) -> TileConfig:
    for last, cfg in _TILE_TABLE:
        if num_tokens <= last:
            return cfg
    raise ValueError(f"no compact MoE tiles for {num_tokens} tokens")


@functools.cache
def _on_gfx950() -> bool:
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def supports_compact_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> bool:
    """Whether `compact_moe_experts` takes these bf16 rows `x` (`[T, H]`), their
    int32 routes `topk_ids` (`[T, top_k]`) and preshuffled MXFP4 expert weights
    (`w13`: `[E, 2 * I, H / 2]`, `w2`: `[E, H, I / 2]`, E8M0 scales per 32
    columns)."""
    if x.dim() != 2 or topk_ids.dim() != 2 or w13.dim() != 3 or w2.dim() != 3:
        return False
    num_tokens, hidden = x.shape
    e, inter = w13.shape[0], w2.shape[2] * 2
    return (
        _mxfp4_quant_op is not None
        and _on_gfx950()
        and 0 < num_tokens <= MAX_TOKENS
        and topk_ids.shape == (num_tokens, TOP_K)
        and x.dtype == torch.bfloat16
        and topk_ids.dtype == torch.int32
        and x.is_contiguous()
        and topk_ids.is_contiguous()
        and hidden % 1024 == 0
        and inter % 1024 == 0
        and 0 < e <= 128
        and e & (e - 1) == 0
        and w13.shape == (e, 2 * inter, hidden // 2)
        and w2.shape == (e, hidden, inter // 2)
        and w13_scale.shape == (e, 2 * inter, hidden // 32)
        and w2_scale.shape == (e, hidden, inter // 32)
        and getattr(w13, "is_shuffled", False)
        and getattr(w2, "is_shuffled", False)
        and all(
            t.element_size() == 1 and t.is_contiguous()
            for t in (w13, w2, w13_scale, w2_scale)
        )
    )


@triton.jit(do_not_specialize=["M", "NB"])
def _route_quant_kernel(
    ids_ptr,
    lmap_ptr,
    sorted_ptr,
    block_expert_ptr,
    x_ptr,
    q_ptr,
    qs_ptr,
    M,
    NB,
    LE: tl.constexpr,
    S: tl.constexpr,
    PER_E: tl.constexpr,
    BM: tl.constexpr,
    NBB: tl.constexpr,
    TOPK: tl.constexpr,
    D: tl.constexpr,
    DC: tl.constexpr,
):
    """Programs `[0, LE)`: local expert `pid`'s routes; the rest: one `DC`-column
    chunk of one row's MXFP4 quant."""
    pid = tl.program_id(0)
    if pid < LE:
        e = pid
        le = tl.arange(0, LE)
        s = tl.arange(0, S)
        valid = s < M * TOPK
        gid = tl.load(ids_ptr + s, mask=valid, other=-1)
        lid = tl.load(lmap_ptr + gid, mask=valid & (gid >= 0), other=-1)
        # Other ranks' routes and the padding go to bins [LE, 2 LE) (a masked
        # tl.histogram miscounts on ROCm Triton).
        h2 = tl.reshape(tl.histogram(tl.where(lid >= 0, lid, LE), 2 * LE), [2, LE])
        counts = tl.sum(tl.where(tl.arange(0, 2)[:, None] == 0, h2, 0), axis=0)
        blocks_all = (counts + BM - 1) // BM
        first = tl.sum(tl.where(le < e, blocks_all, 0), axis=0)
        if e == LE - 1:
            b = tl.arange(0, NBB)
            tl.store(
                block_expert_ptr + b,
                -1,
                mask=(b >= tl.sum(blocks_all, axis=0)) & (b < NB),
            )
        hit = lid == e
        count = tl.sum(hit.to(tl.int32), axis=0)
        blocks = (count + BM - 1) // BM
        if blocks > 0:
            eb = tl.arange(0, PER_E)
            tl.store(block_expert_ptr + first + eb, e, mask=eb < blocks)
            rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
            base = first * BM
            c = tl.arange(0, PER_E * BM)
            # Padding and routes write disjoint slots of the expert's blocks.
            tl.store(
                sorted_ptr + base + c, M * TOPK, mask=(c < blocks * BM) & (c >= count)
            )
            tl.store(sorted_ptr + base + rank, s, mask=hit)
    else:
        chunks: tl.constexpr = D // DC
        row = (pid - LE) // chunks
        chunk = (pid - LE) % chunks
        col = chunk * DC + tl.arange(0, DC)
        x = tl.load(x_ptr + row * D + col).to(tl.float32)
        q, qs = _quant_rows(tl.reshape(x, [1, DC]), 1, DC)
        tl.store(
            q_ptr + row * (D // 2) + chunk * (DC // 2) + tl.arange(0, DC // 2),
            tl.reshape(q, [DC // 2]),
        )
        tl.store(
            qs_ptr + row * (D // 32) + chunk * (DC // 32) + tl.arange(0, DC // 32),
            tl.reshape(qs, [DC // 32]),
        )


@triton.jit
def _load_w(W, n0, k0, KROW: tl.constexpr, BN: tl.constexpr, KB: tl.constexpr):
    """`[KB bytes, BN rows]` tile from row `n0`, byte `k0` of one expert's
    preshuffled `[N, KROW]` packed E2M1 rows: `[N / 16, KROW / 16, 16 rows,
    16 bytes]`, loaded as whole 16-row groups and permuted in registers."""
    g = tl.arange(0, BN // 16)
    kc = tl.arange(0, KB // 16)
    r = tl.arange(0, 16)
    b = tl.arange(0, 16)
    offset = (
        ((n0 // 16 + g) * (16 * KROW))[:, None, None, None]
        + ((k0 // 16 + kc) * 256)[None, :, None, None]
        + (r * 16)[None, None, :, None]
        + b[None, None, None, :]
    )
    x = tl.permute(tl.load(W + offset, cache_modifier=".cs"), (1, 3, 0, 2))
    return tl.reshape(x, (KB, BN))


@triton.jit
def _s_offset(n, c, SC: tl.constexpr):
    """`[N, scale col]` offsets in one expert's preshuffled `[N, SC]` E8M0
    scales: `[N / 32, SC / 8, 4 (c % 4), 16 (n % 16), 2 ((c % 8) // 4),
    2 ((n % 32) // 16)]`."""
    n2, c2 = n[:, None], c[None, :]
    return (
        ((((n2 // 32) * (SC // 8) + c2 // 8) * 4 + c2 % 4) * 16 + n2 % 16) * 4
        + ((c2 % 8) // 4) * 2
        + (n2 % 32) // 16
    )


@triton.jit
def _quant_rows(x, BM: tl.constexpr, BN: tl.constexpr):
    """MXFP4 quant of `[BM, BN]` fp32 rows per 32 columns, as sort_quant: E8M0
    scale 2^ceil(log2(amax / 6)) (all-zero blocks 2^-126), round-half-even
    E2M1 codes packed two per byte, the even column in the low nibble."""
    return _mxfp4_quant_op(x, BN, BM, 32, SCALING_MODE=1, USE_ASM=True)


@triton.jit
def _silu_mul_quant(gate, up, BM: tl.constexpr, BN: tl.constexpr):
    """SiLU(gate) * up, rounded to bf16 as a bf16 stage-1 output is, and its
    MXFP4 quant per 32 columns (`_quant_rows`, the stage-2 input quant of the
    bf16-output expert stages)."""
    # 1 / (1 + 2^(-gate * log2(e))) with the hardware exp2 and reciprocal, one
    # instruction per asm block so the compiler resolves their hazards.
    exp = tl.inline_asm_elementwise(
        "v_exp_f32 $0, $1",
        "=v,v",
        args=[gate * -1.4426950408889634],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    sig = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        "=v,v",
        args=[exp + 1.0],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    act = (gate * sig * up).to(tl.bfloat16).to(tl.float32)
    return _quant_rows(act, BM, BN)


@triton.jit(do_not_specialize=["rows"])
def _expert_kernel(
    a_ptr,
    as_ptr,
    w_ptr,
    ws_ptr,
    sorted_ptr,
    block_expert_ptr,
    out_ptr,
    os_ptr,
    rows,
    K: tl.constexpr,
    N: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    GATE_UP: tl.constexpr,
    TOPK: tl.constexpr,
):
    """Program `(n tile, block)`. GATE_UP: `a` holds one MXFP4 row per token and
    `w` `[E, 2 N, K / 2]` (gate rows, then up rows); the output is MXFP4 per
    route. Else `a` holds one MXFP4 row per route and the output is BF16."""
    block, n_tile = tl.program_id(1), tl.program_id(0)
    expert = tl.load(block_expert_ptr + block)
    if expert >= 0:
        slot = tl.load(sorted_ptr + block * BM + tl.arange(0, BM))
        valid = slot < rows
        src = slot // TOPK if GATE_UP else slot
        n = n_tile * BN + tl.arange(0, BN)
        kp = tl.arange(0, BK // 2)
        ks = tl.arange(0, BK // 32)
        WN: tl.constexpr = 2 * N if GATE_UP else N
        w_e = w_ptr + expert.to(tl.int64) * (WN * (K // 2))
        ws_e = ws_ptr + expert.to(tl.int64) * (WN * (K // 32))
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if GATE_UP:
            up = tl.zeros((BM, BN), dtype=tl.float32)
        for tile in range(K // BK):
            a = tl.load(
                a_ptr + src[:, None] * (K // 2) + tile * (BK // 2) + kp[None, :],
                mask=valid[:, None],
                other=0,
            )
            a_s = tl.load(
                as_ptr + src[:, None] * (K // 32) + tile * (BK // 32) + ks[None, :],
                mask=valid[:, None],
                other=127,
            )
            w = _load_w(w_e, n_tile * BN, tile * (BK // 2), K // 2, BN, BK // 2)
            w_s = tl.load(ws_e + _s_offset(n, tile * (BK // 32) + ks, K // 32))
            acc = tl.dot_scaled(a, a_s, "e2m1", w, w_s, "e2m1", acc)
            if GATE_UP:
                wu = _load_w(
                    w_e, n_tile * BN + N, tile * (BK // 2), K // 2, BN, BK // 2
                )
                wu_s = tl.load(ws_e + _s_offset(n + N, tile * (BK // 32) + ks, K // 32))
                up = tl.dot_scaled(a, a_s, "e2m1", wu, wu_s, "e2m1", up)
        if GATE_UP:
            q, q_s = _silu_mul_quant(acc, up, BM, BN)
            pn = n_tile * (BN // 2) + tl.arange(0, BN // 2)
            sn = n_tile * (BN // 32) + tl.arange(0, BN // 32)
            tl.store(
                out_ptr + slot[:, None] * (N // 2) + pn[None, :], q, mask=valid[:, None]
            )
            tl.store(
                os_ptr + slot[:, None] * (N // 32) + sn[None, :],
                q_s,
                mask=valid[:, None],
            )
        else:
            tl.store(
                out_ptr + slot[:, None] * N + n[None, :],
                acc.to(out_ptr.dtype.element_ty),
                mask=valid[:, None],
            )


def compact_moe_experts(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    local_index: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """Unweighted per-(token, slot) expert outputs `[T, top_k, H]` (BF16) of the T
    bf16 rows `x` on their local routes; rows of other ranks' routes are left
    unwritten.

    Args:
        x: `[T, H]` bf16 rows.
        topk_ids: `[T, top_k]` int32 global expert ids (< 0: padding route).
        local_index: `[E]` int32 local expert of each global expert (-1:
            another rank's).
        w13: the local experts' preshuffled MXFP4 gate/up weights.
        w2: the local experts' preshuffled MXFP4 down weights.
        w13_scale: `w13`'s E8M0 scales.
        w2_scale: `w2`'s E8M0 scales (shapes: `supports_compact_moe`).

    """
    T, H = x.shape
    top_k = topk_ids.shape[1]
    num_local, inter = w13.shape[0], w2.shape[2] * 2
    assert supports_compact_moe(x, topk_ids, w13, w2, w13_scale, w2_scale), (
        "compact MoE does not support these rows, routes or weights"
    )
    cfg = tile_config(T)
    bm = cfg.block_m
    dev = x.device
    rows = T * top_k
    num_blocks = triton.cdiv(rows + num_local * (bm - 1), bm)
    sorted_slots = torch.empty(num_blocks * bm, dtype=torch.int32, device=dev)
    block_expert = torch.empty(num_blocks, dtype=torch.int32, device=dev)
    q = torch.empty((T, H // 2), dtype=torch.uint8, device=dev)
    q_scale = torch.empty((T, H // 32), dtype=torch.uint8, device=dev)
    inter_q = torch.empty((rows, inter // 2), dtype=torch.uint8, device=dev)
    inter_scale = torch.empty((rows, inter // 32), dtype=torch.uint8, device=dev)
    out = torch.empty((T, top_k, H), dtype=torch.bfloat16, device=dev)
    dc = 1024
    _route_quant_kernel[(num_local + T * (H // dc),)](
        topk_ids,
        local_index,
        sorted_slots,
        block_expert,
        x,
        q,
        q_scale,
        T,
        num_blocks,
        LE=num_local,
        S=triton.next_power_of_2(rows),
        PER_E=triton.next_power_of_2(triton.cdiv(rows, bm)),
        BM=bm,
        NBB=triton.next_power_of_2(num_blocks),
        TOPK=top_k,
        D=H,
        DC=dc,
        num_warps=4,
    )
    w13_u8, w2_u8 = w13.view(torch.uint8), w2.view(torch.uint8)
    s13_u8, s2_u8 = w13_scale.view(torch.uint8), w2_scale.view(torch.uint8)
    _expert_kernel[(inter // cfg.bn1, num_blocks)](
        q,
        q_scale,
        w13_u8,
        s13_u8,
        sorted_slots,
        block_expert,
        inter_q,
        inter_scale,
        rows,
        K=H,
        N=inter,
        BN=cfg.bn1,
        BK=cfg.bk1,
        BM=bm,
        GATE_UP=True,
        TOPK=top_k,
        num_warps=cfg.warps1,
        num_stages=cfg.stages1,
        matrix_instr_nonkdim=16,
    )
    _expert_kernel[(H // cfg.bn2, num_blocks)](
        inter_q,
        inter_scale,
        w2_u8,
        s2_u8,
        sorted_slots,
        block_expert,
        out,
        out,
        rows,
        K=inter,
        N=H,
        BN=cfg.bn2,
        BK=cfg.bk2,
        BM=bm,
        GATE_UP=False,
        TOPK=top_k,
        num_warps=cfg.warps2,
        num_stages=cfg.stages2,
        matrix_instr_nonkdim=16,
    )
    return out
