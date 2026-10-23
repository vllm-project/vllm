# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused NVFP4 W4A16 routed-expert MoE for Flint (plus decode kernels for the
BF16 checkpoint's TRTLLM-layout experts).

Numerics: activations stay BF16; weights are dequantized exactly (E2M1 * E4M3)
and products are accumulated in fp32 with the per-expert fp32 global scale
applied to the fp32 accumulator. Router logits are BF16, top-k selection and
sigmoid/renormalization are fp32 (exact exp/division) and the routing weights
are rounded to BF16. Rounding points follow the reference (TRTLLM MoE rule):
gate/up stay fp32 and act = bf16(silu(g) * u), the sigmoid computed as
rcp.approx(1 + ex2.approx(-g * log2(e))) as in FlashInfer's kernel; each expert
output is y_e = bf16(acc_e * global_scale_e); the routed sum is
bf16(sum_e tw_e * y_e) accumulated in fp32; the output is bf16(routed +
shared). The decode, decode-block and grouped paths implement this rule.

Decode (few tokens): three Triton kernels, (1) router GEMV + top-k, (2) per
(token, expert) gate/up GEMV + SiLU * up, (3) down GEMV over the top-k experts
+ weighted combine + shared-expert add. Larger batches: cuBLAS router GEMM +
Triton top-k, FlashInfer's SM100 W4A16 grouped GEMMs (see ``moe_cute``) and a
Triton combine. Both paths read the same weights in FlashInfer's layout (see
``FlintMoEWeights.from_modelopt``).
"""

from collections.abc import Iterable
from dataclasses import dataclass

import regex as re
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    interleave_linear_and_gate,
    nvfp4_swizzled_scale_to_cutedsl_mma_view,
    reorder_w13_to_w31_for_flashinfer_cutedsl,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.torch_utils import aux_stream, current_stream

from . import moe_cute, moe_gemv
from .norm import _rsqrt

# Largest token count served by the GEMV decode kernels.
DECODE_MAX_TOKENS = 6
# Largest token count served by the BF16-expert decode kernels.
BF16_DECODE_MAX_TOKENS = 12
# Largest token count routed by the router GEMV kernel (cuBLAS above).
ROUTE_GEMV_MAX_TOKENS = 64


def _ptxas_has_bf16x2_e2m1() -> bool:
    # cvt.rn.bf16x2.e2m1x2 needs PTX ISA 9.2 (ptxas >= 13.2); some images run
    # Triton with an older system ptxas.
    if not current_platform.is_cuda():
        return False
    from triton.backends.nvidia.compiler import get_ptxas

    major, minor = map(int, get_ptxas(100).version.split(".")[:2])
    return (major, minor) >= (13, 2)


# Eight E2M1 codes of one int32 word -> eight bf16 or fp16 (code value *
# scale): the hardware E2M1x2 conversion then a multiply by the E4M3 block
# scale. Both are exact in either type: E2M1 * E4M3 has at most 6 significant
# bits and lies in [2^-10, 2688]. BF16 is faster (its fp32 widening is free),
# FP16 assembles with older ptxas.
_DEQUANT_BF16 = tl.constexpr(_ptxas_has_bf16x2_e2m1())
_DEQUANT_T = "bf16" if _DEQUANT_BF16 else "f16"
_DEQUANT_ASM = tl.constexpr(
    f"""{{
.reg .b8 b0, b1, b2, b3;
.reg .b32 p0, p1, p2, p3;
mov.b32 {{b0, b1, b2, b3}}, $8;
cvt.rn.{_DEQUANT_T}x2.e2m1x2 p0, b0;
cvt.rn.{_DEQUANT_T}x2.e2m1x2 p1, b1;
cvt.rn.{_DEQUANT_T}x2.e2m1x2 p2, b2;
cvt.rn.{_DEQUANT_T}x2.e2m1x2 p3, b3;
mul.rn.{_DEQUANT_T}x2 p0, p0, $9;
mul.rn.{_DEQUANT_T}x2 p1, p1, $9;
mul.rn.{_DEQUANT_T}x2 p2, p2, $9;
mul.rn.{_DEQUANT_T}x2 p3, p3, $9;
mov.b32 {{$0, $1}}, p0;
mov.b32 {{$2, $3}}, p1;
mov.b32 {{$4, $5}}, p2;
mov.b32 {{$6, $7}}, p3;
}}"""
)


@triton.jit
def _silu(g):
    # g * sigmoid(g), sigmoid = rcp.approx(1 + ex2.approx(-g * log2(e))): the
    # instructions of FlashInfer's W4A16 MoE epilogue (and TRTLLM's).
    e = tl.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        [g * -1.4426950408889634],
        tl.float32,
        True,
        1,
    )
    sig = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [1.0 + e], tl.float32, True, 1
    )
    return g * sig


@triton.jit
def _dequant8(q, s):
    """int32 words of E2M1 codes + per-word E4M3 scales (uint8), same shape ->
    eight bf16 or fp16 tensors (code j of each word, value * scale, exact)."""
    dt = tl.bfloat16 if _DEQUANT_BF16 else tl.float16
    s = s.to(tl.float8e4nv, bitcast=True).to(tl.float32).to(dt)
    s = s.to(tl.uint16, bitcast=True).to(tl.uint32)
    s = (s | (s << 16)).to(tl.int32, bitcast=True)
    return tl.inline_asm_elementwise(
        _DEQUANT_ASM,
        "=h,=h,=h,=h,=h,=h,=h,=h,r,r",
        [q, s],
        dtype=(dt,) * 8,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _dequant_rows(q, s, BN: tl.constexpr, BK: tl.constexpr):
    """[BN, BK / 8] int32 code words + per-word E4M3 scales -> [BN, BK] bf16
    or fp16 (value * scale, exact), K in natural order."""
    o = _dequant8(q, s)
    w = tl.join(
        tl.join(tl.join(o[0], o[4]), tl.join(o[2], o[6])),
        tl.join(tl.join(o[1], o[5]), tl.join(o[3], o[7])),
    )
    return tl.reshape(w, [BN, BK])


@triton.jit
def _scale_offsets(e, rows, words, ROWS: tl.constexpr, K: tl.constexpr):
    """Byte offsets of the E4M3 block scales of (rows, int32 words) in
    FlashInfer's MMA scale layout: physical (E, ROWS / 128, K / 64, 32, 4, 4)
    for logical (row % 32, row % 128 // 32, row // 128, group % 4, group // 4)."""
    g = words // 2
    return (
        ((e * (ROWS // 128) + rows[:, None] // 128) * (K // 64) + g[None, :] // 4) * 512
        + (rows[:, None] % 32) * 16
        + (rows[:, None] % 128) // 32 * 4
        + g[None, :] % 4
    )


@triton.jit
def _topk_row(
    logits,
    offs,
    ids_ptr,
    tw_ptr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
):
    """Top-k of one row of bf16 logits (ties -> lowest expert index), fp32
    sigmoid scores renormalized over the selected experts, ``s / (sum +
    1e-20)``, rounded to bf16 (the reference packs bf16 routing weights).

    Each logit and its index are packed into one order-preserving int32 key so
    every selection step is a single max-reduction.
    """
    tl.static_assert(E <= 256)
    bits = logits.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    ordered = tl.where(bits >= 0x8000, bits ^ 0xFFFF, bits | 0x8000)
    keys = (ordered << 8) | (E - 1 - offs)
    koffs = tl.arange(0, TOPK_P2)
    sel_ids = tl.zeros([TOPK_P2], dtype=tl.int32)
    sel_s = tl.zeros([TOPK_P2], dtype=tl.float32)
    for i in tl.static_range(TOPK):
        best = tl.max(keys, axis=0)
        o = best >> 8
        b16 = tl.where(o >= 0x8000, o ^ 0x8000, o ^ 0xFFFF)
        logit = b16.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        s = tl.math.div_rn(1.0, 1.0 + tldevice.exp(-logit))
        sel_ids = tl.where(koffs == i, E - 1 - (best & 0xFF), sel_ids)
        sel_s = tl.where(koffs == i, s, sel_s)
        keys = tl.where(keys == best, -1, keys)
    kmask = koffs < TOPK
    tl.store(ids_ptr + koffs, sel_ids, mask=kmask)
    tw = tl.math.div_rn(sel_s, tl.sum(sel_s, axis=0) + 1e-20)
    tl.store(tw_ptr + koffs, tw.to(tl.bfloat16).to(tl.float32), mask=kmask)


@triton.jit
def _add_rms_norm(a_ptr, r_ptr, w_ptr, t, eps, H: tl.constexpr):
    """``norm.fused_add_rms_norm`` for row t: (bf16 residual, bf16 out), both
    from the unrounded fp32 sum s = a + r."""
    offs = tl.arange(0, H)
    a = tl.load(a_ptr + t * H + offs).to(tl.float32)
    r = tl.load(r_ptr + t * H + offs).to(tl.float32)
    s = a + r
    inv = _rsqrt(tl.sum(s * s, axis=0) / H + eps)
    w = tl.load(w_ptr + offs).to(tl.float32)
    return s.to(tl.bfloat16), (s * inv * w).to(tl.bfloat16)


@triton.jit
def _router_kernel(
    x_ptr,
    res_ptr,
    ln_w_ptr,
    eps,
    res_out_ptr,
    xn_ptr,
    wr_ptr,
    logits_ptr,
    ids_ptr,
    tw_ptr,
    counter_ptr,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    EB: tl.constexpr,
    FUSE_NORM: tl.constexpr,
):
    """Router GEMV (bf16 logits) of experts [pe * EB, + EB) for token t, then
    top-k by the token's last program. Grid (M, E // EB), or (M * E // EB,)
    with FUSE_NORM, where the GEMV input is ``norm.fused_add_rms_norm`` of x and
    res (program pe == 0 writes the residual and normed rows). Router weights
    are fetched before the PDL dependency wait.

    The last program is found with a per-token atomic counter that is never
    reset (the low bits of the signed int32 count; wraparound-safe since the
    program count is a power of two), so the buffer must be zeroed once at
    allocation and only used by launches serialized on one stream.
    """
    NR: tl.constexpr = E // EB
    tl.static_assert((NR & (NR - 1)) == 0)
    if FUSE_NORM:
        t = tl.program_id(0) // NR
        pe = tl.program_id(0) % NR
    else:
        t = tl.program_id(0)
        pe = tl.program_id(1)
    offs_e = pe * EB + tl.arange(0, EB)
    offs_k = tl.arange(0, H)
    w = tl.load(wr_ptr + offs_e[:, None] * H + offs_k[None, :]).to(tl.float32)
    tl.extra.cuda.gdc_wait()
    if FUSE_NORM:
        s, xn = _add_rms_norm(x_ptr, res_ptr, ln_w_ptr, t, eps, H)
        if pe == 0:
            tl.store(res_out_ptr + t * H + offs_k, s)
            tl.store(xn_ptr + t * H + offs_k, xn)
        x = xn.to(tl.float32)
    else:
        x = tl.load(x_ptr + t * H + offs_k).to(tl.float32)
    logits = tl.sum(w * x[None, :], axis=1).to(tl.bfloat16)
    tl.store(logits_ptr + t * E + offs_e, logits)
    tl.debug_barrier()
    count = tl.atomic_add(counter_ptr + t, 1, sem="acq_rel")
    if (count & (NR - 1)) == NR - 1:
        offs = tl.arange(0, E)
        vals = tl.load(logits_ptr + t * E + offs, volatile=True)
        _topk_row(
            vals,
            offs,
            ids_ptr + t * TOPK,
            tw_ptr + t * TOPK,
            E,
            TOPK,
            triton.next_power_of_2(TOPK),
        )


@triton.jit
def _topk_kernel(
    logits_ptr,
    ids_ptr,
    tw_ptr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
):
    """Top-k of bf16 router logits. Grid (M,)."""
    m = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, E)
    vals = tl.load(logits_ptr + m * E + offs)
    _topk_row(
        vals,
        offs,
        ids_ptr + m * TOPK,
        tw_ptr + m * TOPK,
        E,
        TOPK,
        triton.next_power_of_2(TOPK),
    )


@triton.jit
def _gate_up_gemv_kernel(
    x_ptr,
    ids_ptr,
    w_ptr,
    s_ptr,
    gs_ptr,
    act_ptr,
    H: tl.constexpr,
    INTER: tl.constexpr,
    TOPK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """act[p] = silu(x[t] @ Wg[e]^T) * (x[t] @ Wu[e]^T), pair p = t * TOPK + j.

    w_ptr: int32 view of the FlashInfer GEMM1 weight ([up; gate] interleaved
    in 64-row groups). Grid (M * TOPK, INTER // BN).
    """
    tl.extra.cuda.gdc_wait()
    p = tl.program_id(0)
    pn = tl.program_id(1)
    t = p // TOPK
    e = tl.load(ids_ptr + p).to(tl.int64)
    cols = pn * BN + tl.arange(0, BN)
    rows_u = cols // 64 * 128 + cols % 64
    rows_g = rows_u + 64
    offs_w = tl.arange(0, BK // 8)
    w_base = w_ptr + e * (2 * INTER * H // 8) + offs_w[None, :]
    # Full-shape (broadcast) activation pointers keep x in the weight tile's
    # layout (vectorized loads, no layout conversion); repeats hit L1.
    x_ptrs = x_ptr + t * H + tl.arange(0, BK)[None, :]
    x_ptrs += tl.zeros([BN, BK], dtype=tl.int32)
    acc_g = tl.zeros([BN, BK], dtype=tl.float32)
    acc_u = tl.zeros([BN, BK], dtype=tl.float32)
    for k0 in tl.static_range(0, H, BK):
        words = k0 // 8 + offs_w
        qg = tl.load(w_base + rows_g[:, None] * (H // 8) + k0 // 8)
        qu = tl.load(w_base + rows_u[:, None] * (H // 8) + k0 // 8)
        sg = tl.load(s_ptr + _scale_offsets(e, rows_g, words, 2 * INTER, H))
        su = tl.load(s_ptr + _scale_offsets(e, rows_u, words, 2 * INTER, H))
        xk = tl.load(x_ptrs + k0).to(tl.float32)
        acc_g += _dequant_rows(qg, sg, BN, BK).to(tl.float32) * xk
        acc_u += _dequant_rows(qu, su, BN, BK).to(tl.float32) * xk
    gs = tl.load(gs_ptr + e)
    g = tl.sum(acc_g, axis=1) * gs
    u = tl.sum(acc_u, axis=1) * gs
    tl.store(act_ptr + p * INTER + cols, (_silu(g) * u).to(tl.bfloat16))


@triton.jit
def _down_gemv_kernel(
    act_ptr,
    ids_ptr,
    tw_ptr,
    w_ptr,
    s_ptr,
    gs_ptr,
    shared_ptr,
    out_ptr,
    H: tl.constexpr,
    INTER: tl.constexpr,
    TOPK: tl.constexpr,
    BN: tl.constexpr,
):
    """out[t] = bf16(bf16(sum_j tw[t, j] * y_j) + shared[t]) with
    y_j = bf16(act[t * TOPK + j] @ W2[e_j]^T), weighted sum in fp32.

    w_ptr: int32 view of the down weight. Grid (M, H // BN).
    """
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pn * BN + tl.arange(0, BN)
    offs_w = tl.arange(0, INTER // 8)
    out = tl.zeros([BN], dtype=tl.float32)
    for j in tl.static_range(TOPK):
        p = t * TOPK + j
        e = tl.load(ids_ptr + p).to(tl.int64)
        q = tl.load(
            w_ptr
            + e * (H * INTER // 8)
            + rows[:, None] * (INTER // 8)
            + offs_w[None, :]
        )
        sc = tl.load(s_ptr + _scale_offsets(e, rows, offs_w, H, INTER))
        a_ptrs = act_ptr + p * INTER + tl.arange(0, INTER)[None, :]
        a = tl.load(a_ptrs + tl.zeros([BN, INTER], dtype=tl.int32)).to(tl.float32)
        acc = _dequant_rows(q, sc, BN, INTER).to(tl.float32) * a
        gs = tl.load(gs_ptr + e)
        y = (tl.sum(acc, axis=1) * gs).to(tl.bfloat16).to(tl.float32)
        out += tl.load(tw_ptr + p) * y
    shared = tl.load(shared_ptr + t * H + rows).to(tl.float32)
    out = (out.to(tl.bfloat16).to(tl.float32) + shared).to(tl.bfloat16)
    tl.store(out_ptr + t * H + rows, out)


@triton.jit
def _combine_kernel(
    y_ptr,
    perm_ptr,
    tw_ptr,
    shared_ptr,
    out_ptr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BH: tl.constexpr,
):
    """out[t] = bf16(bf16(sum_j tw[t, j] * y[perm[t * TOPK + j]]) + shared[t]),
    weighted sum in fp32.

    Grid (M, H // BH).
    """
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BH + tl.arange(0, BH)
    acc = tl.zeros([BH], dtype=tl.float32)
    for j in tl.static_range(TOPK):
        row = tl.load(perm_ptr + t * TOPK + j).to(tl.int64)
        y = tl.load(y_ptr + row * H + offs).to(tl.float32)
        acc += tl.load(tw_ptr + t * TOPK + j) * y
    shared = tl.load(shared_ptr + t * H + offs).to(tl.float32)
    out = (acc.to(tl.bfloat16).to(tl.float32) + shared).to(tl.bfloat16)
    tl.store(out_ptr + t * H + offs, out)


@triton.jit
def _block_major_offsets(rows, k, ROWS: tl.constexpr):
    """Element offsets of (rows, k) in FlashInfer's BF16 BlockMajorK expert
    layout: [K / 64, ROWS, 64]."""
    return (k[None, :] // 64) * (ROWS * 64) + rows[:, None] * 64 + k[None, :] % 64


@triton.jit
def _gate_up_gemv_bf16_kernel(
    x_ptr,
    ids_ptr,
    w_ptr,
    rowmap_ptr,
    act_ptr,
    H: tl.constexpr,
    INTER: tl.constexpr,
    TOPK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """BF16 experts: act[p] = silu(x[t] @ Wg[e]^T) * (x[t] @ Wu[e]^T).

    w_ptr: TRTLLM BlockMajorK gate/up weight [E, H / 64, 2I, 64]; rowmap:
    logical row (gate rows, then up rows) -> physical row. Grid
    (M * TOPK, INTER // BN).
    """
    tl.extra.cuda.gdc_wait()
    p = tl.program_id(0)
    pn = tl.program_id(1)
    t = p // TOPK
    e = tl.load(ids_ptr + p).to(tl.int64)
    cols = pn * BN + tl.arange(0, BN)
    rows_g = tl.load(rowmap_ptr + cols)
    rows_u = tl.load(rowmap_ptr + INTER + cols)
    offs_k = tl.arange(0, BK)
    w_base = w_ptr + e * (2 * INTER * H)
    x_ptrs = x_ptr + t * H + offs_k[None, :] + tl.zeros([BN, BK], dtype=tl.int32)
    acc_g = tl.zeros([BN, BK], dtype=tl.float32)
    acc_u = tl.zeros([BN, BK], dtype=tl.float32)
    for k0 in tl.static_range(0, H, BK):
        k = k0 + offs_k
        wg = tl.load(w_base + _block_major_offsets(rows_g, k, 2 * INTER))
        wu = tl.load(w_base + _block_major_offsets(rows_u, k, 2 * INTER))
        xk = tl.load(x_ptrs + k0).to(tl.float32)
        acc_g += wg.to(tl.float32) * xk
        acc_u += wu.to(tl.float32) * xk
    g = tl.sum(acc_g, axis=1)
    u = tl.sum(acc_u, axis=1)
    # TRTLLM BF16 MoE semantics (the BF16 baseline used for larger batches):
    # gate/up stay fp32, one bf16 rounding of silu(g) * u.
    tl.store(act_ptr + p * INTER + cols, (_silu(g) * u).to(tl.bfloat16))


@triton.jit
def _down_gemv_bf16_kernel(
    act_ptr,
    ids_ptr,
    tw_ptr,
    w_ptr,
    rowmap_ptr,
    shared_ptr,
    out_ptr,
    H: tl.constexpr,
    INTER: tl.constexpr,
    TOPK: tl.constexpr,
    BN: tl.constexpr,
):
    """BF16 experts: out[t] = bf16(bf16(sum_j tw[t, j] * y_j) + shared[t]),
    y_j = bf16(act[t * TOPK + j] @ W2[e_j]^T).

    w_ptr: TRTLLM BlockMajorK down weight [E, I / 64, H, 64]; rowmap: output
    row -> physical row. Grid (M, H // BN).
    """
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pn * BN + tl.arange(0, BN)
    prow = tl.load(rowmap_ptr + rows)
    offs_k = tl.arange(0, INTER)
    w_offs = _block_major_offsets(prow, offs_k, H)
    out = tl.zeros([BN], dtype=tl.float32)
    for j in tl.static_range(TOPK):
        p = t * TOPK + j
        e = tl.load(ids_ptr + p).to(tl.int64)
        w = tl.load(w_ptr + e * (H * INTER) + w_offs).to(tl.float32)
        a_ptrs = act_ptr + p * INTER + offs_k[None, :]
        a = tl.load(a_ptrs + tl.zeros([BN, INTER], dtype=tl.int32)).to(tl.float32)
        y = tl.sum(w * a, axis=1).to(tl.bfloat16).to(tl.float32)
        out += tl.load(tw_ptr + p) * y
    shared = tl.load(shared_ptr + t * H + rows).to(tl.float32)
    out = (out.to(tl.bfloat16).to(tl.float32) + shared).to(tl.bfloat16)
    tl.store(out_ptr + t * H + rows, out)


def trtllm_bf16_row_maps(
    hidden_size: int, intermediate_size: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logical -> physical row maps of the TRTLLM BF16 expert weights as
    produced by ``convert_moe_weights_to_flashinfer_trtllm_block_layout``
    (gated gate/up interleave + epilogue-tile shuffle). Returns (w13 map over
    [gate rows; up rows], w2 map over output rows), int32."""
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    rows13 = 2 * intermediate_size
    w13 = torch.empty(rows13, hidden_size * 2, dtype=torch.uint8, device=device)
    w2 = torch.empty(
        hidden_size, intermediate_size * 2, dtype=torch.uint8, device=device
    )
    src13 = _maybe_get_cached_w3_w1_permute_indices(
        {}, w13, 128, is_gated_act_gemm=True
    )
    src13 = (src13.to(device) + rows13 // 2) % rows13
    src2 = get_w2_permute_indices_with_cache({}, w2, 128).to(device)
    # Physical row j holds logical row src[j]; invert.
    return (
        torch.argsort(src13).to(torch.int32).contiguous(),
        torch.argsort(src2).to(torch.int32).contiguous(),
    )


def flint_moe_bf16_decode(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    row_maps: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    counter: torch.Tensor,
    shared_output: torch.Tensor,
    shared_ready: torch.cuda.Event | None = None,
) -> torch.Tensor:
    """Decode-path MoE on TRTLLM-layout BF16 experts (w13 [E, 2I, H], w2
    [E, H, I] storage in BlockMajorK order) with the TRTLLM BF16 MoE rounding
    points: act = bf16(silu(g) * u) from fp32 gate/up, y_e = bf16(acc),
    routed = bf16(sum tw * y_e) in fp32, out = bf16(routed + shared)."""
    num_tokens, hidden = x.shape
    inter = w2.shape[2]
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    ids, tw = route(x, router_weight, top_k, counter)
    act = torch.empty(num_tokens * top_k, inter, dtype=x.dtype, device=x.device)
    _gate_up_gemv_bf16_kernel[(num_tokens * top_k, inter)](
        x,
        ids,
        w13,
        row_maps[0],
        act,
        H=hidden,
        INTER=inter,
        TOPK=top_k,
        BN=1,
        BK=256,
        num_warps=1,
        launch_pdl=True,
    )
    out = torch.empty_like(x)
    if shared_ready is not None:
        shared_ready.wait(current_stream())
    bn = 1 if num_tokens <= 2 else 2
    _down_gemv_bf16_kernel[(num_tokens, hidden // bn)](
        act,
        ids,
        tw,
        w2,
        row_maps[1],
        shared_output,
        out,
        H=hidden,
        INTER=inter,
        TOPK=top_k,
        BN=bn,
        num_warps=1,
        launch_pdl=True,
    )
    return out


@dataclass
class FlintMoEWeights:
    """Routed-expert NVFP4 weights in FlashInfer's W4A16 layout."""

    w1: torch.Tensor  # uint8 [E, 2I, H/2], [up; gate] interleaved by 64 rows
    w1_sf: torch.Tensor  # float8_e4m3fn, 6D MMA scale layout
    w1_alpha: torch.Tensor  # fp32 [E] global scale of gate/up
    w2: torch.Tensor  # uint8 [E, H, I/2]
    w2_sf: torch.Tensor
    w2_alpha: torch.Tensor

    @classmethod
    def from_modelopt(
        cls,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w13_scale_2: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        w2_scale_2: torch.Tensor,
    ) -> "FlintMoEWeights":
        """From ModelOpt tensors: w13 [E, 2I, H/2] ([gate; up]), w13_scale
        [E, 2I, H/16], w2 [E, H, I/2], w2_scale [E, H, I/16], fp32 [E]."""
        w1, w1_sf = reorder_w13_to_w31_for_flashinfer_cutedsl(
            MoEActivation.SILU, w13, w13_scale
        )
        w1 = interleave_linear_and_gate(w1, group_size=64, dim=1)
        w1_sf = interleave_linear_and_gate(w1_sf, group_size=64, dim=1)
        return cls(
            w1,
            nvfp4_swizzled_scale_to_cutedsl_mma_view(swizzle_blockscale(w1_sf)),
            w13_scale_2.float().contiguous(),
            w2.contiguous(),
            nvfp4_swizzled_scale_to_cutedsl_mma_view(swizzle_blockscale(w2_scale)),
            w2_scale_2.float().contiguous(),
        )

    @property
    def intermediate_size(self) -> int:
        return self.w2.shape[2] * 2


def make_route_counter(device: torch.device | str | None = None) -> torch.Tensor:
    """Zeroed counter buffer for the router GEMV kernel; allocate it outside
    CUDA graph capture and use it from one stream only."""
    return torch.zeros(ROUTE_GEMV_MAX_TOKENS, dtype=torch.int32, device=device)


def route(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
    counter: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 router logits -> (top-k ids int32, renormalized fp32 weights)."""
    num_tokens, hidden = x.shape
    num_experts = router_weight.shape[0]
    ids = torch.empty(num_tokens, top_k, dtype=torch.int32, device=x.device)
    tw = torch.empty(num_tokens, top_k, dtype=torch.float32, device=x.device)
    if num_tokens <= ROUTE_GEMV_MAX_TOKENS:
        logits = torch.empty(
            num_tokens, num_experts, dtype=torch.bfloat16, device=x.device
        )
        eb = 2
        _router_kernel[(num_tokens, num_experts // eb)](
            x,
            x,
            x,
            0.0,
            x,
            x,
            router_weight,
            logits,
            ids,
            tw,
            counter,
            H=hidden,
            E=num_experts,
            TOPK=top_k,
            EB=eb,
            FUSE_NORM=False,
            num_warps=4,
            launch_pdl=True,
        )
    else:
        logits = torch.mm(x, router_weight.t())
        _topk_kernel[(num_tokens,)](
            logits, ids, tw, E=num_experts, TOPK=top_k, num_warps=4
        )
    return ids, tw


def flint_moe(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    weights: FlintMoEWeights,
    top_k: int,
    counter: torch.Tensor,
    shared_output: torch.Tensor,
    shared_ready: torch.cuda.Event | None = None,
) -> torch.Tensor:
    """Routed-expert MoE output + shared_output for BF16 x [M, H].

    counter: buffer from ``make_route_counter``. shared_ready: event the
    current stream waits on before the first read of shared_output (which
    may be produced on another stream).
    """
    num_tokens, hidden = x.shape
    inter = weights.intermediate_size
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    ids, tw = route(x, router_weight, top_k, counter)
    out = torch.empty_like(x)
    if num_tokens <= DECODE_MAX_TOKENS:
        act = torch.empty(num_tokens * top_k, inter, dtype=x.dtype, device=x.device)
        _gate_up_gemv_kernel[(num_tokens * top_k, inter)](
            x,
            ids,
            weights.w1.view(torch.int32),
            weights.w1_sf,
            weights.w1_alpha,
            act,
            H=hidden,
            INTER=inter,
            TOPK=top_k,
            BN=1,
            BK=256,
            num_warps=1,
            launch_pdl=True,
        )
        bn = 1 if num_tokens <= 2 else 2
        if shared_ready is not None:
            shared_ready.wait(current_stream())
        _down_gemv_kernel[(num_tokens, hidden // bn)](
            act,
            ids,
            tw,
            weights.w2.view(torch.int32),
            weights.w2_sf,
            weights.w2_alpha,
            shared_output,
            out,
            H=hidden,
            INTER=inter,
            TOPK=top_k,
            BN=bn,
            num_warps=1,
            launch_pdl=True,
        )
        return out
    y, perm = moe_cute.grouped_moe(
        x,
        ids,
        tw,
        weights.w1,
        weights.w1_sf,
        weights.w1_alpha,
        weights.w2,
        weights.w2_sf,
        weights.w2_alpha,
    )
    if shared_ready is not None:
        shared_ready.wait(current_stream())
    bh = 1024
    _combine_kernel[(num_tokens, hidden // bh)](
        y,
        perm,
        tw,
        shared_output,
        out,
        H=hidden,
        TOPK=top_k,
        BH=bh,
        num_warps=4,
        launch_pdl=True,
    )
    return out


_EXPERT_PARAM = re.compile(
    r"(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_2)"
)


def _start_shared(
    shared: torch.Tensor | nn.Module,
    x: torch.Tensor,
    fork: torch.cuda.Event,
    join: torch.cuda.Event,
) -> tuple[torch.Tensor, torch.cuda.Event | None]:
    """Run a shared-expert module on the auxiliary stream (fork/join events,
    CUDA-graph safe). Returns (output, event to wait on before reading it)."""
    if not isinstance(shared, nn.Module):
        return shared, None
    stream = aux_stream()
    fork.record(current_stream())
    with torch.cuda.stream(stream):
        fork.wait(stream)
        out = shared(x)
        join.record(stream)
    return out, join


class FlintBf16DecodeMoE(nn.Module):
    """Decode path (few tokens) for Flint's BF16 routed experts.

    Reads the expert weights of a vLLM FusedMoE layer that uses the FlashInfer
    TRTLLM BF16 backend (BlockMajorK, rows shuffled), so no extra weight copy
    is needed; larger batches keep using that layer.
    """

    def __init__(
        self, num_experts: int, hidden_size: int, intermediate_size: int, top_k: int
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        device = torch.empty(0).device
        self.register_buffer("route_counter", make_route_counter(device), False)
        map13, map2 = trtllm_bf16_row_maps(hidden_size, intermediate_size, device)
        self.register_buffer("row_map13", map13, persistent=False)
        self.register_buffer("row_map2", map2, persistent=False)
        self._fork = torch.cuda.Event()
        self._join = torch.cuda.Event()
        self._checked = False

    def _check(self, routed_experts: nn.Module) -> None:
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            UnquantizedMoeBackend,
        )

        backend = getattr(routed_experts.quant_method, "unquantized_backend", None)
        e, h, i = self.num_experts, self.hidden_size, self.intermediate_size
        if (
            backend != UnquantizedMoeBackend.FLASHINFER_TRTLLM
            or tuple(routed_experts.w13_weight.shape) != (e, 2 * i, h)
            or tuple(routed_experts.w2_weight.shape) != (e, h, i)
        ):
            raise RuntimeError(
                "Flint BF16 decode MoE requires the FlashInfer TRTLLM BF16 MoE "
                f"backend without padding (got {backend})"
            )
        self._checked = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        routed_experts: nn.Module,
        shared: torch.Tensor | nn.Module,
    ) -> torch.Tensor:
        if not self._checked:
            self._check(routed_experts)
        shared, ready = _start_shared(shared, hidden_states, self._fork, self._join)
        return flint_moe_bf16_decode(
            hidden_states,
            router_weight,
            routed_experts.w13_weight,
            routed_experts.w2_weight,
            (self.row_map13, self.row_map2),
            self.top_k,
            self.route_counter,
            shared,
            ready,
        )


class FlintNvfp4Experts(nn.Module):
    """Routed experts of one Flint MoE layer (NVFP4 GGUF checkpoints).

    Loads per-expert ModelOpt NVFP4 tensors (``{e}.{gate,up,down}_proj.
    {weight, weight_scale, weight_scale_2}``) and, once every expert is loaded,
    repacks them into ``FlintMoEWeights`` and frees the checkpoint layout.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        max_num_tokens: int = 0,
    ) -> None:
        """max_num_tokens: largest batch; the grouped-GEMM kernels for every
        tactic up to it are compiled when the weights are loaded."""
        super().__init__()
        self.max_num_tokens = max_num_tokens
        e, h, i = num_experts, hidden_size, intermediate_size
        self.num_experts = e
        self.top_k = top_k

        def param(*shape: int, dtype: torch.dtype) -> nn.Parameter:
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        self.w13_weight = param(e, 2 * i, h // 2, dtype=torch.uint8)
        self.w13_weight_scale = param(e, 2 * i, h // 16, dtype=torch.float8_e4m3fn)
        self.w13_weight_scale_2 = param(e, 2, dtype=torch.float32)
        self.w2_weight = param(e, h, i // 2, dtype=torch.uint8)
        self.w2_weight_scale = param(e, h, i // 16, dtype=torch.float8_e4m3fn)
        self.w2_weight_scale_2 = param(e, dtype=torch.float32)
        self.register_buffer("route_counter", make_route_counter(), persistent=False)
        self._loaded_names: set[str] = set()
        self._fork = torch.cuda.Event()
        self._join = torch.cuda.Event()
        self.weights: FlintMoEWeights | None = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        inter = self.w2_weight.shape[2] * 2
        for name, w in weights:
            m = _EXPERT_PARAM.fullmatch(name)
            if m is None:
                raise ValueError(f"Unexpected Flint expert weight: {name}")
            e, proj, kind = int(m.group(1)), m.group(2), m.group(3)
            param_name = ("w2_" if proj == "down_proj" else "w13_") + kind
            param = getattr(self, param_name)
            if proj == "down_proj":
                param.data[e].copy_(w)
            elif kind == "weight_scale_2":
                param.data[e, int(proj == "up_proj")].copy_(w)
            else:
                rows = slice(inter, 2 * inter) if proj == "up_proj" else slice(inter)
                param.data[e, rows].copy_(w)
            loaded.add(param_name)
            self._loaded_names.add(name)
        if len(self._loaded_names) == 9 * self.num_experts:
            self._repack()
        return loaded

    def _repack(self) -> None:
        gs = self.w13_weight_scale_2.data
        if not torch.equal(gs[:, 0], gs[:, 1]):
            raise ValueError("Flint NVFP4 gate/up global scales must match")
        self.weights = FlintMoEWeights.from_modelopt(
            self.w13_weight.data,
            self.w13_weight_scale.data,
            gs[:, 0],
            self.w2_weight.data,
            self.w2_weight_scale.data,
            self.w2_weight_scale_2.data,
        )
        for name in list(self._parameters):
            delattr(self, name)
        if self.max_num_tokens > DECODE_MAX_TOKENS:
            moe_cute.warmup(
                self.weights, self.top_k, DECODE_MAX_TOKENS + 1, self.max_num_tokens
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        shared: torch.Tensor | nn.Module,
    ) -> torch.Tensor:
        """shared: shared-expert output, or the shared-expert module, which
        then runs on the auxiliary stream concurrently with the routed path."""
        if self.weights is None:
            raise RuntimeError("Flint NVFP4 expert weights were not fully loaded")
        shared, ready = _start_shared(shared, hidden_states, self._fork, self._join)
        return flint_moe(
            hidden_states,
            router_weight,
            self.weights,
            self.top_k,
            self.route_counter,
            shared,
            ready,
        )


# Low-latency decode MoE block: each kernel prefetches weights before its PDL wait.

# Largest batch for which the fused decode MoE block beats the regular path.
LL_MOE_MAX_TOKENS = 2


@dataclass
class LLMoEWeights:
    """Weights of one NVFP4 MoE layer for ``flint_moe_ll`` (views, no copies).

    Shared expert: ModelOpt-packed weights (int32 views) with flat
    128x4-swizzled scale bytes and fp32 global scales.
    """

    router: torch.Tensor  # bf16 [E, H]
    experts: FlintMoEWeights
    shared_gu: torch.Tensor  # [2 * SI, H / 8] int32
    shared_gu_scale: torch.Tensor
    shared_gu_gs: torch.Tensor
    shared_down: torch.Tensor  # [H, SI / 8] int32
    shared_down_scale: torch.Tensor
    shared_down_gs: torch.Tensor


def flint_moe_ll(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    ln_weight: torch.Tensor,
    eps: float,
    w: LLMoEWeights,
    top_k: int,
    counter: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode MoE block for M <= LL_MOE_MAX_TOKENS tokens.

    Computes ``residual' = attn_out + residual`` and ``x = rms_norm(residual')``
    as ``norm.fused_add_rms_norm``, then the routed + shared expert MoE of
    ``x`` (NVFP4 rule). Returns ``(moe_out, residual')``; inputs are not
    modified. A Triton router kernel and two CuTe split-K expert kernels
    (``moe_gemv``) chained with programmatic dependent launch. counter: buffer
    from ``make_route_counter``.
    """
    m, hidden = attn_out.shape
    assert 0 < m <= LL_MOE_MAX_TOKENS and attn_out.is_contiguous()
    assert residual.is_contiguous()
    num_experts = w.router.shape[0]
    si = w.shared_down.shape[1] * 8
    ew = w.experts
    dev = attn_out.device
    bf16 = torch.bfloat16
    res_out = torch.empty_like(residual)
    xn = torch.empty_like(attn_out)
    logits = torch.empty(m, num_experts, dtype=bf16, device=dev)
    ids = torch.empty(m, top_k, dtype=torch.int32, device=dev)
    tw = torch.empty(m, top_k, dtype=torch.float32, device=dev)
    sact = torch.empty(m, si, dtype=bf16, device=dev)
    act = torch.empty(m * top_k, ew.intermediate_size, dtype=bf16, device=dev)
    out = torch.empty_like(attn_out)
    eb = 2
    _router_kernel[(m * (num_experts // eb),)](
        attn_out,
        residual,
        ln_weight,
        eps,
        res_out,
        xn,
        w.router,
        logits,
        ids,
        tw,
        counter,
        H=hidden,
        E=num_experts,
        TOPK=top_k,
        EB=eb,
        FUSE_NORM=True,
        num_warps=2,
        launch_pdl=True,
    )
    moe_gemv.gate_up(
        xn,
        ids,
        ew.w1,
        ew.w1_sf,
        ew.w1_alpha,
        act,
        w.shared_gu,
        w.shared_gu_scale,
        w.shared_gu_gs,
        sact,
    )
    moe_gemv.down_combine(
        act,
        ids,
        tw,
        ew.w2,
        ew.w2_sf,
        ew.w2_alpha,
        sact,
        w.shared_down.view(torch.uint8),
        w.shared_down_scale,
        w.shared_down_gs,
        out,
    )
    return out, res_out


def ll_weights_from_layers(
    router_weight: torch.Tensor,
    shared_gate_up: nn.Module,
    shared_down: nn.Module,
    experts: FlintMoEWeights,
) -> LLMoEWeights:
    """``LLMoEWeights`` viewing the loaded layers' tensors: the shared-expert
    linears are vLLM linear layers holding the FlashInfer-prepared NVFP4
    tensors (ModelOpt-packed weight, 128x4-swizzled scales, fp32 global
    scale)."""

    def nvfp4(layer: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = layer.weight
        n, k2 = weight.shape
        k = k2 * 2
        scale = layer.weight_scale
        nbytes = (n + 127) // 128 * 128 * ((k // 16 + 3) // 4 * 4)
        storage = scale.untyped_storage()
        assert (
            weight.dtype == torch.uint8
            and weight.is_contiguous()
            and not getattr(layer, "weights_padding_cols", 0)
            and scale.storage_offset() == 0
            and storage.nbytes() == nbytes
            and n % 128 == 0
            and k % 64 == 0
        ), "Unexpected NVFP4 shared-expert weight layout"
        flat = torch.empty(0, dtype=torch.uint8, device=weight.device).set_(
            storage, 0, (nbytes,), (1,)
        )
        gs = layer.weight_global_scale.reshape(1).float()
        return weight.view(torch.int32), flat, gs

    gu_w, gu_s, gu_gs = nvfp4(shared_gate_up)
    dn_w, dn_s, dn_gs = nvfp4(shared_down)
    return LLMoEWeights(
        router=router_weight,
        experts=experts,
        shared_gu=gu_w,
        shared_gu_scale=gu_s,
        shared_gu_gs=gu_gs,
        shared_down=dn_w,
        shared_down_scale=dn_s,
        shared_down_gs=dn_gs,
    )
