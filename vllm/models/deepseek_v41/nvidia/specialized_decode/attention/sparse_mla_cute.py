# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sparse-MLA decode of small batches, in CuTe DSL.

The problem. In a decode step every query token (DeepSeek-V4's MLA after the
query projection: 16 heads a rank at TP4, FP8 E4M3, head dim 512) attends to
its own sparse set of KV rows. A row is 512 FP8 bytes and serves as both K and
V. The set holds up to 640 columns: the 128 sliding-window rows of one KV pool,
then the indexer's top-k rows of the compressed pool, all addressed as storage
offsets into their pool, with -1 for an unused column and a per-token count of
active columns. TRTLLM-gen splits a token's 640 rows over at most 3 CTAs, so at
one or two requests (6 or 12 tokens with DSpark) its kernel runs about 10 µs
on a GPU of 152 SMs; most of that is the latency of one CTA's serial work.

What this module computes. ``sparse_mla_decode`` writes, for each token ``t``
and head ``h``, the BF16 attention output over the token's active columns ``c``
(``c < lens[t]`` and ``indices[t, c] >= 0``)::

    s_c = bmm1_scale * q[t, h] . kv[c]
    p_c = exp(s_c - m)
    out[t, h] = bmm2_scale * sum_c p_c kv[c] / (sum_c p_c + exp(sink_h - m))

where ``kv[c]`` is row ``indices[t, c]`` of the sliding-window pool for the
first 128 columns and of the compressed pool for the others, ``m`` is the
maximum of the scores and the sink, and the sink term is omitted without sinks.
A token with no active column gets zeros. This is what
``trtllm_batch_decode_sparse_mla_dsv4`` computes for these shapes.

How. One CTA of 128 threads takes one token and one 128-column chunk of its
sparse set (chunk 0 = the sliding-window columns); the ``n = ceil(width / 128)``
chunks of a token form a thread block cluster, so a 640-wide layer runs 5 CTAs
a token in parallel::

    gather the chunk's KV rows and the query (cp.async, one group per 128-byte K
    block) into 128B-swizzled tiles
      └► S^T = KV x Q^T on tcgen05 (M 128 rows, N 16 heads, K 512, FP8 -> FP32 TMEM)
           └► chunk softmax per head in base 2: max m_c, sum l_c, and
              P^T = 256 * exp2(S - m_c) as FP8 (scaled: small weights stay normal)
                └► O^T = V^T x P^T on tcgen05, V the same tile read MN-major
                   (four MMAs of 128 head dims) ...
                └► ... while the cluster exchanges (m_c, l_c) per head (st.async)
                     └► each CTA scales its partial O by exp2(m_c - M) / L
                        (M, L over the cluster's chunks and the sink)
                          └► block j of the 512 head dims goes to CTA j % n, which
                             sums the n partials and stores BF16

The partial blocks travel by bulk copies between the CTAs' shared memory into
the receiving CTA's KV tile, whose MMAs are done by then: a cluster barrier
after each CTA's second MMA separates the two uses. A last cluster barrier keeps
every CTA resident until the bulk copies reading its shared memory are complete.

Synchronization. The kernel issues the loads no earlier kernel of the step
writes (the sinks and, with ``QRope``, the query scale and the token's
position) before its grid dependency wait and everything else after it, the
index loads first. It lets its dependents launch once its S MMAs are issued
(programmatic dependent launch). Masked columns gather zeros and score -inf.

Query RoPE (``QRope``). The kernel can instead take ``wq_b``'s BF16 output and
apply the query's GPT-J RoPE (on dims 448-511), its FP8 scale and E4M3 rounding
as it loads it, in the rounding of the CUDA kernel it replaces
(``fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert`` without the Q
norm): the same E4M3 bytes land in the query tile. The KV insert then runs
elsewhere, and the attention follows ``wq_b`` directly (its PDL prologue
overlapping ``wq_b``'s tail).

Local indices (``LocalIndices``). The kernel can instead take the sparse set
before ``build_flashinfer_mixed_sparse_indices`` turns it into storage offsets:
the token's SWA slot ids and its indexer's request-local top-k positions, which
it translates as it loads them (the SWA ids through the pool's block span, the
top-k positions through the request's compressed block table and span) and
masks as that build's lengths do. The KV rows it gathers are the table's, so an
index source's attention follows its top-k directly, without the table build in
between.

L2 prefetch (``SparseMlaPrefetchCute``). It reads a built sparse-index table
and prefetches each active column's KV row (and the sinks and the query scale)
into L2; ``sparse_mla.SparseMlaPrefetch`` runs it on a side stream at the start
of each layer whose table an earlier layer built, before the attention gathers
the same rows.

Scope. Two CTAs fit an SM (about 78 KB of shared memory each), so a launch is
one wave only while every token's cluster can be resident: ``_max_tokens`` allows
2 x the device's resident clusters of ``n`` CTAs at one CTA an SM (on a 152-SM
GB300, 8 requests with DSpark at width 640).
Larger batches keep TRTLLM-gen, which is faster once the clusters run in waves.
"""

import math
from functools import cache
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.cute_utils import _tcgen05, recast_val
from vllm.cute_utils import mbarrier as mbar_ops

from ...ops.cute_dsl.primitives import current_cu_stream

HEADS = 16
HEAD_DIM = 512
ROWS = 128  # KV rows (columns of the sparse set) a CTA
SWA_COLUMNS = ROWS  # chunk 0: the sliding-window columns, from the SWA pool
THREADS = 128
MAX_CHUNKS = 5  # widths up to 640
BLOCK_BYTES = ROWS * 128  # one 128-byte K block of the KV tile
Q_BLOCK_BYTES = HEADS * 128  # one 128-byte K block of the query tile
PART_BYTES = ROWS * HEADS * 4  # one 128-dim block of a partial O, FP32
TMEM_COLS = 128  # S: 16 columns, O: 4 x 16
P_SCALE = 256.0
LOG2E = 1.4426950408889634
CTAS_PER_SM = 2
# tcgen05 instruction descriptors, kind::f8f6f4: D FP32, A and B E4M3, M = 128,
# N = 16; bit 15 makes A MN-major (V^T read from the K-major KV tile).
IDESC_QK = (1 << 4) | ((HEADS >> 3) << 17) | ((ROWS >> 4) << 24)
IDESC_PV = IDESC_QK | (1 << 15)
# The query RoPE's cos || sin rows: 2 x 32 FP32 a position.
ROPE_ROW_BYTES = 64 * 4
# The local top-k columns: four a thread, counted as the table build counts them.
LOCAL_TOPK = 4 * THREADS
# The E4M3 range the CUDA kernel clamps to before its saturating conversion
# (+-448 as PTX hexadecimal floats).
_CLAMP = "max.f32 {r}, {r}, 0fC3E00000;\n\tmin.f32 {r}, {r}, 0f43E00000;\n\t"


class QRope(NamedTuple):
    """What ``SparseMlaDecodeCute`` needs to rope and quantize a BF16 query as
    it loads it."""

    # [>= tokens] int64 positions of the query's tokens.
    positions: torch.Tensor
    # [max_position, 64] FP32: each position's 32 cos, then its 32 sin.
    cos_sin_cache: torch.Tensor
    # [1] FP32: the FP8 query's reciprocal scale.
    scale_inv: torch.Tensor


class LocalIndices(NamedTuple):
    """The sparse set of a step's decode tokens as the inputs of
    ``build_flashinfer_mixed_sparse_indices`` (decode tokens, request-local
    top-k), which ``SparseMlaDecodeCute`` translates as it loads them."""

    # [tokens, 512] int32 request-local compressed positions, -1 for none.
    topk: torch.Tensor
    # [tokens, 128] int32 SWA slot ids (block * block_size + offset), -1 for none.
    swa: torch.Tensor
    swa_block_size: int
    # Rows from one SWA block to the next in the SWA pool (packed pages).
    swa_block_span: int
    # [>= tokens] int32 request of each token.
    token_to_req: torch.Tensor
    # [>= tokens] bool: the token's compressed columns count.
    token_valid: torch.Tensor
    # [requests, blocks] int32 compressed block table.
    block_table: torch.Tensor
    block_size: int
    block_span: int


def _sdesc(lbo: int, sbo: int) -> int:
    """A 128B-swizzled shared-memory matrix descriptor without its start
    address: leading and stride byte offsets, version 1."""
    return ((lbo >> 4) << 16) | ((sbo >> 4) << 32) | (1 << 46) | (2 << 61)


SDESC_K = _sdesc(0, 1024)  # K-major operands: 8-row groups 1024 bytes apart
SDESC_MN = _sdesc(BLOCK_BYTES, 1024)  # V^T: 128-row K blocks a tile block apart


def _asm(result, operands, text, constraints, side_effects, loc, ip):
    """One PTX inline assembly operation."""
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(value, loc, ip):
    """IR value of a DSL scalar."""
    return value.ir_value(loc=loc, ip=ip)


@dsl_user_op
def cp_async_16(smem: Int32, gmem: Int64, *, loc=None, ip=None) -> None:
    """Copy 16 bytes from global to this CTA's shared memory (cp.async)."""
    _asm(
        None,
        [_v(Int32(smem), loc, ip), _v(Int64(gmem), loc, ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_shared_zero16(smem: Int32, *, loc=None, ip=None) -> None:
    """Store 16 zero bytes at a 16-byte aligned shared address."""
    _asm(
        None,
        [_v(Int32(smem), loc, ip), _v(Int32(0), loc, ip)],
        "st.shared.v4.b32 [$0], {$1, $1, $1, $1};",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_shared_v4(
    smem: Int32, w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32, *, loc=None, ip=None
) -> None:
    """Store four 32-bit words at a 16-byte aligned shared address."""
    _asm(
        None,
        [_v(Int32(smem), loc, ip)] + [_v(Uint32(w), loc, ip) for w in (w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_shared_bf16(smem: Int32, value: Float32, *, loc=None, ip=None) -> None:
    """Store ``value`` rounded to BF16 at a shared address."""
    _asm(
        None,
        [_v(Int32(smem), loc, ip), _v(Float32(value), loc, ip)],
        "{\n\t.reg .b16 b;\n\tcvt.rn.bf16.f32 b, $1;\n\tst.shared.b16 [$0], b;\n\t}",
        "r,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def ld_global_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    """A 32-bit integer from global memory an earlier grid wrote (side-effecting,
    so it stays after the grid dependency wait)."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(address), loc, ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_global_u8(address: Int64, *, loc=None, ip=None) -> Int32:
    """A byte from global memory, zero-extended (side-effecting, so it stays
    after the grid dependency wait)."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(address), loc, ip)],
            "ld.global.u8 $0, [$1];",
            "=r,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_global_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    """A float from read-only global memory (side-effecting, so it stays where
    it is issued, before the grid dependency wait, instead of moving to its
    first use)."""
    return Float32(
        _asm(
            T.f32(),
            [_v(Int64(address), loc, ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def prefetch_l2_row(address: Int64, *, loc=None, ip=None) -> None:
    """Prefetch the 512-byte KV row at a 16-byte aligned global address into
    L2."""
    _asm(
        None,
        [_v(Int64(address), loc, ip), _v(Int32(HEAD_DIM), loc, ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def prefetch_l2_line(address: Int64, *, loc=None, ip=None) -> None:
    """Prefetch the cache line of a global address into L2."""
    _asm(
        None,
        [_v(Int64(address), loc, ip)],
        "prefetch.global.L2 [$0];",
        "l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def ld_global_s64(address: Int64, *, loc=None, ip=None) -> Int64:
    """A 64-bit integer from global memory no kernel of the step writes
    (side-effecting, so it stays where it is issued, before the grid dependency
    wait)."""
    return Int64(
        _asm(
            T.i64(),
            [_v(Int64(address), loc, ip)],
            "ld.global.b64 $0, [$1];",
            "=l,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_global_v4(address: Int64, *, loc=None, ip=None) -> list:
    """Four 32-bit words at a 16-byte aligned global address an earlier grid
    wrote (side-effecting, so they stay after the grid dependency wait)."""
    result = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_v(Int64(address), loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        True,
        loc,
        ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def ld_global_nc_v4_f32(address: Int64, *, loc=None, ip=None) -> list:
    """Four floats at a 16-byte aligned address of read-only global memory."""
    result = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_v(Int64(address), loc, ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        False,
        loc,
        ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def query_e4m3x4(
    lo: Uint32, hi: Uint32, scale: Float32, *, loc=None, ip=None
) -> Uint32:
    """The four BF16 query values of ``lo`` and ``hi`` (each word's low half
    first) times ``scale``, clamped to +-448 and rounded to E4M3, the first in
    the lowest byte: the CUDA kernel's FP8 query store of its dims without
    RoPE."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Uint32(lo), loc, ip), _v(Uint32(hi), loc, ip), _v(scale, loc, ip)],
            "{\n\t.reg .f32 qa, qb, qc, qd;\n\t.reg .b16 lo, hi;\n\t"
            "shl.b32 qa, $1, 16;\n\tand.b32 qb, $1, 0xFFFF0000;\n\t"
            "shl.b32 qc, $2, 16;\n\tand.b32 qd, $2, 0xFFFF0000;\n\t"
            "mul.rn.f32 qa, qa, $3;\n\tmul.rn.f32 qb, qb, $3;\n\t"
            "mul.rn.f32 qc, qc, $3;\n\tmul.rn.f32 qd, qd, $3;\n\t"
            + "".join(_CLAMP.format(r=r) for r in ("qa", "qb", "qc", "qd"))
            + "cvt.rn.satfinite.e4m3x2.f32 lo, qb, qa;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, qd, qc;\n\tmov.b32 $0, {lo, hi};\n\t}",
            "=r,r,r,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def roped_query_e4m3x4(
    lo: Uint32,
    hi: Uint32,
    cos0: Float32,
    sin0: Float32,
    cos1: Float32,
    sin1: Float32,
    scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """``query_e4m3x4`` of two GPT-J RoPE pairs, ``lo``'s by (``cos0``,
    ``sin0``) and ``hi``'s by (``cos1``, ``sin1``), rounded as the CUDA kernel's
    ``even * cos - odd * sin`` and ``even * sin + odd * cos`` in nvcc's FMA
    contraction: fma(even, cos, -(odd * sin)) and fma(even, sin, odd * cos)."""
    pair = (
        "shl.b32 qe, ${w}, 16;\n\tand.b32 qo, ${w}, 0xFFFF0000;\n\t"
        "mul.rn.f32 qt, qo, ${s};\n\tneg.f32 qt, qt;\n\t"
        "fma.rn.f32 {e}, qe, ${c}, qt;\n\t"
        "mul.rn.f32 qt, qo, ${c};\n\tfma.rn.f32 {o}, qe, ${s}, qt;\n\t"
        "mul.rn.f32 {e}, {e}, $7;\n\tmul.rn.f32 {o}, {o}, $7;\n\t"
    )
    return Uint32(
        _asm(
            T.i32(),
            [_v(Uint32(lo), loc, ip), _v(Uint32(hi), loc, ip)]
            + [_v(v, loc, ip) for v in (cos0, sin0, cos1, sin1, scale)],
            "{\n\t.reg .f32 qe, qo, qt, qa, qb, qc, qd;\n\t.reg .b16 lo, hi;\n\t"
            + pair.format(w=1, c=3, s=4, e="qa", o="qb")
            + pair.format(w=2, c=5, s=6, e="qc", o="qd")
            + "".join(_CLAMP.format(r=r) for r in ("qa", "qb", "qc", "qd"))
            + "cvt.rn.satfinite.e4m3x2.f32 lo, qb, qa;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, qd, qc;\n\tmov.b32 $0, {lo, hi};\n\t}",
            "=r,r,r,f,f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def select_u32(cond: Int32, a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """``a`` where ``cond`` is nonzero, else ``b``."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Int32(cond), loc, ip), _v(Uint32(a), loc, ip), _v(Uint32(b), loc, ip)],
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $1, 0;\n\t"
            "selp.b32 $0, $2, $3, p;\n\t}",
            "=r,r,r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def copy16_shared_to_global(smem: Int32, gmem: Int64, *, loc=None, ip=None) -> None:
    """Copy 16 bytes from this CTA's shared memory to global memory."""
    _asm(
        None,
        [_v(Int32(smem), loc, ip), _v(Int64(gmem), loc, ip)],
        "{\n\t.reg .b32 a, b, c, d;\n\tld.shared.v4.b32 {a, b, c, d}, [$0];\n\t"
        "st.global.v4.b32 [$1], {a, b, c, d};\n\t}",
        "r,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def pack_e4m3x4(
    a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None
) -> Uint32:
    """Four floats as E4M3 bytes, ``a`` in the lowest byte (round to nearest,
    saturating)."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Float32(v), loc, ip) for v in (a, b, c, d)],
            "{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\tmov.b32 $0, {lo, hi};\n\t}",
            "=r,f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def mma_f8(
    d_tmem: Int32,
    adesc: Uint64,
    bdesc: Uint64,
    idesc: Uint32,
    accumulate: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """One tcgen05 FP8 MMA of this CTA: D (TMEM) = A x B (+ D if
    ``accumulate``), A and B from shared memory descriptors."""
    _asm(
        None,
        [
            _v(Int32(d_tmem), loc, ip),
            _v(Uint64(adesc), loc, ip),
            _v(Uint64(bdesc), loc, ip),
            _v(Uint32(idesc), loc, ip),
            _v(Int32(accumulate), loc, ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], $1, $2, $3, p;\n\t}",
        "r,l,l,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_async_peer_b32(
    smem: Int32, mbar: Int32, peer: Int32, value: Uint32, *, loc=None, ip=None
) -> None:
    """Store a word at ``smem`` of cluster CTA ``peer`` (the same offset as a
    local address), completing 4 bytes on the peer's ``mbar``."""
    _asm(
        None,
        [_v(Int32(v), loc, ip) for v in (smem, mbar, peer)]
        + [_v(Uint32(value), loc, ip)],
        "{\n\t.reg .b32 ra, rb;\n\tmapa.shared::cluster.u32 ra, $0, $2;\n\t"
        "mapa.shared::cluster.u32 rb, $1, $2;\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 "
        "[ra], $3, [rb];\n\t}",
        "r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def bulk_copy_to_peer(
    dest: Int32,
    source: Int32,
    size: Int32,
    mbar: Int32,
    peer: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Bulk-copy ``size`` bytes of this CTA's shared memory at ``source`` to
    ``dest`` of cluster CTA ``peer`` (the same offset as a local address),
    completing on the peer's ``mbar``."""
    _asm(
        None,
        [_v(Int32(v), loc, ip) for v in (dest, source, size, mbar, peer)],
        "{\n\t.reg .b32 rd, rb;\n\tmapa.shared::cluster.u32 rd, $0, $4;\n\t"
        "mapa.shared::cluster.u32 rb, $3, $4;\n\t"
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
        "[rd], [$1], $2, [rb];\n\t}",
        "r,r,r,r,r",
        True,
        loc,
        ip,
    )


class SparseMlaDecodeCute:
    """The sparse-MLA decode (module docstring) of tokens whose sparse sets
    are at most ``chunks`` x 128 columns wide; ``has_sinks``: add each head's
    attention sink to the softmax denominator; ``q_rope``: the query is BF16,
    roped and quantized as it loads (``QRope``); ``local``: the indices are
    ``LocalIndices``, translated as they load (640 columns)."""

    def __init__(
        self,
        chunks: int,
        has_sinks: bool,
        q_rope: bool = False,
        local: bool = False,
    ) -> None:
        assert 1 <= chunks <= MAX_CHUNKS
        assert not local or chunks == MAX_CHUNKS
        self.n = chunks
        self.has_sinks = has_sinks
        self.q_rope = q_rope
        self.local = local
        # Received partial blocks a CTA owns at most (block j -> CTA j % n).
        self.slots = (4 + chunks - 1) // chunks
        # Partials go out from the KV tile's upper half, part 4 + j for block j,
        # and arrive in its lower half (at most 4 parts: slots x (n - 1) <= 4).
        self.stage_part = 4
        assert self.slots * (chunks - 1) <= self.stage_part

    @cute.jit
    def __call__(
        self,
        q_ptr: Int64,
        q_stride: Int32,
        pos_ptr: Int64,
        rope_ptr: Int64,
        q_scale_ptr: Int64,
        swa_ptr: Int64,
        swa_stride: Int32,
        comp_ptr: Int64,
        comp_stride: Int32,
        idx_ptr: Int64,
        idx_stride: Int32,
        idx_cols: Int32,
        lens_ptr: Int64,
        sinks_ptr: Int64,
        out_ptr: Int64,
        out_stride: Int32,
        scale_log2: Float32,
        out_scale: Float32,
        swa_idx_ptr: Int64,
        swa_idx_stride: Int32,
        swa_block: Int32,
        swa_span: Int32,
        req_ptr: Int64,
        valid_ptr: Int64,
        table_ptr: Int64,
        table_stride: Int32,
        comp_block: Int32,
        comp_span: Int32,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on ``stream``: one cluster of ``n`` CTAs a token. Pointers
        are addresses; strides are in elements (bytes for the FP8 query and
        pools, BF16 elements for a ``q_rope`` query and ``out``); ``pos_ptr``,
        ``rope_ptr`` and ``q_scale_ptr`` are ``QRope``'s, read only with
        ``q_rope``; ``scale_log2`` is ``bmm1_scale * log2(e)``. With
        ``local``, ``idx_*`` are ``LocalIndices.topk``'s, ``lens_ptr`` is
        unused, and ``swa_idx_ptr`` .. ``comp_span`` are the other
        ``LocalIndices`` fields (unused otherwise)."""
        self.kernel(
            q_ptr,
            q_stride,
            pos_ptr,
            rope_ptr,
            q_scale_ptr,
            swa_ptr,
            swa_stride,
            comp_ptr,
            comp_stride,
            idx_ptr,
            idx_stride,
            idx_cols,
            lens_ptr,
            sinks_ptr,
            out_ptr,
            out_stride,
            scale_log2,
            out_scale,
            swa_idx_ptr,
            swa_idx_stride,
            swa_block,
            swa_span,
            req_ptr,
            valid_ptr,
            table_ptr,
            table_stride,
            comp_block,
            comp_span,
        ).launch(
            grid=(self.n, tokens, 1),
            block=(THREADS, 1, 1),
            cluster=(self.n, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        q_ptr: Int64,
        q_stride: Int32,
        pos_ptr: Int64,
        rope_ptr: Int64,
        q_scale_ptr: Int64,
        swa_ptr: Int64,
        swa_stride: Int32,
        comp_ptr: Int64,
        comp_stride: Int32,
        idx_ptr: Int64,
        idx_stride: Int32,
        idx_cols: Int32,
        lens_ptr: Int64,
        sinks_ptr: Int64,
        out_ptr: Int64,
        out_stride: Int32,
        scale_log2: Float32,
        out_scale: Float32,
        swa_idx_ptr: Int64,
        swa_idx_stride: Int32,
        swa_block: Int32,
        swa_span: Int32,
        req_ptr: Int64,
        valid_ptr: Int64,
        table_ptr: Int64,
        table_stride: Int32,
        comp_block: Int32,
        comp_span: Int32,
    ):
        n = self.n
        tid, _, _ = cute.arch.thread_idx()
        chunk, token, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        # Softmax groups: 8 threads a head, 16 rows each.
        head = tid // 8
        part = tid % 8
        neg_inf = Float32(-math.inf)

        smem = utils.SmemAllocator()
        kv = smem.allocate_array(cutlass.Uint8, 4 * BLOCK_BYTES, byte_alignment=1024)
        sq = smem.allocate_array(cutlass.Uint8, 4 * Q_BLOCK_BYTES, byte_alignment=1024)
        sp = smem.allocate_array(cutlass.Uint8, HEADS * 128, byte_alignment=1024)
        s_addr = smem.allocate_tensor(Int64, cute.make_layout(ROWS), 16)
        # Per head: chunk max, chunk sum, merge factor, sink (base 2).
        s_stat = smem.allocate_tensor(Float32, cute.make_layout((HEADS, 4)), 16)
        # Per head and chunk of the cluster: max (rows 0-15), sum (rows 16-31).
        s_ml = smem.allocate_tensor(Float32, cute.make_layout((2 * HEADS, n)), 16)
        mbars = smem.allocate_array(Int64, 3, byte_alignment=8)
        taddr = smem.allocate_array(Int32, 4, byte_alignment=16)
        if cutlass.const_expr(self.local):
            # Each warp's count of the token's valid top-k columns.
            s_count = smem.allocate_tensor(Int32, cute.make_layout(THREADS // 32), 16)
        mbar_mma = mbars
        mbar_ml = mbars + 1
        mbar_o = mbars + 2
        if tid == 0:
            cute.arch.mbarrier_init(mbar_mma, 1)
            cute.arch.mbarrier_init(mbar_ml, 1)
            cute.arch.mbarrier_init(mbar_o, 1)
            cute.arch.mbarrier_init_fence()
        # The sinks are weights: issued now, stored once the gather is issued
        # (every thread loads one of the 16, same line).
        sink_raw = Float32(0.0)
        if cutlass.const_expr(self.has_sinks):
            sink_raw = ld_global_f32(sinks_ptr + Int64((tid % HEADS) * 4))
        cute.arch.sync_threads()
        if cutlass.const_expr(n > 1):
            # Peers must see these barriers initialized before their st.async.
            cute.arch.cluster_arrive_relaxed()
        q_scale = Float32(1.0)
        if cutlass.const_expr(self.q_rope):
            # The scale is set at load time, as the sinks; the positions are a
            # step input, written before the step's graph runs.
            q_scale = ld_global_f32(q_scale_ptr)
            position = ld_global_s64(pos_ptr + Int64(token) * 8)
        # Indices, query and KV all come from earlier kernels.
        cute.arch.griddepcontrol_wait()

        # This thread's query piece: 16 dims of head q_head in each K block.
        q_head = tid // 8
        q_piece = tid % 8

        # 1. This thread's KV row: its address, or 0 when masked. The gather's
        # chain (index, then KV row) loads first.
        col = chunk * ROWS + tid
        if cutlass.const_expr(self.local):
            # The token's top-k row (this thread's column of chunks 1-4, and
            # four columns to count), its SWA slot id (chunk 0), its request
            # and whether its compressed columns count, all independently.
            topk_row = idx_ptr + Int64(token) * Int64(idx_stride) * 4
            counted = ld_global_v4(topk_row + Int64(tid * 16))
            index = ld_global_s32(
                topk_row
                + Int64(cutlass.max(cutlass.min(col - ROWS, idx_cols - 1), Int32(0)))
                * 4
            )
            swa_index = ld_global_s32(
                swa_idx_ptr + (Int64(token) * Int64(swa_idx_stride) + Int64(tid)) * 4
            )
            req = ld_global_s32(req_ptr + Int64(token) * 4)
            token_valid = ld_global_u8(valid_ptr + Int64(token))
        else:
            length = ld_global_s32(lens_ptr + Int64(token) * 4)
            index = ld_global_s32(
                idx_ptr
                + (
                    Int64(token) * Int64(idx_stride)
                    + Int64(cute.min(col, idx_cols - 1))
                )
                * 4
            )
        if cutlass.const_expr(self.q_rope):
            # wq_b's BF16 query (two 16-byte words a K block), then the cos and
            # sin of the 8 RoPE pairs of this thread's last K block (dims
            # 448 + 16 (q_piece - 4) on, for q_piece >= 4; the other threads
            # load a row's first ones and do not use them).
            q_src = (
                q_ptr
                + (
                    Int64(token) * Int64(q_stride)
                    + Int64(q_head * HEAD_DIM + q_piece * 16)
                )
                * 2
            )
            q_words = []
            for kb in cutlass.range_constexpr(4):
                for half in cutlass.range_constexpr(2):
                    q_words += ld_global_v4(q_src + Int64(kb * 256 + half * 16))
            rope_row = (
                rope_ptr + position * Int64(ROPE_ROW_BYTES) + Int64((q_piece % 4) * 32)
            )
            q_cos = ld_global_nc_v4_f32(rope_row) + ld_global_nc_v4_f32(
                rope_row + Int64(16)
            )
            q_sin = ld_global_nc_v4_f32(rope_row + Int64(128)) + ld_global_nc_v4_f32(
                rope_row + Int64(144)
            )
        # While this warp's loads are in flight.
        if warp == 0:
            cute.arch.alloc_tmem(TMEM_COLS, taddr)
            cute.arch.relinquish_tmem_alloc_permit()
        address = Int64(0)
        if cutlass.const_expr(self.local):
            # build_flashinfer_mixed_sparse_indices' table entry, in its int32
            # arithmetic: a valid top-k position's block through the request's
            # block table (issued before the count, which it does not need) ...
            comp_block_id = Int32(-1)
            if (chunk > 0) & (index >= 0):
                comp_block_id = ld_global_s32(
                    table_ptr
                    + (Int64(req) * Int64(table_stride) + Int64(index // comp_block))
                    * 4
                )
            # ... and its length: the token's valid top-k columns (none for an
            # invalid token), the columns below it active.
            warp_valid = Int32(0)
            for e in cutlass.range_constexpr(4):
                ballot = Uint32(
                    cute.arch.vote_ballot_sync((counted[e] >> Uint32(31)) == Uint32(0))
                )
                warp_valid = warp_valid + Int32(cute.arch.popc(ballot))
            if lane == 0:
                s_count[warp] = warp_valid
            cute.arch.sync_threads()
            count = Int32(0)
            if token_valid != 0:
                count = s_count[0] + s_count[1] + s_count[2] + s_count[3]
            if (chunk == 0) & (swa_index >= 0):
                swa_block_id = swa_index // swa_block
                swa_row = swa_block_id * swa_span + (
                    swa_index - swa_block_id * swa_block
                )
                address = swa_ptr + Int64(swa_row) * Int64(swa_stride)
            if (chunk > 0) & (col - ROWS < count) & (index >= 0) & (comp_block_id >= 0):
                comp_row = comp_block_id * comp_span + (
                    index - (index // comp_block) * comp_block
                )
                address = comp_ptr + Int64(comp_row) * Int64(comp_stride)
        else:
            if (col < length) & (col < idx_cols):  # noqa: SIM102
                if index >= 0:
                    if chunk == 0:
                        address = swa_ptr + Int64(index) * Int64(swa_stride)
                    else:
                        address = comp_ptr + Int64(index) * Int64(comp_stride)
        masked = address == Int64(0)
        s_addr[tid] = address
        cute.arch.sync_threads()

        # Gather in four cp.async groups, one per 128-byte K block of the rows (with
        # the query's K block): in a group a warp copies 4 rows a step, 8 lanes a
        # row, each 16-byte piece to its 128B-swizzled slot. The MMAs of a K block
        # start as soon as its group has landed, while the later groups load.
        kv_base = kv.toint()
        sq_base = sq.toint()
        sp_base = sp.toint()
        q_row = q_ptr + Int64(token) * Int64(q_stride)
        piece = lane % 8
        for kb in cutlass.range_constexpr(4):
            if cutlass.const_expr(not self.q_rope):
                cp_async_16(
                    sq_base
                    + kb * Q_BLOCK_BYTES
                    + q_head * 128
                    + ((q_piece ^ (q_head % 8)) * 16),
                    q_row + Int64(q_head * HEAD_DIM + kb * 128 + q_piece * 16),
                )
            for i in cutlass.range_constexpr(ROWS // 16):
                row = (i * 4 + warp) * 4 + lane // 8
                source = s_addr[row]
                dest = (
                    kv_base + kb * BLOCK_BYTES + row * 128 + ((piece ^ (row % 8)) * 16)
                )
                if source != 0:
                    cp_async_16(dest, source + Int64(kb * 128 + piece * 16))
                else:
                    st_shared_zero16(dest)
            cute.arch.cp_async_commit_group()
        if cutlass.const_expr(self.q_rope):
            # The query tile as the FP8 query would fill it, while the KV rows
            # load; the first K block's fence and barrier below publish it to
            # the MMAs. RoPE applies to the last K block's pieces 4-7 only.
            rope_lane = q_piece // 4
            for kb in cutlass.range_constexpr(4):
                w = q_words[kb * 8 : kb * 8 + 8]
                packed = []
                for j in cutlass.range_constexpr(4):
                    word = query_e4m3x4(w[2 * j], w[2 * j + 1], q_scale)
                    if cutlass.const_expr(kb == 3):
                        roped = roped_query_e4m3x4(
                            w[2 * j],
                            w[2 * j + 1],
                            q_cos[2 * j],
                            q_sin[2 * j],
                            q_cos[2 * j + 1],
                            q_sin[2 * j + 1],
                            q_scale,
                        )
                        word = select_u32(rope_lane, roped, word)
                    packed.append(word)
                st_shared_v4(
                    sq_base
                    + kb * Q_BLOCK_BYTES
                    + q_head * 128
                    + ((q_piece ^ (q_head % 8)) * 16),
                    *packed,
                )

        # 2. S^T = KV x Q^T, 4 MMAs of K = 32 a K block.
        tmem = cute.make_tensor(taddr, cute.make_layout(1))[0]
        for kb in cutlass.range_constexpr(4):
            cute.arch.cp_async_wait_group(3 - kb)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if warp == 0:
                _tcgen05.fence_after_thread_sync()
                for k4 in cutlass.range_constexpr(4):
                    adesc = Uint64(SDESC_K) | Uint64(
                        (kv_base + kb * BLOCK_BYTES + k4 * 32) >> 4
                    )
                    bdesc = Uint64(SDESC_K) | Uint64(
                        (sq_base + kb * Q_BLOCK_BYTES + k4 * 32) >> 4
                    )
                    with cute.arch.elect_one():
                        mma_f8(
                            tmem,
                            adesc,
                            bdesc,
                            Uint32(IDESC_QK),
                            Int32(1 if kb + k4 > 0 else 0),
                        )
                if cutlass.const_expr(kb == 3):
                    _tcgen05.commit(mbar_mma)
                    with cute.arch.elect_one():
                        cute.arch.griddepcontrol_launch_dependents()
        # The sink, which the softmax's barrier publishes to the merge (step 5).
        if tid < HEADS:
            sink = neg_inf
            if cutlass.const_expr(self.has_sinks):
                sink = sink_raw * Float32(LOG2E)
            s_stat[tid, 3] = sink
        cute.arch.mbarrier_wait(mbar_mma, 0)
        _tcgen05.fence_after_thread_sync()

        # 3. Chunk softmax. Each thread's TMEM lane is one KV row; the scores are
        # transposed through the (now free) query tile so that 8 threads reduce
        # one head's 128 rows.
        s = cute.make_rmem_tensor((HEADS,), Float32)
        s.store(_tcgen05.ld(warp * 32, tmem, "32x32b", HEADS))
        _tcgen05.wait_ld()
        s_t = cute.make_tensor(
            cute.recast_ptr(sq, dtype=Float32),
            cute.make_layout((HEADS, ROWS), stride=(ROWS, 1)),
        )
        for h in cutlass.range_constexpr(HEADS):
            value = s[h] * scale_log2
            if masked:
                value = neg_inf
            s_t[h, tid] = value
        cute.arch.sync_threads()
        vals = cute.make_rmem_tensor((16,), Float32)
        for i in cutlass.range_constexpr(16):
            vals[i] = s_t[head, part * 16 + i]
        peak = vals[0]
        for i in cutlass.range_constexpr(1, 16):
            peak = cute.arch.fmax(peak, vals[i])
        peak = cute.arch.warp_reduction_max(peak, threads_in_group=8)
        base = peak
        if peak == neg_inf:
            base = Float32(0.0)
        total = Float32(0.0)
        for i in cutlass.range_constexpr(16):
            vals[i] = cute.math.exp2(vals[i] - base, fastmath=True)
            total = total + vals[i]
        total = cute.arch.warp_reduction_sum(total, threads_in_group=8)
        # P^T as the K-major B operand: head h's 128 rows are one 128-byte row,
        # this thread's 16 at its swizzled 16-byte slot.
        scale_p = Float32(P_SCALE)
        st_shared_v4(
            sp_base + head * 128 + ((part ^ (head % 8)) * 16),
            pack_e4m3x4(
                vals[0] * scale_p,
                vals[1] * scale_p,
                vals[2] * scale_p,
                vals[3] * scale_p,
            ),
            pack_e4m3x4(
                vals[4] * scale_p,
                vals[5] * scale_p,
                vals[6] * scale_p,
                vals[7] * scale_p,
            ),
            pack_e4m3x4(
                vals[8] * scale_p,
                vals[9] * scale_p,
                vals[10] * scale_p,
                vals[11] * scale_p,
            ),
            pack_e4m3x4(
                vals[12] * scale_p,
                vals[13] * scale_p,
                vals[14] * scale_p,
                vals[15] * scale_p,
            ),
        )
        if part == 0:
            s_stat[head, 0] = peak
            s_stat[head, 1] = total
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        # 4. O^T = V^T x P^T, four MMAs of 128 head dims (K = 128 rows each);
        # the statistics exchange and the merge factors overlap them.
        if warp == 0:
            _tcgen05.fence_after_thread_sync()
            for j in cutlass.range_constexpr(4):
                for kk in cutlass.range_constexpr(ROWS // 32):
                    adesc = Uint64(SDESC_MN) | Uint64(
                        (kv_base + j * BLOCK_BYTES + kk * 32 * 128) >> 4
                    )
                    bdesc = Uint64(SDESC_K) | Uint64((sp_base + kk * 32) >> 4)
                    with cute.arch.elect_one():
                        mma_f8(
                            tmem + HEADS + HEADS * j,
                            adesc,
                            bdesc,
                            Uint32(IDESC_PV),
                            Int32(1 if kk > 0 else 0),
                        )
            _tcgen05.commit(mbar_mma)
        if cutlass.const_expr(n > 1):
            mbar_ml_addr = mbar_ml.toint()
            s_ml_base = s_ml.iterator.toint()
            cute.arch.cluster_wait()
            if tid == 0:
                mbar_ops.arrive_expect_tx(mbar_ml, n * 2 * HEADS * 4)
            if tid < 2 * HEADS:
                stat = s_stat[tid % HEADS, tid // HEADS]
                for peer in cutlass.range_constexpr(n):
                    st_async_peer_b32(
                        s_ml_base + (chunk * 2 * HEADS + tid) * 4,
                        mbar_ml_addr,
                        Int32(peer),
                        recast_val(stat, Uint32),
                    )
            cute.arch.mbarrier_wait(mbar_ml, 0)
        # 5. Merge factors, one thread a head: exp2(m_c - M) / L, L with the sink.
        if tid < HEADS:
            own = s_stat[tid, 0]
            sink = s_stat[tid, 3]
            top = sink
            denom = Float32(0.0)
            if cutlass.const_expr(n > 1):
                for c in cutlass.range_constexpr(n):
                    top = cute.arch.fmax(top, s_ml[tid, c])
                for c in cutlass.range_constexpr(n):
                    peak_c = s_ml[tid, c]
                    if peak_c != neg_inf:
                        denom = (
                            denom
                            + cute.math.exp2(peak_c - top, fastmath=True)
                            * s_ml[HEADS + tid, c]
                        )
            else:
                top = cute.arch.fmax(top, own)
                if own != neg_inf:
                    denom = cute.math.exp2(own - top, fastmath=True) * s_stat[tid, 1]
            if cutlass.const_expr(self.has_sinks):
                denom = denom + cute.math.exp2(sink - top, fastmath=True)
            f = Float32(0.0)
            if own != neg_inf:  # noqa: SIM102
                if denom > Float32(0.0):
                    f = (
                        cute.math.exp2(own - top, fastmath=True)
                        / denom
                        * out_scale
                        * Float32(1.0 / P_SCALE)
                    )
            s_stat[tid, 2] = f
        cute.arch.mbarrier_wait(mbar_mma, 1)
        _tcgen05.fence_after_thread_sync()
        cute.arch.sync_threads()
        if cutlass.const_expr(n > 1):
            # This CTA's MMAs are done: peers may now write into its tile.
            cute.arch.cluster_arrive()
        o = cute.make_rmem_tensor((4 * HEADS,), Float32)
        o.store(_tcgen05.ld(warp * 32, tmem + HEADS, "32x32b", 4 * HEADS))
        _tcgen05.wait_ld()
        factor = cute.make_rmem_tensor((HEADS,), Float32)
        for h in cutlass.range_constexpr(HEADS):
            factor[h] = s_stat[h, 2]
        for j in cutlass.range_constexpr(4):
            for h in cutlass.range_constexpr(HEADS):
                o[j * HEADS + h] = o[j * HEADS + h] * factor[h]
        # Thread tid holds head dim (128 j + tid) of every head.
        out_row = out_ptr + Int64(token) * Int64(out_stride) * 2
        if cutlass.const_expr(n == 1):
            # Stage [block][head][dim] BF16 in the tile, then 16-byte stores.
            for j in cutlass.range_constexpr(4):
                for h in cutlass.range_constexpr(HEADS):
                    st_shared_bf16(
                        kv_base + ((j * HEADS + h) * 128 + tid) * 2, o[j * HEADS + h]
                    )
            cute.arch.sync_threads()
            for k in cutlass.range_constexpr(4 * HEADS * 16 // THREADS):
                piece = k * THREADS + tid
                rowid = piece // 16
                d16 = piece % 16
                blk = rowid // HEADS
                h = rowid % HEADS
                copy16_shared_to_global(
                    kv_base + piece * 16,
                    out_row + Int64((h * HEAD_DIM + blk * 128 + d16 * 8) * 2),
                )
        else:
            mbar_o_addr = mbar_o.toint()
            owned = Int32(0)
            for j in cutlass.range_constexpr(4):
                if chunk == j % n:
                    owned = owned + 1
            # The tile as 8 FP32 parts of [head][row]: head-major, so a warp's
            # 32 rows of one head are 32 consecutive words (no bank conflicts
            # on either side).
            parts = cute.make_tensor(
                cute.recast_ptr(kv, dtype=Float32),
                cute.make_layout((HEADS, ROWS, 8), stride=(ROWS, 1, ROWS * HEADS)),
            )
            for j in cutlass.range_constexpr(4):
                if chunk != j % n:
                    for h in cutlass.range_constexpr(HEADS):
                        parts[h, tid, self.stage_part + j] = o[j * HEADS + h]
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if tid == 0:  # noqa: SIM102
                if owned > 0:
                    mbar_ops.arrive_expect_tx(mbar_o, owned * (n - 1) * PART_BYTES)
            # Every peer's MMAs are done, so its tile's lower half can take
            # partials: sender s of block j lands in part (j // n) (n - 1) + s,
            # the senders numbered without the owner.
            cute.arch.cluster_wait()
            if tid == 0:
                for j in cutlass.range_constexpr(4):
                    owner = j % n
                    if chunk != owner:
                        sender = chunk
                        if chunk > owner:
                            sender = chunk - 1
                        bulk_copy_to_peer(
                            kv_base + ((j // n) * (n - 1) + sender) * PART_BYTES,
                            kv_base + (self.stage_part + j) * PART_BYTES,
                            Int32(PART_BYTES),
                            mbar_o_addr,
                            Int32(owner),
                        )
            if owned > 0:
                cute.arch.mbarrier_wait(mbar_o, 0)
                # Owned block j (slot j // n): the sum, BF16 [slot][head][dim]
                # in the query tile, then 16-byte stores.
                for j in cutlass.range_constexpr(4):
                    if chunk == j % n:
                        for h in cutlass.range_constexpr(HEADS):
                            acc = o[j * HEADS + h]
                            for c in cutlass.range_constexpr(n - 1):
                                acc = acc + parts[h, tid, (j // n) * (n - 1) + c]
                            st_shared_bf16(
                                sq_base + (((j // n) * HEADS + h) * 128 + tid) * 2, acc
                            )
                cute.arch.sync_threads()
                for slot in cutlass.range_constexpr(self.slots):
                    if chunk + slot * n < 4:
                        for k in cutlass.range_constexpr(HEADS * 16 // THREADS):
                            piece = k * THREADS + tid
                            h = piece // 16
                            d16 = piece % 16
                            copy16_shared_to_global(
                                sq_base + (slot * HEADS * 128) * 2 + piece * 16,
                                out_row
                                + Int64(
                                    (h * HEAD_DIM + (chunk + slot * n) * 128 + d16 * 8)
                                    * 2
                                ),
                            )
            # The bulk copies read this CTA's staged partials asynchronously:
            # stay resident until every owner has received (its arrive follows
            # its mbarrier wait).
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        _tcgen05.fence_before_thread_sync()
        cute.arch.sync_threads()
        if warp == 0:
            _tcgen05.fence_after_thread_sync()
            cute.arch.dealloc_tmem(
                cute.make_ptr(Float32, tmem, cute.AddressSpace.tmem), TMEM_COLS
            )


class SparseMlaPrefetchCute:
    """Prefetch into L2 what ``SparseMlaDecodeCute`` of ``chunks`` chunks reads
    first from an already built sparse-index table: the table's rows (by
    loading them, as the attention does), each active column's KV row, the
    sinks and the query scale. One CTA of 128 threads a token and chunk, one
    column a thread, as in the attention, whose masking it copies: it addresses
    exactly the rows the attention then gathers. It writes nothing."""

    def __init__(self, chunks: int) -> None:
        assert 1 <= chunks <= MAX_CHUNKS
        self.n = chunks

    @cute.jit
    def __call__(
        self,
        swa_ptr: Int64,
        swa_stride: Int32,
        comp_ptr: Int64,
        comp_stride: Int32,
        idx_ptr: Int64,
        idx_stride: Int32,
        idx_cols: Int32,
        lens_ptr: Int64,
        sinks_ptr: Int64,
        q_scale_ptr: Int64,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on ``stream`` (no programmatic dependency: the table is read
        after every earlier kernel of the stream finished); ``sinks_ptr`` and
        ``q_scale_ptr`` may be 0."""
        self.kernel(
            swa_ptr,
            swa_stride,
            comp_ptr,
            comp_stride,
            idx_ptr,
            idx_stride,
            idx_cols,
            lens_ptr,
            sinks_ptr,
            q_scale_ptr,
        ).launch(grid=(self.n, tokens, 1), block=(THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        swa_ptr: Int64,
        swa_stride: Int32,
        comp_ptr: Int64,
        comp_stride: Int32,
        idx_ptr: Int64,
        idx_stride: Int32,
        idx_cols: Int32,
        lens_ptr: Int64,
        sinks_ptr: Int64,
        q_scale_ptr: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        chunk, token, _ = cute.arch.block_idx()
        col = chunk * ROWS + tid
        length = ld_global_s32(lens_ptr + Int64(token) * 4)
        index = ld_global_s32(
            idx_ptr
            + (Int64(token) * Int64(idx_stride) + Int64(cute.min(col, idx_cols - 1)))
            * 4
        )
        address = Int64(0)
        if (col < length) & (col < idx_cols):  # noqa: SIM102
            if index >= 0:
                if chunk == 0:
                    address = swa_ptr + Int64(index) * Int64(swa_stride)
                else:
                    address = comp_ptr + Int64(index) * Int64(comp_stride)
        if address != Int64(0):
            prefetch_l2_row(address)
        if (chunk == 0) & (token == 0) & (tid == 0):
            if sinks_ptr != Int64(0):
                prefetch_l2_line(sinks_ptr)
            if q_scale_ptr != Int64(0):
                prefetch_l2_line(q_scale_ptr)


@cache
def _prefetch_kernel(device_index: int, chunks: int):
    """``SparseMlaPrefetchCute`` compiled for a device and chunk count."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            SparseMlaPrefetchCute(chunks),
            Int64(0),
            Int32(0),
            Int64(0),
            Int32(0),
            Int64(0),
            Int32(0),
            Int32(1),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _kernel(
    device_index: int,
    chunks: int,
    has_sinks: bool,
    q_rope: bool,
    local: bool,
):
    """``SparseMlaDecodeCute`` compiled for a device, chunk count, sinks, query
    RoPE and local indices (all positional: ``cache`` keys on the arguments as
    passed)."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            SparseMlaDecodeCute(chunks, has_sinks, q_rope, local),
            Int64(0),
            Int32(0),
            Int64(0),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(0),
            Int64(0),
            Int32(0),
            Int64(0),
            Int32(0),
            Int32(1),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(0),
            Float32(0),
            Float32(0),
            Int64(0),
            Int32(0),
            Int32(1),
            Int32(1),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(0),
            Int32(1),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _max_tokens(device_index: int, chunks: int) -> int:
    """Tokens whose clusters are all resident at once: ``CTAS_PER_SM`` times
    the resident clusters of ``chunks`` CTAs at one CTA an SM."""
    with torch.accelerator.device_index(device_index):
        clusters = cutlass.utils.HardwareInfo(device_index).get_max_active_clusters(
            chunks
        )
    return CTAS_PER_SM * clusters


def precompile(device_index: int, has_sinks: bool) -> None:
    """Compile the kernels of DeepSeek-V4's two widths (128 sliding-window
    columns alone, and 640) and count the resident clusters, outside any CUDA
    graph capture; other widths compile at their first launch.

    Args:
        device_index: The CUDA device to compile for.
        has_sinks: Whether the layers add attention sinks.

    """
    for chunks in (1, MAX_CHUNKS):
        for local in (False, True) if chunks == MAX_CHUNKS else (False,):
            for q_rope in (False, True):
                _kernel(device_index, chunks, has_sinks, q_rope, local)
        _prefetch_kernel(device_index, chunks)
        _max_tokens(device_index, chunks)


def _is_pool(cache: torch.Tensor) -> bool:
    """An FP8 KV pool of 512-byte rows ``stride(-2)`` apart, 16-byte aligned."""
    return (
        cache.dtype == torch.float8_e4m3fn
        and cache.dim() >= 2
        and cache.shape[-1] == HEAD_DIM
        and cache.stride(-1) == 1
        and cache.stride(-2) % 16 == 0
        and cache.data_ptr() % 16 == 0
    )


def _takes_q_rope(query: torch.Tensor, q_rope: QRope) -> bool:
    """A BF16 query of 16-byte aligned rows, int64 positions for its tokens,
    a contiguous FP32 [*, 64] cos || sin table and one FP32 scale."""
    positions, cos_sin_cache, scale_inv = q_rope
    return (
        query.dtype == torch.bfloat16
        and query.stride(0) % 8 == 0
        and positions.dtype == torch.int64
        and positions.dim() == 1
        and positions.stride(0) == 1
        and positions.numel() >= query.shape[0]
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.shape[1] == 64
        and cos_sin_cache.is_contiguous()
        and cos_sin_cache.data_ptr() % 16 == 0
        and scale_inv.dtype == torch.float32
        and scale_inv.numel() == 1
    )


def _takes_launch(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    out: torch.Tensor,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: torch.Tensor | None,
    q_rope: QRope | None,
    chunks: int,
) -> bool:
    """The checks of `supports_sparse_mla_cute` other than the indices'."""
    if (
        query.dim() != 3
        or query.shape[1:] != (HEADS, HEAD_DIM)
        or query.stride(2) != 1
        or query.stride(1) != HEAD_DIM
        or query.data_ptr() % 16 != 0
    ):
        return False
    if q_rope is None:
        if query.dtype != torch.float8_e4m3fn or query.stride(0) % 16 != 0:
            return False
    elif not _takes_q_rope(query, q_rope):
        return False
    tokens = query.shape[0]
    if (
        out.dtype != torch.bfloat16
        or out.shape != query.shape
        or out.stride(2) != 1
        or out.stride(1) != HEAD_DIM
        or out.stride(0) % 8 != 0
        or out.data_ptr() % 16 != 0
    ):
        return False
    if sinks is not None and (
        sinks.dtype != torch.float32
        or not sinks.is_contiguous()
        or sinks.numel() < HEADS
    ):
        return False
    if not (
        isinstance(bmm1_scale, float | int) and isinstance(bmm2_scale, float | int)
    ):
        return False
    if not (_is_pool(swa_kv_cache) and _is_pool(compressed_kv_cache)):
        return False
    return 0 < tokens <= _max_tokens(query.device.index, chunks)


def supports_sparse_mla_cute(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    out: torch.Tensor,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: torch.Tensor | None,
    q_rope: QRope | None = None,
) -> bool:
    """Whether ``sparse_mla_decode`` takes these arguments: 16 FP8 query heads
    of 512 (BF16 and unroped with ``q_rope``; BF16 output), FP8 pools, int32
    index rows of at most 640 columns, scalar scales, FP32 sinks, and a batch
    whose clusters are all resident."""
    tokens = query.shape[0]
    if (
        sparse_indices.dtype != torch.int32
        or sparse_indices.dim() != 2
        or sparse_indices.shape[0] != tokens
        or sparse_indices.stride(1) != 1
        or not 0 < sparse_indices.shape[1] <= MAX_CHUNKS * ROWS
    ):
        return False
    if (
        sparse_topk_lens.dtype != torch.int32
        or not sparse_topk_lens.is_contiguous()
        or sparse_topk_lens.numel() != tokens
    ):
        return False
    chunks = (sparse_indices.shape[1] + ROWS - 1) // ROWS
    return _takes_launch(
        query,
        swa_kv_cache,
        compressed_kv_cache,
        out,
        bmm1_scale,
        bmm2_scale,
        sinks,
        q_rope,
        chunks,
    )


def supports_local_indices(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    local: LocalIndices,
    out: torch.Tensor,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: torch.Tensor | None,
    q_rope: QRope | None = None,
) -> bool:
    """Whether ``sparse_mla_decode_local`` takes these arguments: those of
    `supports_sparse_mla_cute` at 640 columns, with ``LocalIndices`` of 512
    16-byte aligned int32 top-k columns, 128 int32 SWA columns, int32 requests,
    bool token flags, an int32 block table and positive block sizes."""
    tokens = query.shape[0]
    topk, swa = local.topk, local.swa
    if (
        topk.dtype != torch.int32
        or topk.shape != (tokens, LOCAL_TOPK)
        or topk.stride(1) != 1
        or topk.stride(0) % 4 != 0
        or topk.data_ptr() % 16 != 0
    ):
        return False
    if (
        swa.dtype != torch.int32
        or swa.shape != (tokens, SWA_COLUMNS)
        or swa.stride(1) != 1
    ):
        return False
    for flags, dtype in (
        (local.token_to_req, torch.int32),
        (local.token_valid, torch.bool),
    ):
        if (
            flags.dtype != dtype
            or flags.dim() != 1
            or flags.stride(0) != 1
            or flags.numel() < tokens
        ):
            return False
    table = local.block_table
    if table.dtype != torch.int32 or table.dim() != 2 or table.stride(1) != 1:
        return False
    if (
        min(
            local.swa_block_size,
            local.swa_block_span,
            local.block_size,
            local.block_span,
        )
        <= 0
    ):
        return False
    return _takes_launch(
        query,
        swa_kv_cache,
        compressed_kv_cache,
        out,
        bmm1_scale,
        bmm2_scale,
        sinks,
        q_rope,
        MAX_CHUNKS,
    )


def _launch(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor | None,
    local: LocalIndices | None,
    out: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    sinks: torch.Tensor | None,
    q_rope: QRope | None,
) -> None:
    """Launch `SparseMlaDecodeCute` on ``indices`` (with ``local``, its top-k)."""
    tokens = query.shape[0]
    width = indices.shape[1]
    chunks = MAX_CHUNKS if local is not None else (width + ROWS - 1) // ROWS
    _kernel(
        query.device.index,
        chunks,
        sinks is not None,
        q_rope is not None,
        local is not None,
    )(
        Int64(query.data_ptr()),
        Int32(query.stride(0)),
        Int64(q_rope.positions.data_ptr() if q_rope is not None else 0),
        Int64(q_rope.cos_sin_cache.data_ptr() if q_rope is not None else 0),
        Int64(q_rope.scale_inv.data_ptr() if q_rope is not None else 0),
        Int64(swa_kv_cache.data_ptr()),
        Int32(swa_kv_cache.stride(-2)),
        Int64(compressed_kv_cache.data_ptr()),
        Int32(compressed_kv_cache.stride(-2)),
        Int64(indices.data_ptr()),
        Int32(indices.stride(0)),
        Int32(width),
        Int64(lens.data_ptr() if lens is not None else 0),
        Int64(sinks.data_ptr() if sinks is not None else 0),
        Int64(out.data_ptr()),
        Int32(out.stride(0)),
        Float32(bmm1_scale * LOG2E),
        Float32(bmm2_scale),
        Int64(local.swa.data_ptr() if local is not None else 0),
        Int32(local.swa.stride(0) if local is not None else 0),
        Int32(local.swa_block_size if local is not None else 1),
        Int32(local.swa_block_span if local is not None else 1),
        Int64(local.token_to_req.data_ptr() if local is not None else 0),
        Int64(local.token_valid.data_ptr() if local is not None else 0),
        Int64(local.block_table.data_ptr() if local is not None else 0),
        Int32(local.block_table.stride(0) if local is not None else 0),
        Int32(local.block_size if local is not None else 1),
        Int32(local.block_span if local is not None else 1),
        Int32(tokens),
        current_cu_stream(),
    )


def sparse_mla_decode(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    out: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    sinks: torch.Tensor | None,
    q_rope: QRope | None = None,
) -> None:
    """Launch `SparseMlaDecodeCute` on the current stream (see
    `supports_sparse_mla_cute` for the accepted layouts).

    Args:
        query: [tokens, 16, 512] FP8 E4M3, or with ``q_rope`` BF16 before its
            RoPE and quantization.
        swa_kv_cache: FP8 pool of 512-byte rows ``stride(-2)`` apart; the first
            128 columns of ``sparse_indices`` are its row offsets.
        compressed_kv_cache: The pool of the other columns, likewise.
        sparse_indices: [tokens, width] int32 row offsets, -1 for none
            (``width`` <= 640).
        sparse_topk_lens: [tokens] int32 active columns of each token.
        out: [tokens, 16, 512] BF16 attention output.
        bmm1_scale: Score scale (the softmax scale times the FP8 scales).
        bmm2_scale: Output scale.
        sinks: [16] FP32 attention sinks (natural log), or None.
        q_rope: The RoPE and FP8 scale the kernel applies to a BF16 query.

    """
    assert supports_sparse_mla_cute(
        query,
        swa_kv_cache,
        compressed_kv_cache,
        sparse_indices,
        sparse_topk_lens,
        out,
        bmm1_scale,
        bmm2_scale,
        sinks,
        q_rope,
    )
    _launch(
        query,
        swa_kv_cache,
        compressed_kv_cache,
        sparse_indices,
        sparse_topk_lens,
        None,
        out,
        bmm1_scale,
        bmm2_scale,
        sinks,
        q_rope,
    )


def sparse_mla_decode_local(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    local: LocalIndices,
    out: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    sinks: torch.Tensor | None,
    q_rope: QRope | None = None,
) -> None:
    """`sparse_mla_decode` on the table ``build_flashinfer_mixed_sparse_indices``
    would build from ``local`` (decode tokens, request-local top-k), without
    building it: the kernel gathers the same KV rows under the same mask (see
    `supports_local_indices` for the accepted layouts)."""
    assert supports_local_indices(
        query,
        swa_kv_cache,
        compressed_kv_cache,
        local,
        out,
        bmm1_scale,
        bmm2_scale,
        sinks,
        q_rope,
    )
    _launch(
        query,
        swa_kv_cache,
        compressed_kv_cache,
        local.topk,
        None,
        local,
        out,
        bmm1_scale,
        bmm2_scale,
        sinks,
        q_rope,
    )


def supports_sparse_mla_prefetch(
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
) -> bool:
    """Whether ``sparse_mla_prefetch`` takes these arguments: the pools and
    table ``sparse_mla_decode`` takes, for a batch it takes."""
    if (
        sparse_indices.dtype != torch.int32
        or sparse_indices.dim() != 2
        or sparse_indices.stride(1) != 1
        or not 0 < sparse_indices.shape[1] <= MAX_CHUNKS * ROWS
    ):
        return False
    tokens = sparse_indices.shape[0]
    if (
        sparse_topk_lens.dtype != torch.int32
        or not sparse_topk_lens.is_contiguous()
        or sparse_topk_lens.numel() != tokens
    ):
        return False
    if not (_is_pool(swa_kv_cache) and _is_pool(compressed_kv_cache)):
        return False
    chunks = (sparse_indices.shape[1] + ROWS - 1) // ROWS
    return 0 < tokens <= _max_tokens(sparse_indices.device.index, chunks)


def sparse_mla_prefetch(
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    sinks: torch.Tensor | None,
    q_scale: torch.Tensor | None,
) -> None:
    """Launch `SparseMlaPrefetchCute` on the current stream for the table that
    a later `sparse_mla_decode` of the same pools reads (see
    `supports_sparse_mla_prefetch`); ``sinks`` and ``q_scale`` are that
    decode's sinks and ``QRope.scale_inv``, or None."""
    assert supports_sparse_mla_prefetch(
        swa_kv_cache, compressed_kv_cache, sparse_indices, sparse_topk_lens
    )
    width = sparse_indices.shape[1]
    chunks = (width + ROWS - 1) // ROWS
    _prefetch_kernel(sparse_indices.device.index, chunks)(
        Int64(swa_kv_cache.data_ptr()),
        Int32(swa_kv_cache.stride(-2)),
        Int64(compressed_kv_cache.data_ptr()),
        Int32(compressed_kv_cache.stride(-2)),
        Int64(sparse_indices.data_ptr()),
        Int32(sparse_indices.stride(0)),
        Int32(width),
        Int64(sparse_topk_lens.data_ptr()),
        Int64(sinks.data_ptr() if sinks is not None else 0),
        Int64(q_scale.data_ptr() if q_scale is not None else 0),
        Int32(sparse_indices.shape[0]),
        current_cu_stream(),
    )
