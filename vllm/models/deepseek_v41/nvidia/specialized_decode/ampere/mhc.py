# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mHC layer boundary of DeepSeek-V4.1 on SM80, in CuTe DSL.

DeepSeek-V4.1 carries `hc = 4` residual streams per token (manifold
hyper-connections, mHC). Every sublayer boundary does two things with them:

1. **Post and collapse** (`reference.mhc_post_collapse`). The sublayer
   output `x` enters the streams through the previous boundary's
   coefficients, `r'[j] = post[j] x + sum_k comb[k, j] r[k]` (rounded to
   BF16, the stored residual), and the next sublayer's input is the RMSNorm
   of the streams collapsed with the previous pre-mix,
   `norm(sum_j pre[j] r'[j])`. The collapse uses the *carried* pre-mix, so
   this step does not wait for this boundary's own coefficients.
2. **Coefficients** (`reference.mhc_coefficients`). A projection of the
   flattened streams `r'` (`4 x hidden` wide) onto 24 mixes and their
   RMS, then sigmoids for the pre- and post-mixes and a 20-step Sinkhorn
   normalization for the 4 x 4 `comb` matrix. They are used only at the
   next boundary.

The stock path runs both in two TileLang kernels with one CTA per token for
the coefficient math and the collapse; at a handful of decode rows they are
latency-bound (about 14 us per boundary at six rows on A100). Here each step
is its own kernel:

* `PostCollapse` -- one CTA per row, one 16-byte vector of every stream per
  thread: the post-mix, the stored residual, the collapse and its RMSNorm
  (one block reduction), and optionally the stream mean that draft models
  read.
* `Coefficients` -- `hidden / 128` CTAs, each owning 128 hidden columns
  of all four streams for every row, taken in groups of 16 rows. A CTA keeps
  the slice of the projection weights in registers, recomputes `r'` for
  its slice (so it does not wait for `PostCollapse`) and writes per-row
  partial dot products. The last CTA to finish a row group (an atomic ticket
  per group) sums the group's partials in CTA order and runs the Sinkhorn
  normalization, one 16-lane group per row; the CTAs start at different
  groups, so that different CTAs finish them. The ticket counters reset
  themselves, so the kernel needs no initialization per launch and is
  CUDA-graph safe.
* `CoefficientsMma` -- the same work on tensor cores, taken on SM80 for
  the four streams when the hidden size is a multiple of 320 (SM90, whose
  small decode batches run these kernels too, keeps `Coefficients`, which
  is faster there at six rows): `hidden / 320` CTAs,
  each staging `r'` of its 320 columns for a row group in shared memory as
  BF16, then multiplying it by the projection with `mma.sync` m16n8k16
  (each of eight warps one stream's half of the slice, all 24 mixes). The
  FP32 projection is split into two BF16 terms, `hi = bf16(w)` and
  `lo = bf16(w - hi)`, multiplied separately and summed in FP32. The sums
  of squares stay on CUDA cores; the partials, tickets and Sinkhorn finish
  are those of `Coefficients`.

Because the coefficients are needed only at the next boundary, the model
runs them on a side stream inside FULL CUDA graphs (see `mhc_post_pre`):
the critical path keeps only `PostCollapse`. At DeepSeek-V4.1's hidden
size of 5120 the side kernel holds 16 SMs (`CoefficientsMma`) or 40
(`Coefficients`) at any batch. With a CTA per slice and row group it took
120 CTAs at 48 rows, more than A100's 108 SMs, and `PostCollapse`, whose
640-thread CTAs do not fit on an SM beside one of them, waited about 14 us
for SMs at every boundary. Fewer side CTAs leave more SMs to the sublayers
that run beside them.

Numerics: the residual and the collapse round exactly as the stock kernels
do; the projection reads the BF16 residual (as the PyTorch reference and
the stock large-batch path do; the stock fused small-batch kernel projects
the unrounded FP32 values instead, a difference within BF16 rounding of the
projection's input), and the sums run in a different order, so results
differ from the stock kernels by FP32 rounding of the reductions. The
coefficients stay within about 1e-6 of an FP64 reference. The two BF16
terms of `CoefficientsMma` keep about 16 significant bits of each
projection weight, more than the TF32 projection of the stock SM90 kernel;
their effect on the coefficients is bounded in the tests.

The ticket counters are shared per device, so launches of `Coefficients`
must be ordered on one stream at a time, which the model's boundaries are.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream

from . import is_ampere
from .codecs import f32_from_bits, store_u32

# Residual streams and the projection's mixes: pre (4), post (4), comb (16).
HC = 4
MIXES = HC * (HC + 2)
# A row's statistics: the 24 dot products and the sum of squares.
STATS = MIXES + 1
# BF16 values per 16-byte vector; PostCollapse has one per thread.
VEC = 8
# Coefficients: hidden columns per CTA, rows per CTA, threads, and the three
# mixes each of its eight warps projects.
SLICE = 128
ROW_GROUP = 16
COEF_THREADS = 256
COEF_WARPS = COEF_THREADS // 32
MIXES_PER_WARP = MIXES // COEF_WARPS
# The most rows these kernels take; larger batches keep the stock kernels.
MAX_ROWS = 256
MAX_ROW_GROUPS = MAX_ROWS // ROW_GROUP
# CoefficientsMma: hidden columns per CTA; the projection width each of its
# eight warps multiplies (a stream's half of the slice) in m16n8k16 tiles,
# and the mixes in tiles of eight.
MMA_SLICE = 320
MMA_WARP_K = HC * MMA_SLICE // COEF_WARPS
MMA_K_TILES = MMA_WARP_K // 16
MMA_N_TILES = MIXES // 8
# Words per staged row (the slice's four streams as BF16 pairs), padded by
# four words so that a warp's fragment loads fall on 32 different banks.
MMA_ROW_WORDS = HC * MMA_SLICE // 2 + 4
# Staging threads per row, and the four-column quads each of them stages.
MMA_ROW_THREADS = COEF_THREADS // ROW_GROUP
MMA_QUADS = MMA_SLICE // 4 // MMA_ROW_THREADS


def _asm(result, operands, text, constraints, side_effects, loc, ip):
    """One inline PTX statement."""
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


@dsl_user_op
def load_words(address: Int64, count: int, *, loc=None, ip=None) -> list:
    """`count` (2 or 4) consecutive 32-bit words in one vector load."""
    registers = ", ".join(f"${i}" for i in range(count))
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * count),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        f"ld.global.v{count}.u32 {{{registers}}}, [${count}];",
        ",".join(["=r"] * count + ["l"]),
        False,
        loc,
        ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), loaded, [i], loc=loc, ip=ip))
        for i in range(count)
    ]


@dsl_user_op
def load_weights4(address: Int64, *, loc=None, ip=None) -> list:
    """Four FP32 weights (read-only for the kernel) in one vector load."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        False,
        loc,
        ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), loaded, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def load_weights2(address: Int64, *, loc=None, ip=None) -> list:
    """Two FP32 weights (read-only for the kernel) in one vector load."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v2.f32 {$0, $1}, [$2];",
        "=f,=f,l",
        False,
        loc,
        ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), loaded, [i], loc=loc, ip=ip))
        for i in range(2)
    ]


@dsl_user_op
def mma_bf16(c: list, a: list, b: list, *, loc=None, ip=None) -> list:
    """`c + a b` for one m16n8k16 BF16 tile with FP32 accumulation, given
    this thread's fragments: four words of `a`, two of `b` and four values
    of `c` (the PTX `mma.sync` layouts)."""
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [
            *(Uint32(word).ir_value(loc=loc, ip=ip) for word in a),
            *(Uint32(word).ir_value(loc=loc, ip=ip) for word in b),
            *(Float32(value).ir_value(loc=loc, ip=ip) for value in c),
        ],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        ",".join(["=f"] * 4 + ["r"] * 6 + ["f"] * 4),
        False,
        loc,
        ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def load_f32x4(address: Int64, *, loc=None, ip=None) -> list:
    """Four FP32 values (16-byte aligned) in one vector load."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        False,
        loc,
        ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), loaded, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def store_words(address: Int64, words: list, *, loc=None, ip=None) -> None:
    """Store 2 or 4 consecutive 32-bit words in one vector store."""
    count = len(words)
    registers = ", ".join(f"${i + 1}" for i in range(count))
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            *(Uint32(word).ir_value(loc=loc, ip=ip) for word in words),
        ],
        f"st.global.v{count}.u32 [$0], {{{registers}}};",
        ",".join(["l"] + ["r"] * count),
        True,
        loc,
        ip,
    )


@dsl_user_op
def load_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    """One FP32 global load."""
    return Float32(
        _asm(
            T.f32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def load_f32_shared_by_ctas(address: Int64, *, loc=None, ip=None) -> Float32:
    """One FP32 load from L2 of a value another CTA of this grid wrote; it
    stays after the ticket that orders it."""
    return Float32(
        _asm(
            T.f32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def store_f32(address: Int64, value: Float32, *, loc=None, ip=None) -> None:
    """One FP32 global store."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.f32 [$0], $1;",
        "l,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def fence_gpu(*, loc=None, ip=None) -> None:
    """Order this thread's memory accesses at GPU scope (acquire-release)."""
    _asm(None, [], "fence.acq_rel.gpu;", "", True, loc, ip)


@dsl_user_op
def ticket(address: Int64, *, loc=None, ip=None) -> Uint32:
    """Atomically add one to the counter at `address`; the old value."""
    return Uint32(
        _asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Uint32(1).ir_value(loc=loc, ip=ip),
            ],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def fma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """`a * b + c` with one rounding."""
    return Float32(
        _asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def pack_bf16x2(low: Float32, high: Float32, *, loc=None, ip=None) -> Uint32:
    """Two FP32 values rounded to nearest-even BF16, packed low-first."""
    return Uint32(
        _asm(
            T.i32(),
            [
                Float32(high).ir_value(loc=loc, ip=ip),
                Float32(low).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@cute.jit
def bf16_low(word: Uint32) -> Float32:
    """The FP32 value of the low BF16 of a packed pair."""
    return f32_from_bits(word << 16)


@cute.jit
def bf16_high(word: Uint32) -> Float32:
    """The FP32 value of the high BF16 of a packed pair."""
    return f32_from_bits(word & Uint32(0xFFFF0000))


def unpack(words: list) -> list:
    """BF16 values of packed words as FP32, in memory order."""
    values = []
    for word in words:
        values.append(bf16_low(word))
        values.append(bf16_high(word))
    return values


def pack(values: list) -> list:
    """FP32 values rounded to BF16 and packed in pairs."""
    return [
        pack_bf16x2(values[2 * i], values[2 * i + 1]) for i in range(len(values) // 2)
    ]


def rounded(values: list) -> tuple[list, list]:
    """The packed BF16 words of `values` and their FP32 values."""
    words = pack(values)
    return words, unpack(words)


@cute.jit
def warp_sum(value: Float32, width: cutlass.Constexpr[int]) -> Float32:
    """Butterfly sum over aligned groups of `width` lanes."""
    offset = 1
    for _ in cutlass.range_constexpr(width.bit_length() - 1):
        value = value + cute.arch.shuffle_sync_bfly(value, offset=offset)
        offset *= 2
    return value


@cute.jit
def transpose_sum(values: cute.Tensor, count: cutlass.Constexpr[int], lane: Int32):
    """Warp sums of the first `count` values of every lane, taken as 16
    groups of `count / 16` consecutive values (group `g` is
    `values[g * count / 16 :]`): on return, the first `count / 16` values
    of lane `l` hold the sums of group `g`, whose 4-bit index is
    `l % 16` reversed. The additions are those of `warp_sum(value, 32)`."""
    assert count % 16 == 0
    n = count
    offset = 1
    for _ in cutlass.range_constexpr(4):
        half = n // 2
        upper = (lane & offset) != 0
        for i in cutlass.range_constexpr(half):
            keep = values[i]
            send = values[i + half]
            if upper:
                keep = values[i + half]
                send = values[i]
            values[i] = keep + cute.arch.shuffle_sync_bfly(send, offset=offset)
        n = half
        offset *= 2
    for i in cutlass.range_constexpr(n):
        values[i] = values[i] + cute.arch.shuffle_sync_bfly(values[i], offset=16)


@cute.jit
def column_sum(value: Float32) -> Float32:
    """Sum over lanes `e ^ 4` and `e ^ 8`: a comb column of a 16-lane
    group."""
    value = value + cute.arch.shuffle_sync_bfly(value, offset=4)
    return value + cute.arch.shuffle_sync_bfly(value, offset=8)


@cute.jit
def sigmoid(x: Float32) -> Float32:
    """`1 / (1 + exp(-x))`."""
    return Float32(1.0) / (Float32(1.0) + cute.math.exp(-x, fastmath=False))


class PostCollapse:
    """Post-mix, collapse and RMSNorm of one row per CTA; see the module
    docstring and `reference.mhc_post_collapse`."""

    def __init__(self, hidden: int, has_post: bool, write_aux: bool):
        assert hidden % (VEC * 32) == 0 and hidden // VEC <= 1024
        self.hidden = hidden
        self.threads = hidden // VEC
        self.warps = self.threads // 32
        self.has_post = has_post
        self.write_aux = write_aux

    @cute.jit
    def __call__(
        self,
        x: Int64,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sums: cute.struct.MemRange[Float32, self.warps]

        self.kernel(
            x,
            residual,
            post,
            comb,
            pre,
            pre_stride,
            weight,
            residual_out,
            layer_input,
            aux,
            eps,
            Storage,
        ).launch(grid=(rows, 1, 1), block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        eps: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSums = storage.sums.get_tensor(cute.make_layout(self.warps))
        loaded = self.load_row(residual, post, comb, pre, pre_stride, weight, row, tid)
        xs = None
        if cutlass.const_expr(self.has_post):
            xs = unpack(
                load_words(
                    x + Int64(row) * (self.hidden * 2) + Int64(tid) * (VEC * 2), 4
                )
            )
        self.finish_row(
            xs, loaded, residual_out, layer_input, aux, eps, sSums, row, tid
        )

    @cute.jit
    def load_row(
        self,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        row: Int32,
        tid: Int32,
    ) -> tuple:
        """Everything but the sublayer output that thread `tid` of row
        `row` reads: its vector of the norm weight and of every stream, and
        the row's mixes. A caller can issue these loads early."""
        hidden = self.hidden
        # Byte offsets of this thread's vector in a row and in the streams.
        column = Int64(tid) * (VEC * 2)
        streams = Int64(row) * (HC * hidden * 2) + column
        w = load_words(weight + column, 4)
        words = [load_words(residual + streams + j * hidden * 2, 4) for j in range(HC)]
        p = load_f32x4(pre + Int64(row) * Int64(pre_stride) * 4)
        q = None
        c = None
        if cutlass.const_expr(self.has_post):
            q = load_f32x4(post + Int64(row) * (HC * 4))
            c = [load_f32x4(comb + (Int64(row) * HC + k) * (HC * 4)) for k in range(HC)]
        return w, words, p, q, c

    @cute.jit
    def finish_row(
        self,
        xs,
        loaded: tuple,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        eps: Float32,
        sSums: cute.Tensor,
        row: Int32,
        tid: Int32,
    ):
        """The post-mix (with the sublayer output's values `xs`, eight per
        thread), stored streams, stream mean, collapse and RMSNorm of a row,
        from `load_row`'s values; every thread of the CTA takes part."""
        hidden = self.hidden
        column = Int64(tid) * (VEC * 2)
        streams = Int64(row) * (HC * hidden * 2) + column
        w_words, words, p, q, c = loaded
        w = unpack(w_words)
        r = [unpack(words[j]) for j in range(HC)]
        if cutlass.const_expr(self.has_post):
            mixed = []
            for j in cutlass.range_constexpr(HC):
                values = []
                for e in cutlass.range_constexpr(VEC):
                    value = q[j] * xs[e]
                    for k in cutlass.range_constexpr(HC):
                        value = fma(c[k][j], r[k][e], value)
                    values.append(value)
                out_words, out_values = rounded(values)
                store_words(residual_out + streams + j * hidden * 2, out_words)
                mixed.append(out_values)
            r = mixed

        if cutlass.const_expr(self.write_aux):
            means = []
            for e in cutlass.range_constexpr(VEC):
                total = r[0][e]
                for j in cutlass.range_constexpr(1, HC):
                    total = total + r[j][e]
                means.append(total * Float32(1.0 / HC))
            store_words(aux + Int64(row) * (hidden * 2) + column, pack(means))

        collapsed = []
        for e in cutlass.range_constexpr(VEC):
            value = p[0] * r[0][e]
            for j in cutlass.range_constexpr(1, HC):
                value = fma(p[j], r[j][e], value)
            collapsed.append(value)
        _, collapsed = rounded(collapsed)
        squares = Float32(0.0)
        for e in cutlass.range_constexpr(VEC):
            squares = fma(collapsed[e], collapsed[e], squares)

        squares = warp_sum(squares, 32)
        if tid % 32 == 0:
            sSums[tid // 32] = squares
        cute.arch.barrier()
        total = sSums[0]
        for i in cutlass.range_constexpr(1, self.warps):
            total = total + sSums[i]
        inverse_rms = cute.math.rsqrt(total / Float32(hidden) + eps)
        normed = [collapsed[e] * inverse_rms * w[e] for e in range(VEC)]
        store_words(layer_input + Int64(row) * (hidden * 2) + column, pack(normed))


class Coefficients:
    """The next pre-, post- and comb mixes; see the module docstring and
    `reference.mhc_coefficients`.

    `streams` is 4 for the residual streams, or 1 when the projection reads
    one hidden-wide row (the first layer's broadcast embedding). With
    `has_post` the streams are the post-mix of `x` and `residual`
    (`PostCollapse`'s output, recomputed); otherwise `residual` itself.
    """

    def __init__(self, hidden: int, streams: int, has_post: bool):
        assert hidden % SLICE == 0 and MIXES % COEF_WARPS == 0
        assert not has_post or streams == HC
        self.hidden = hidden
        self.streams = streams
        self.has_post = has_post
        self.ctas = hidden // SLICE
        self.width = streams * hidden

    @cute.jit
    def __call__(
        self,
        x: Int64,
        residual: Int64,
        post: Int64,
        comb: Int64,
        fn: Int64,
        scale: Int64,
        base: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        partials: Int64,
        counters: Int64,
        rows: Int32,
        rms_eps: Float32,
        pre_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        iterations: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            values: cute.struct.MemRange[Uint32, ROW_GROUP * self.streams * SLICE // 2]
            stats: cute.struct.MemRange[Float32, ROW_GROUP * STATS]
            last: cute.struct.MemRange[Int32, 1]

        self.kernel(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            post_out,
            comb_out,
            pre_out,
            partials,
            counters,
            rows,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_mult,
            iterations,
            Storage,
        ).launch(grid=(self.ctas, 1, 1), block=[COEF_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        residual: Int64,
        post: Int64,
        comb: Int64,
        fn: Int64,
        scale: Int64,
        base: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        partials: Int64,
        counters: Int64,
        rows: Int32,
        rms_eps: Float32,
        pre_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        iterations: Int32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        hidden = self.hidden
        streams = self.streams
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        # values[row][stream][w][lane]: columns 4 * lane + 2 w and + 1 of the
        # slice as a BF16 pair (the values are BF16), so that a warp's
        # accesses for one w are consecutive words.
        sValues = storage.values.get_tensor(
            cute.make_layout(ROW_GROUP * streams * SLICE // 2)
        )
        sStats = storage.stats.get_tensor(cute.make_layout(ROW_GROUP * STATS))
        sLast = storage.last.get_tensor(cute.make_layout(1))
        column = cta * SLICE + 4 * lane
        groups = (rows + ROW_GROUP - 1) // ROW_GROUP

        # This warp's three mixes over this lane's four columns of each
        # stream, held in registers for every row.
        mix0 = warp * MIXES_PER_WARP
        weights = cute.make_rmem_tensor(
            cute.make_layout((MIXES_PER_WARP, streams, 4)), Float32
        )
        for m in cutlass.range_constexpr(MIXES_PER_WARP):
            for j in cutlass.range_constexpr(streams):
                quad = load_weights4(
                    fn + ((Int64(mix0 + m) * self.width) + j * hidden + column) * 4
                )
                for e in cutlass.range_constexpr(4):
                    weights[m, j, e] = quad[e]

        for g in cutlass.range(groups, unroll=1):
            # The CTAs start at different groups, so that the last arrivals
            # at the groups, which finish them, fall on different CTAs.
            group = (cta + g) % groups
            first = group * ROW_GROUP
            # The group's streams for this CTA's columns, as BF16 pairs;
            # eight rows at a time, a warp per row and four columns
            # per lane. A row past the batch stages the last row again, so
            # that no branch holds back the loads; its statistics are not
            # stored.
            for it in cutlass.range_constexpr(ROW_GROUP // COEF_WARPS):
                local = it * COEF_WARPS + warp
                self.stage(
                    sValues,
                    x,
                    residual,
                    post,
                    comb,
                    local,
                    cutlass.min(first + local, rows - 1),
                    lane,
                    column,
                )
            cute.arch.barrier()

            # Per-lane dot products and (warp 0) sums of squares of every
            # row, reduced over the warp into partials[group][cta][row][stat].
            acc = cute.make_rmem_tensor(
                cute.make_layout((MIXES_PER_WARP, ROW_GROUP)), Float32
            )
            acc.fill(0.0)
            squares = cute.make_rmem_tensor(cute.make_layout(ROW_GROUP), Float32)
            squares.fill(0.0)
            for local in cutlass.range_constexpr(ROW_GROUP):
                self.accumulate(sValues, weights, acc, squares, local, lane, warp)
            self.publish(
                acc,
                squares,
                partials + (Int64(group) * self.ctas + cta) * (ROW_GROUP * STATS * 4),
                rows - first,
                lane,
                warp,
                mix0,
            )

            # The last CTA to arrive at the group finishes its rows. The
            # barrier orders the CTA's partials before thread 0's release
            # fence and ticket (and frees the values for the next group); in
            # the last CTA, the ticket's acquire and the second barrier order
            # the other CTAs' partials before its reads.
            cute.arch.barrier()
            counter = counters + Int64(group) * 4
            if tid == 0:
                sLast[0] = Int32(0)
                fence_gpu()
                if ticket(counter) == Uint32(self.ctas - 1):
                    sLast[0] = Int32(1)
            cute.arch.barrier()
            if sLast[0] == 1:
                self.finish_group(
                    sStats,
                    partials,
                    counter,
                    tid,
                    group,
                    first,
                    rows,
                    scale,
                    base,
                    post_out,
                    comb_out,
                    pre_out,
                    rms_eps,
                    pre_eps,
                    sinkhorn_eps,
                    post_mult,
                    iterations,
                )

    @cute.jit
    def stage(
        self,
        sValues: cute.Tensor,
        x: Int64,
        residual: Int64,
        post: Int64,
        comb: Int64,
        local: Int32,
        row: Int32,
        lane: Int32,
        column: Int32,
    ):
        """One row's four columns per lane of every stream into shared
        memory as BF16 pairs, post-mixed and rounded with `has_post`."""
        words = self.stream_words(x, residual, post, comb, row, column)
        for j in cutlass.range_constexpr(self.streams):
            for w in cutlass.range_constexpr(2):
                sValues[((local * self.streams + j) * 2 + w) * 32 + lane] = words[j][w]

    @cute.jit
    def stream_words(
        self,
        x: Int64,
        residual: Int64,
        post: Int64,
        comb: Int64,
        row: Int32,
        column: Int32,
    ) -> list:
        """Four columns of every stream of a row as two words of BF16 pairs
        each, post-mixed and rounded with `has_post`."""
        hidden = self.hidden
        streams = self.streams
        words = [
            load_words(residual + ((Int64(row) * streams + j) * hidden + column) * 2, 2)
            for j in range(streams)
        ]
        if cutlass.const_expr(self.has_post):
            r = [unpack(words[j]) for j in range(streams)]
            xs = unpack(load_words(x + (Int64(row) * hidden + column) * 2, 2))
            q = load_f32x4(post + Int64(row) * (HC * 4))
            c = [load_f32x4(comb + (Int64(row) * HC + k) * (HC * 4)) for k in range(HC)]
            mixed = []
            for j in cutlass.range_constexpr(HC):
                values = []
                for e in cutlass.range_constexpr(4):
                    value = q[j] * xs[e]
                    for k in cutlass.range_constexpr(HC):
                        value = fma(c[k][j], r[k][e], value)
                    values.append(value)
                mixed.append(pack(values))
            words = mixed
        return words

    @cute.jit
    def accumulate(
        self,
        sValues: cute.Tensor,
        weights: cute.Tensor,
        acc: cute.Tensor,
        squares: cute.Tensor,
        local: cutlass.Constexpr[int],
        lane: Int32,
        warp: Int32,
    ):
        """This lane's share of row `local`'s dot products (and, in warp
        0, its sum of squares)."""
        streams = self.streams
        v = [
            unpack(
                [sValues[((local * streams + j) * 2 + w) * 32 + lane] for w in range(2)]
            )
            for j in range(streams)
        ]
        for m in cutlass.range_constexpr(MIXES_PER_WARP):
            total = acc[m, local]
            for j in cutlass.range_constexpr(streams):
                for e in cutlass.range_constexpr(4):
                    total = fma(weights[m, j, e], v[j][e], total)
            acc[m, local] = total
        if warp == 0:
            self.accumulate_squares(squares, v, local)

    @cute.jit
    def accumulate_squares(
        self, squares: cute.Tensor, v: list, local: cutlass.Constexpr[int]
    ):
        """Row `local`'s sum of squares over this lane's columns."""
        total = squares[local]
        for j in cutlass.range_constexpr(self.streams):
            for e in cutlass.range_constexpr(4):
                total = fma(v[j][e], v[j][e], total)
        squares[local] = total

    @cute.jit
    def publish(
        self,
        acc: cute.Tensor,
        squares: cute.Tensor,
        address: Int64,
        valid: Int32,
        lane: Int32,
        warp: Int32,
        mix0: Int32,
    ):
        """Warp sums of the group's statistics into its partials at
        `address`, for its first `valid` rows.

        A transposed butterfly: at each of the offsets 1, 2, 4 and 8 a lane
        keeps one half of its values (by its lane bit) and adds its
        partner's copy of that half, so the 48 sums (16 for the squares)
        take 48 shuffles (16) instead of five each. The additions are those
        of `warp_sum`, so the sums are bitwise the same. Lane `l` and
        `l ^ 16` end with the row whose index is the low four bits of
        `l` reversed."""
        row = (
            ((lane & 1) << 3)
            | ((lane & 2) << 1)
            | ((lane & 4) >> 1)
            | ((lane & 8) >> 3)
        )
        transpose_sum(acc, MIXES_PER_WARP * ROW_GROUP, lane)
        if lane < 16 and row < valid:
            for m in cutlass.range_constexpr(MIXES_PER_WARP):
                store_f32(address + (row * STATS + mix0 + m) * 4, acc[m])
        if warp == 0:
            transpose_sum(squares, ROW_GROUP, lane)
            if lane < 16 and row < valid:
                store_f32(address + (row * STATS + MIXES) * 4, squares[0])

    @cute.jit
    def finish_group(
        self,
        sStats: cute.Tensor,
        partials: Int64,
        counter: Int64,
        tid: Int32,
        group: Int32,
        first: Int32,
        rows: Int32,
        scale: Int64,
        base: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        rms_eps: Float32,
        pre_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        iterations: Int32,
    ):
        """In the row group's last CTA: reset its counter, sum every row's
        partials in CTA order and compute its mixes."""
        if tid == 0:
            store_u32(counter, Uint32(0))
        for it in cutlass.range_constexpr(
            (ROW_GROUP * STATS + COEF_THREADS - 1) // COEF_THREADS
        ):
            if it * COEF_THREADS + tid < ROW_GROUP * STATS:
                self.gather(
                    sStats, partials, it * COEF_THREADS + tid, group, first, rows
                )
        cute.arch.barrier()
        self.finish(
            sStats,
            tid % 32,
            tid // 32,
            first,
            rows,
            scale,
            base,
            post_out,
            comb_out,
            pre_out,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_mult,
            iterations,
        )

    @cute.jit
    def gather(
        self,
        sStats: cute.Tensor,
        partials: Int64,
        pair: Int32,
        group: Int32,
        first: Int32,
        rows: Int32,
    ):
        """Statistic `pair % STATS` of row `pair // STATS`, summed over
        the CTAs in order (zero for a row past the batch)."""
        local = pair // STATS
        address = (
            partials
            + (Int64(group) * self.ctas * ROW_GROUP + local) * (STATS * 4)
            + (pair % STATS) * 4
        )
        if first + local < rows:
            sStats[pair] = self.sum_partials(address)
        else:
            sStats[pair] = Float32(0.0)

    @cute.jit
    def sum_partials(self, address: Int64) -> Float32:
        """The sum, in CTA order, of one statistic's partials."""
        total = load_f32_shared_by_ctas(address)
        for c in cutlass.range_constexpr(1, self.ctas):
            total = total + load_f32_shared_by_ctas(
                address + c * (ROW_GROUP * STATS * 4)
            )
        return total

    @cute.jit
    def finish(
        self,
        sStats: cute.Tensor,
        lane: Int32,
        warp: Int32,
        first: Int32,
        rows: Int32,
        scale: Int64,
        base: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        rms_eps: Float32,
        pre_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        iterations: Int32,
    ):
        """Mixes of two rows per warp, a 16-lane group per row: lane `e`
        owns comb entry `(e // 4, e % 4)` and, below 4, the pre- and
        post-mix `e`."""
        local = warp * 2 + lane // 16
        e = lane % 16
        row = first + local
        stats = local * STATS
        inverse_rms = cute.math.rsqrt(
            sStats[stats + MIXES] / Float32(self.width) + rms_eps
        )
        if e < HC and row < rows:
            self.store_pre_post(
                sStats,
                stats,
                e,
                row,
                inverse_rms,
                scale,
                base,
                post_out,
                pre_out,
                pre_eps,
                post_mult,
            )

        # Softmax over each comb row, then alternating column and row
        # normalizations (Sinkhorn); row sums use lanes e ^ 1, e ^ 2 and
        # column sums lanes e ^ 4, e ^ 8.
        value = fma(
            sStats[stats + 2 * HC + e] * inverse_rms,
            load_f32(scale + 8),
            load_f32(base + (2 * HC + e) * 4),
        )
        top = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=1))
        top = cute.arch.fmax(top, cute.arch.shuffle_sync_bfly(top, offset=2))
        value = cute.math.exp(value - top, fastmath=False)
        value = value / warp_sum(value, 4) + sinkhorn_eps
        value = value / (column_sum(value) + sinkhorn_eps)
        # A NaN row (its RMS is NaN) stays NaN; when the whole warp is NaN,
        # skip the remaining divisions, which take a slow path on NaN.
        repeats = (iterations - 1) * Int32(
            cute.arch.vote_any_sync(inverse_rms >= Float32(0.0))
        )
        for _ in cutlass.range(repeats, unroll=1):
            value = value / (warp_sum(value, 4) + sinkhorn_eps)
            value = value / (column_sum(value) + sinkhorn_eps)
        if row < rows:
            store_f32(comb_out + (Int64(row) * HC * HC + e) * 4, value)

    @cute.jit
    def store_pre_post(
        self,
        sStats: cute.Tensor,
        stats: Int32,
        e: Int32,
        row: Int32,
        inverse_rms: Float32,
        scale: Int64,
        base: Int64,
        post_out: Int64,
        pre_out: Int64,
        pre_eps: Float32,
        post_mult: Float32,
    ):
        """Pre- and post-mix `e` of a row."""
        pre_logit = fma(
            sStats[stats + e] * inverse_rms, load_f32(scale), load_f32(base + e * 4)
        )
        post_logit = fma(
            sStats[stats + HC + e] * inverse_rms,
            load_f32(scale + 4),
            load_f32(base + (HC + e) * 4),
        )
        store_f32(pre_out + (Int64(row) * HC + e) * 4, sigmoid(pre_logit) + pre_eps)
        store_f32(post_out + (Int64(row) * HC + e) * 4, sigmoid(post_logit) * post_mult)


class CoefficientsMma(Coefficients):
    """`Coefficients` of the four residual streams with the projection on
    tensor cores, in `hidden / 320` CTAs; see the module docstring.

    A CTA owns 320 hidden columns of the four streams, a projection width of
    1280, which its eight warps split into a stream's half each. A warp
    holds its 24 x 160 slice of the FP32 projection in registers as two BF16
    terms, `hi = bf16(w)` and `lo = bf16(w - hi)`, in `mma.sync`
    fragment order. For each group of 16 rows the CTA stages the post-mixed
    streams in shared memory, each warp multiplies them by both terms
    (m16n8k16, FP32 accumulation) and sums the squares of its fragments, and
    the warps' partials are added in warp order into the CTA's partials, which
    the group's last CTA finishes as `Coefficients` does.
    """

    def __init__(self, hidden: int, has_post: bool):
        super().__init__(hidden, HC, has_post)
        assert hidden % MMA_SLICE == 0
        self.ctas = hidden // MMA_SLICE

    @cute.jit
    def __call__(
        self,
        x: Int64,
        residual: Int64,
        post: Int64,
        comb: Int64,
        fn: Int64,
        scale: Int64,
        base: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        partials: Int64,
        counters: Int64,
        rows: Int32,
        rms_eps: Float32,
        pre_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        iterations: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            values: cute.struct.MemRange[Uint32, ROW_GROUP * MMA_ROW_WORDS]
            stats: cute.struct.MemRange[Float32, ROW_GROUP * STATS]
            last: cute.struct.MemRange[Int32, 1]

        self.kernel(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            post_out,
            comb_out,
            pre_out,
            partials,
            counters,
            rows,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            post_mult,
            iterations,
            Storage,
        ).launch(grid=(self.ctas, 1, 1), block=[COEF_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        residual: Int64,
        post: Int64,
        comb: Int64,
        fn: Int64,
        scale: Int64,
        base: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        partials: Int64,
        counters: Int64,
        rows: Int32,
        rms_eps: Float32,
        pre_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        iterations: Int32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        # The fragment coordinates: rows g and g + 8, word t of a k-tile.
        g = lane // 4
        t = lane % 4
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        # values[row][word]: the group's streams of this CTA's columns as
        # BF16 pairs, word = stream * 160 + column / 2. After the MMAs the
        # first words hold the warps' partial statistics,
        # warp_stats[warp][row][stat].
        sValues = storage.values.get_tensor(cute.make_layout(ROW_GROUP * MMA_ROW_WORDS))
        sWarpStats = cute.make_tensor(
            cute.recast_ptr(sValues.iterator, dtype=Float32),
            cute.make_layout(COEF_WARPS * ROW_GROUP * STATS),
        )
        sStats = storage.stats.get_tensor(cute.make_layout(ROW_GROUP * STATS))
        sLast = storage.last.get_tensor(cute.make_layout(1))
        groups = (rows + ROW_GROUP - 1) // ROW_GROUP

        # This warp's projection slice in B-fragment order: mix nt * 8 + g,
        # columns 2t, 2t + 1 (word 0) and 2t + 8, 2t + 9 (word 1) of k-tile
        # kt. The FP32 values stay in registers and are split into the two
        # BF16 terms at each use, so that these loads fly during the first
        # group's staging.
        stream = warp // 2
        half = warp % 2
        k0 = stream * self.hidden + cta * MMA_SLICE + half * MMA_WARP_K + 2 * t
        weights = cute.make_rmem_tensor(
            cute.make_layout((MMA_K_TILES, MMA_N_TILES, 2, 2)), Float32
        )
        for kt in cutlass.range_constexpr(MMA_K_TILES):
            for nt in cutlass.range_constexpr(MMA_N_TILES):
                for w in cutlass.range_constexpr(2):
                    pair = load_weights2(
                        fn + (Int64(nt * 8 + g) * self.width + k0 + kt * 16 + w * 8) * 4
                    )
                    for e in cutlass.range_constexpr(2):
                        weights[kt, nt, w, e] = pair[e]
        # The words of this warp's A fragments in a staged row.
        a0 = (
            g * MMA_ROW_WORDS + stream * (MMA_SLICE // 2) + half * (MMA_WARP_K // 2) + t
        )

        for i in cutlass.range(groups, unroll=1):
            # The CTAs start at different groups, so that the last arrivals
            # at the groups, which finish them, fall on different CTAs.
            group = (cta + i) % groups
            first = group * ROW_GROUP
            # Stage the group: 16 threads per row, each four columns of
            # every stream in five quads. A row past the batch stages the
            # last row again; its statistics are not stored.
            local = tid // MMA_ROW_THREADS
            row = cutlass.min(first + local, rows - 1)
            for q in cutlass.range_constexpr(MMA_QUADS):
                quad = tid % MMA_ROW_THREADS + q * MMA_ROW_THREADS
                words = self.stream_words(
                    x, residual, post, comb, row, cta * MMA_SLICE + 4 * quad
                )
                for j in cutlass.range_constexpr(HC):
                    for w in cutlass.range_constexpr(2):
                        sValues[
                            local * MMA_ROW_WORDS + j * (MMA_SLICE // 2) + 2 * quad + w
                        ] = words[j][w]
            cute.arch.barrier()

            # acc[nt][term] += A x term, and the squares of rows g, g + 8.
            acc = cute.make_rmem_tensor(cute.make_layout((MMA_N_TILES, 2, 4)), Float32)
            acc.fill(0.0)
            squares = cute.make_rmem_tensor(cute.make_layout(2), Float32)
            squares.fill(0.0)
            for kt in cutlass.range_constexpr(MMA_K_TILES):
                a = [
                    sValues[a0 + r * 8 * MMA_ROW_WORDS + kt * 8 + w * 4]
                    for w in range(2)
                    for r in range(2)
                ]
                for nt in cutlass.range_constexpr(MMA_N_TILES):
                    hi = [
                        pack_bf16x2(weights[kt, nt, w, 0], weights[kt, nt, w, 1])
                        for w in range(2)
                    ]
                    lo = [
                        pack_bf16x2(
                            weights[kt, nt, w, 0] - bf16_low(hi[w]),
                            weights[kt, nt, w, 1] - bf16_high(hi[w]),
                        )
                        for w in range(2)
                    ]
                    for term, b in ((0, hi), (1, lo)):
                        c = mma_bf16([acc[nt, term, e] for e in range(4)], a, b)
                        for e in cutlass.range_constexpr(4):
                            acc[nt, term, e] = c[e]
                # a[0], a[2] are row g; a[1], a[3] row g + 8.
                for r in cutlass.range_constexpr(2):
                    for v in unpack([a[r], a[r + 2]]):
                        squares[r] = fma(v, v, squares[r])
            cute.arch.barrier()

            # The warps' partials, then their sums in warp order into
            # partials[group][cta][row][stat].
            for nt in cutlass.range_constexpr(MMA_N_TILES):
                for e in cutlass.range_constexpr(4):
                    sWarpStats[
                        (warp * ROW_GROUP + g + 8 * (e // 2)) * STATS
                        + nt * 8
                        + 2 * t
                        + e % 2
                    ] = acc[nt, 0, e] + acc[nt, 1, e]
            for r in cutlass.range_constexpr(2):
                total = warp_sum(squares[r], 4)
                if t == 0:
                    sWarpStats[(warp * ROW_GROUP + g + 8 * r) * STATS + MIXES] = total
            cute.arch.barrier()
            address = partials + (Int64(group) * self.ctas + cta) * (
                ROW_GROUP * STATS * 4
            )
            for it in cutlass.range_constexpr(
                (ROW_GROUP * STATS + COEF_THREADS - 1) // COEF_THREADS
            ):
                stat = it * COEF_THREADS + tid
                if stat < ROW_GROUP * STATS and stat // STATS < rows - first:
                    total = sWarpStats[stat]
                    for w in cutlass.range_constexpr(1, COEF_WARPS):
                        total = total + sWarpStats[w * ROW_GROUP * STATS + stat]
                    store_f32(address + stat * 4, total)

            # The last CTA to arrive at the group finishes its rows, as in
            # Coefficients; the barrier also frees the values for the next
            # group.
            cute.arch.barrier()
            counter = counters + Int64(group) * 4
            if tid == 0:
                sLast[0] = Int32(0)
                fence_gpu()
                if ticket(counter) == Uint32(self.ctas - 1):
                    sLast[0] = Int32(1)
            cute.arch.barrier()
            if sLast[0] == 1:
                self.finish_group(
                    sStats,
                    partials,
                    counter,
                    tid,
                    group,
                    first,
                    rows,
                    scale,
                    base,
                    post_out,
                    comb_out,
                    pre_out,
                    rms_eps,
                    pre_eps,
                    sinkhorn_eps,
                    post_mult,
                    iterations,
                )


def _fake_stream():
    """A placeholder stream for compilation."""
    return make_fake_stream()


@cache
def _post_collapse_kernel(
    device_index: int, hidden: int, has_post: bool, write_aux: bool
):
    """`PostCollapse` compiled for a device and variant."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            PostCollapse(hidden, has_post, write_aux),
            *(Int64(0),) * 5,
            Int32(4),
            *(Int64(0),) * 4,
            Int32(1),
            Float32(1e-6),
            _fake_stream(),
        )


@cache
def _coefficients_kernel(device_index: int, hidden: int, streams: int, has_post: bool):
    """`Coefficients` compiled for a device and variant."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            Coefficients(hidden, streams, has_post),
            *(Int64(0),) * 12,
            Int32(1),
            *(Float32(1e-6),) * 4,
            Int32(20),
            _fake_stream(),
        )


@cache
def _coefficients_mma_kernel(device_index: int, hidden: int, has_post: bool):
    """`CoefficientsMma` compiled for a device and variant."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            CoefficientsMma(hidden, has_post),
            *(Int64(0),) * 12,
            Int32(1),
            *(Float32(1e-6),) * 4,
            Int32(20),
            _fake_stream(),
        )


def _address(tensor: torch.Tensor | None) -> Int64:
    """A tensor's device address as a kernel argument (0 for None)."""
    return Int64(0 if tensor is None else tensor.data_ptr())


@cache
def _counters(device_index: int) -> torch.Tensor:
    """The per-device ticket counters of `Coefficients`, zeroed once."""
    return torch.zeros(
        MAX_ROW_GROUPS, dtype=torch.int32, device=torch.device("cuda", device_index)
    )


@cache
def _select_first_stream(device_index: int) -> torch.Tensor:
    """The pre-mix `(1, 0, 0, 0)` that selects stream zero, for every row."""
    pre = torch.zeros(
        HC, dtype=torch.float32, device=torch.device("cuda", device_index)
    )
    pre[0] = 1.0
    return pre


def _uses_mma(streams: int, hidden: int) -> bool:
    """Whether `CoefficientsMma` projects these streams: the four residual
    streams in 320-column slices, on SM80. One hidden-wide row (the first
    layer's broadcast embedding), other widths and SM90 project on CUDA
    cores: on H100 at TP8 with five draft tokens the tensor-core kernel made
    the decode step 1.7% slower at 6 rows and 0.5-0.7% faster at 24 and 48."""
    return streams == HC and hidden % MMA_SLICE == 0 and is_ampere()


def supports(rows: int, hidden: int, hc_mult: int) -> bool:
    """Whether these kernels take a boundary of `rows` rows."""
    return (
        0 < rows <= MAX_ROWS
        and hc_mult == HC
        and hidden % SLICE == 0
        and hidden % (VEC * 32) == 0
        and hidden // VEC <= 1024
    )


def prepare(device_index: int, hidden: int) -> None:
    """Allocate the ticket counters and the stream-zero pre-mix, and compile
    every kernel variant the model uses, before any CUDA-graph capture."""
    _counters(device_index)
    _select_first_stream(device_index)
    for has_post, write_aux in ((True, False), (True, True), (False, False)):
        _post_collapse_kernel(device_index, hidden, has_post, write_aux)
    for has_post in (True, False):
        if _uses_mma(HC, hidden):
            _coefficients_mma_kernel(device_index, hidden, has_post)
        else:
            _coefficients_kernel(device_index, hidden, HC, has_post)
    _coefficients_kernel(device_index, hidden, 1, False)


def mhc_post_collapse(
    x: torch.Tensor | None,
    residual: torch.Tensor,
    post_mix: torch.Tensor | None,
    comb_mix: torch.Tensor | None,
    pre_mix: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    write_aux: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`reference.mhc_post_collapse` on the GPU.

    With `x` the streams are first post-mixed (`post_mix` `[rows, 4]`,
    `comb_mix` `[rows, 4, 4]`); `pre_mix` None selects stream zero.
    Returns the streams (new, or `residual` itself without `x`), the
    normalized layer input and the stream mean (empty without
    `write_aux`).
    """
    rows, hc, hidden = residual.shape
    assert hc == HC and residual.dtype == torch.bfloat16 and residual.is_contiguous()
    assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
    device = residual.device
    has_post = x is not None
    out = residual
    if x is not None:
        assert post_mix is not None and comb_mix is not None
        assert x.shape == (rows, hidden) and x.dtype == torch.bfloat16
        assert (
            x.is_contiguous() and post_mix.is_contiguous() and comb_mix.is_contiguous()
        )
        assert post_mix.dtype == comb_mix.dtype == torch.float32
        out = torch.empty_like(residual)
    if pre_mix is None:
        pre, pre_stride = _select_first_stream(device.index), 0
    else:
        assert pre_mix.shape == (rows, HC) and pre_mix.dtype == torch.float32
        assert pre_mix.is_contiguous()
        pre, pre_stride = pre_mix, HC
    layer_input = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
    aux = torch.empty(
        (rows if write_aux else 0, hidden), dtype=torch.bfloat16, device=device
    )
    if rows == 0:
        return out, layer_input, aux
    _post_collapse_kernel(device.index, hidden, has_post, write_aux)(
        _address(x),
        Int64(residual.data_ptr()),
        _address(post_mix),
        _address(comb_mix),
        Int64(pre.data_ptr()),
        Int32(pre_stride),
        Int64(norm_weight.data_ptr()),
        Int64(out.data_ptr()),
        Int64(layer_input.data_ptr()),
        Int64(aux.data_ptr()),
        Int32(rows),
        Float32(norm_eps),
        current_cu_stream(),
    )
    return out, layer_input, aux


def mhc_coefficients(
    x: torch.Tensor | None,
    residual: torch.Tensor,
    post_mix: torch.Tensor | None,
    comb_mix: torch.Tensor | None,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`reference.mhc_coefficients` on the GPU: the post-, comb- and
    pre-mix `[rows, 4]`, `[rows, 4, 4]`, `[rows, 4]` (FP32).

    `residual` is `[rows, 4, hidden]` or, for a projection of one
    hidden-wide row, `[rows, hidden]`; with `x` the streams are first
    post-mixed as in `mhc_post_collapse`. With `stream` the kernel runs
    there after the work queued so far on the current stream; the outputs
    are allocated on the current stream, which must wait for `stream`
    before reading them.
    """
    rows = residual.shape[0]
    hidden = residual.shape[-1]
    streams = residual.shape[1] if residual.dim() == 3 else 1
    width = streams * hidden
    device = residual.device
    has_post = x is not None
    assert residual.dtype == torch.bfloat16 and residual.is_contiguous()
    assert fn.shape == (MIXES, width) and fn.dtype == torch.float32
    assert fn.is_contiguous()
    assert hc_scale.is_contiguous() and hc_base.is_contiguous()
    if x is not None:
        assert post_mix is not None and comb_mix is not None
        assert x.shape == (rows, hidden) and x.is_contiguous()
        assert post_mix.is_contiguous() and comb_mix.is_contiguous()
    post = torch.empty((rows, HC), dtype=torch.float32, device=device)
    comb = torch.empty((rows, HC, HC), dtype=torch.float32, device=device)
    pre = torch.empty((rows, HC), dtype=torch.float32, device=device)
    if rows == 0:
        return post, comb, pre
    assert rows <= MAX_ROWS
    groups = (rows + ROW_GROUP - 1) // ROW_GROUP
    mma = _uses_mma(streams, hidden)
    ctas = hidden // (MMA_SLICE if mma else SLICE)
    partials = torch.empty(
        (groups, ctas, ROW_GROUP, STATS), dtype=torch.float32, device=device
    )
    cu_stream = current_cu_stream()
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream())
        cu_stream = cuda.CUstream(stream.cuda_stream)
        # The caching allocator must not hand these to the current stream
        # before the side stream is done with them.
        for tensor in (x, residual, post_mix, comb_mix, partials):
            if tensor is not None:
                tensor.record_stream(stream)
    kernel = (
        _coefficients_mma_kernel(device.index, hidden, has_post)
        if mma
        else _coefficients_kernel(device.index, hidden, streams, has_post)
    )
    kernel(
        _address(x),
        Int64(residual.data_ptr()),
        _address(post_mix),
        _address(comb_mix),
        Int64(fn.data_ptr()),
        Int64(hc_scale.data_ptr()),
        Int64(hc_base.data_ptr()),
        Int64(post.data_ptr()),
        Int64(comb.data_ptr()),
        Int64(pre.data_ptr()),
        Int64(partials.data_ptr()),
        Int64(_counters(device.index).data_ptr()),
        Int32(rows),
        Float32(rms_eps),
        Float32(hc_pre_eps),
        Float32(hc_sinkhorn_eps),
        Float32(hc_post_mult_value),
        Int32(sinkhorn_repeat),
        cu_stream,
    )
    return post, comb, pre


def mhc_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    capture_aux: bool = False,
    *,
    stream: torch.cuda.Stream | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """A boundary between sublayers: `mhc_post_collapse` and
    `mhc_coefficients` of the sublayer output `x`.

    Returns, as the stock fused post + pre op does, the new streams, the
    post-, comb-mix `[rows, 4, 1]`, `[rows, 4, 4]`, the layer input, the
    next pre-mix and the stream mean (empty without `capture_aux`). With
    `stream` the coefficients run there, concurrently with the collapse
    and whatever follows it; join `stream` before reading them.
    """
    rows = x.shape[0]
    x = x.contiguous()
    post_mix = post_mix.reshape(rows, HC).contiguous()
    # Queue the coefficients first so that the side stream waits only for
    # the sublayer output, not for the collapse too.
    post, comb, pre = mhc_coefficients(
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        stream=stream,
    )
    streams, layer_input, aux = mhc_post_collapse(
        x, residual, post_mix, comb_mix, pre_mix, norm_weight, norm_eps, capture_aux
    )
    return streams, post.unsqueeze(-1), comb, layer_input, pre, aux


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None,
    x: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    *,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A boundary without a sublayer output to add: the coefficients of
    `residual` (or of the one-row projection input `x`) and its collapse
    with `pre_mix` (stream zero when None), as the stock delayed mHC pre
    op returns them: post-, comb-mix, layer input and next pre-mix."""
    post, comb, pre = mhc_coefficients(
        None,
        residual if x is None else x,
        None,
        None,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        stream=stream,
    )
    _, layer_input, _ = mhc_post_collapse(
        None, residual, None, None, pre_mix, norm_weight, norm_eps
    )
    return post.unsqueeze(-1), comb, layer_input, pre
