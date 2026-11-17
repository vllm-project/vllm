# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark draft's dense SwiGLU MLP over NVFP4 weights with F16 activations
(W4A16) for 1 to 8 rows: two CuTe kernels, gate/up + SiLU product, then down.

A weight `W[N, K]` is E2M1 codes (the even column in each byte's low nibble),
one E4M3 scale per 16 K and an FP32 tensor scale. A tile is 16 rows; in the
gate/up weight, rows `16 t .. 16 t + 7` of tile `t` are gate rows `8 t ..` and
the next 8 the matching up rows, so a tile yields 8 SiLU products.

CTA `c` is (row group `c // S`, K split `c % S`), clusters of `S` along K. It
owns `T` tiles over its `K / S` columns. Before `griddepcontrol.wait` it
bulk-copies those weights into shared memory (one mbarrier per tile); after
the wait it stages the rows' x slice as F16 in padded shared memory. Warp `w`
covers tile `w // WK` over K piece `w % WK`: per 64-K block, lane `(g, q)`
dequantizes its rows `g`, `g + 8` over the 16-K scale group `q` (E2M1 to F16
times the E4M3 scale in F16, exact) and feeds `mma.sync.m16n8k16` (F16, FP32
accumulate), x read with the same K permutation. Pieces are summed per CTA,
sent to the tile's owner CTA in the cluster (`tile % S`), summed over the
splits in rank order and scaled; the gate/up kernel applies SiLU(gate) x up
and a power-of-two output scale. The summation order depends only on the
launch shape: runs are bitwise reproducible.

Not bitwise equal to the BF16 MLP: x is F16 (exact for
`2^-14 <= |x| <= 65504` after the power-of-two input scale), the SiLU product
an F16 intermediate; both saturate at 65504 (4x the calibrated amax).
"""

import functools
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Uint16, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.utils.math_utils import cdiv, round_up

from ..common.host import assert_not_capturing, check_operand, pdl_supported
from ..common.primitives import (
    WARP_SIZE,
    _asm,
    _ir,
    _ordered,
    _unpack,
    bf16x2_to_f32x2,
    bulk_prefetch_l2,
    current_cu_stream,
    f32_to_bf16_bits,
    global_ptr,
    load_global_u32x4,
    load_shared_u32x4_address,
    make_fake_dynamic_compact_tensor,
    map_shared_address_to_peer,
    mbarrier_wait_address,
    mma_m16n8k16_f16,
    mul_rn,
    shared_ptr,
    silu_f32,
    store_shared_cluster_f32,
    store_shared_u32x4_address,
    to_cute,
    to_cute_dynamic,
)

TILE_N = 16  # weight rows per tile (the MMA's M)
BLOCK_K = 64  # K per lane block: four MMA k-steps
COPY_CHUNK = 16384  # bytes per bulk copy
X_PAD = 8  # F16 padding per staged x row: conflict-free B-fragment loads
MAX_ROWS = 8
# (splits, tiles per CTA, warps per tile) of the gate/up and the down kernel.
GATE_UP_SHAPE = (4, 8, 2)
DOWN_SHAPE = (1, 4, 4)
# MMA accumulator slots per thread, summed in slot order.
ACCS = 4
F16_HEADROOM = 16384.0  # the power-of-two scales keep |value| below this


@dsl_user_op
def _bulk_g2s(
    smem: Int32, gmem: Int64, nbytes: Int32, bar: Int32, *, loc=None, ip=None
):
    _ordered(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[$0], [$1], $2, [$3];",
        "r,l,r,r",
        (Int32(smem), Int64(gmem), Int32(nbytes), Int32(bar)),
        loc,
        ip,
    )


@dsl_user_op
def _dequant_e2m1x16_f16(w0: Uint32, w1: Uint32, scale: Uint32, *, loc=None, ip=None):
    """Sixteen E2M1 codes (`w0`, `w1`) times the E4M3 scale in the low byte of
    `scale`, as eight F16 pairs (exact: a code x a scale has at most 4
    significant bits and lies in [2^-10, 2688] or is 0).
    Kept as asm: `nvvm.convert_f4x2_to_f16x2` takes one byte per pair, so
    each code byte is unpacked apart (gate_up SASS 1242 -> 1626)."""
    lines = [
        "{",
        ".reg .b32 t, sc, r;",
        ".reg .b16 sp;",
        ".reg .b8 b0, b1, b2, b3;",
        "and.b32 t, $10, 255;",
        "shl.b32 r, t, 8;",
        "or.b32 t, t, r;",
        "cvt.u16.u32 sp, t;",
        "cvt.rn.f16x2.e4m3x2 sc, sp;",
    ]
    for half in range(2):
        lines.append(f"mov.b32 {{b0, b1, b2, b3}}, ${8 + half};")
        for byte in range(4):
            lines += [
                f"cvt.rn.f16x2.e2m1x2 r, b{byte};",
                f"mul.rn.f16x2 ${half * 4 + byte}, r, sc;",
            ]
    lines.append("}")
    struct = _asm(
        llvm.StructType.get_literal([T.i32()] * 8),
        _ir([Uint32(w0), Uint32(w1), Uint32(scale)], loc=loc, ip=ip),
        "\n\t".join(lines),
        "=r,=r,=r,=r,=r,=r,=r,=r,r,r,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _unpack(struct, Uint32, 8, loc=loc, ip=ip)


@dsl_user_op
def _bf16x2_to_f16x2(word: Uint32, scale: Float32, *, loc=None, ip=None) -> Uint32:
    """A BF16 pair times `scale` (a power of two), rounded to an F16 pair
    (saturated to the finite range)."""
    a, b = bf16x2_to_f32x2(word, loc=loc, ip=ip)
    pair = nvvm.convert_f32x2_to_f16x2(
        ir.VectorType.get([2], Float16.mlir_type, loc=loc),
        mul_rn(b, scale, loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
        mul_rn(a, scale, loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
        rnd=nvvm.FPRoundingMode.RN,
        sat=nvvm.SaturationMode.SATFINITE,
        loc=loc,
        ip=ip,
    )
    return Uint32(llvm.bitcast(T.i32(), pair, loc=loc, ip=ip))


@dsl_user_op
def _f32_to_f16_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 bits;\n\tcvt.rn.satfinite.f16.f32 bits, $1;\n\t"
            "cvt.u32.u16 $0, bits;\n\t}",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


class _W4A16Kernel:
    """`out[m, n_out] = epilogue(x[m, K] @ W[N, K]^T)`; see the module
    docstring. `consts` = (scale A, scale B, x scale, output scale), FP32: x
    is staged times the x scale (gate/up only); the gate/up kernel stores
    SiLU(A gate) x (B up), the down kernel A y, times the output scale. With
    `prefetch` > 0 the CTAs prefetch that many bytes at `pf` (the next
    kernel's weights) into L2 before the wait."""

    def __init__(
        self,
        *,
        n: int,
        k: int,
        shape: tuple[int, int, int],
        swiglu: bool,
        prefetch: int,
        enable_pdl: bool,
    ) -> None:
        splits, tpc, wk = shape
        self.n, self.k, self.S, self.T, self.WK = n, k, splits, tpc, wk
        self.swiglu, self.prefetch, self.enable_pdl = swiglu, prefetch, enable_pdl
        self.ks = k // splits
        self.kw = self.ks // wk
        self.blocks = self.kw // BLOCK_K
        self.warps = tpc * wk
        self.threads = self.warps * WARP_SIZE
        self.tiles = n // TILE_N
        self.grid = self.tiles // tpc * splits
        self.slab_codes = TILE_N * self.ks // 2
        self.slab_scales = TILE_N * self.ks // 16
        self.xrow = self.ks + X_PAD
        self.feats = 8 if swiglu else 16
        self.n_out = n // 2 if swiglu else n
        self.pf_chunk = round_up(cdiv(prefetch, self.grid), 16) if prefetch else 0
        self.cluster = {"cluster": (splits, 1, 1)} if splits > 1 else {}

    @staticmethod
    def unsupported(n: int, k: int, shape: tuple[int, int, int]) -> str | None:
        splits, tpc, wk = shape
        if n % (TILE_N * tpc) or tpc % splits:
            return f"N={n}: tiles of {TILE_N} in groups of {tpc}"
        if k % (splits * wk * 2 * BLOCK_K):
            return f"K={k}: {splits} x {wk} pieces of even 64-K blocks"
        if tpc * wk * WARP_SIZE > 1024:
            return "more than 1024 threads"
        slab = TILE_N * (k // splits) // 2
        if slab > COPY_CHUNK and slab % COPY_CHUNK:
            return f"a {slab}-byte weight slab"
        return None

    def smem_bytes(self) -> int:
        return (
            self.T * (self.slab_codes + self.slab_scales)
            + MAX_ROWS * self.xrow * 2
            + self.T * self.WK * MAX_ROWS * TILE_N * 4
            + self.T * MAX_ROWS * TILE_N * 4
            + 8 * self.T
            + 1024
        )

    @cute.jit
    def __call__(
        self,
        codes: cute.Tensor,
        scales: cute.Tensor,
        consts: cute.Tensor,
        x: cute.Tensor,
        out: cute.Tensor,
        pf: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(codes, scales, consts, x, out, pf, m).launch(
            grid=(self.grid, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
            **self.cluster,
        )

    @cute.jit
    def copy(
        self, source: Int64, nbytes: cutlass.Constexpr[int], dst: Int32, bar: Int32
    ):
        chunk = min(nbytes, COPY_CHUNK)
        for c in cutlass.range_constexpr(nbytes // chunk):
            _bulk_g2s(dst + c * chunk, source + c * chunk, Int32(chunk), bar)

    @cute.kernel
    def kernel(
        self,
        codes: cute.Tensor,
        scales: cute.Tensor,
        consts: cute.Tensor,
        x: cute.Tensor,
        out: cute.Tensor,
        pf: cute.Tensor,
        m: Int32,
    ) -> None:
        cta, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        g = lane // 4
        q = lane % 4
        split = cta % self.S
        group = cta // self.S
        per_tile = MAX_ROWS * TILE_N
        owned = self.T // self.S

        smem = cutlass.memory.SmemAllocator()
        weights = smem.allocate_array(
            cutlass.Uint8, self.T * self.slab_codes, byte_alignment=128
        )
        block_scales = smem.allocate_array(
            cutlass.Uint8, self.T * self.slab_scales, byte_alignment=16
        )
        xs = smem.allocate_array(
            cutlass.Uint8, MAX_ROWS * self.xrow * 2, byte_alignment=128
        )
        partials = smem.allocate_array(
            Float32, self.T * self.WK * per_tile, byte_alignment=16
        )
        cross = smem.allocate_array(Float32, self.T * per_tile, byte_alignment=16)
        bars = smem.allocate_array(Int64, self.T, byte_alignment=8)
        w_base = Int32(weights.toint())
        s_base = Int32(block_scales.toint())
        x_base = Int32(xs.toint())
        cross_base = Int32(cross.toint())
        bar_base = Int32(bars.toint())
        if cutlass.const_expr(self.S > 1):
            cute.arch.cluster_arrive_relaxed()

        if tidx == 0:
            for t in cutlass.range_constexpr(self.T):
                cute.arch.mbarrier_init(shared_ptr(Int64, bar_base + t * 8), 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if tidx == 0:
            for t in cutlass.range_constexpr(self.T):
                bar = bar_base + t * 8
                cute.arch.mbarrier_arrive_and_expect_tx(
                    shared_ptr(Int64, bar), Int32(self.slab_codes + self.slab_scales)
                )
                slab = Int64(cta) * self.T + t
                self.copy(
                    Int64(codes.iterator.toint()) + slab * self.slab_codes,
                    self.slab_codes,
                    w_base + t * self.slab_codes,
                    bar,
                )
                self.copy(
                    Int64(scales.iterator.toint()) + slab * self.slab_scales,
                    self.slab_scales,
                    s_base + t * self.slab_scales,
                    bar,
                )
            if cutlass.const_expr(self.prefetch > 0):
                start = Int32(cta) * self.pf_chunk
                nbytes = Int32(self.pf_chunk)
                if start + nbytes > Int32(self.prefetch):
                    nbytes = Int32(self.prefetch) - start
                if nbytes > Int32(0):
                    bulk_prefetch_l2(Int64(pf.iterator.toint()) + Int64(start), nbytes)
        scale_a = cute.arch.load(consts.iterator.llvm_ptr, Float32)
        scale_b = cute.arch.load((consts.iterator + 1).llvm_ptr, Float32)
        x_scale = cute.arch.load((consts.iterator + 2).llvm_ptr, Float32)
        out_scale = cute.arch.load((consts.iterator + 3).llvm_ptr, Float32)

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        # x rows < m over this split's columns, as F16 rows of `xrow`: all loads
        # first (predicated), then the stores.
        chunks = self.ks // 8
        iters = cdiv(MAX_ROWS * chunks, self.threads)
        staged = cute.make_rmem_tensor(cute.make_layout((iters, 4)), Uint32)
        for it in cutlass.range_constexpr(iters):
            c = tidx + it * self.threads
            row = c // chunks
            element = (
                Int64(row) * self.k + Int64(split) * self.ks + Int64((c % chunks) * 8)
            )
            pointer = cute.make_ptr(
                BFloat16 if self.swiglu else Float16,
                (x.iterator + element).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            valid = Int32(1)
            if row >= m:
                valid = Int32(0)
            words = load_global_u32x4(pointer, valid)
            for i in cutlass.range_constexpr(4):
                staged[it, i] = words[i]
        for it in cutlass.range_constexpr(iters):
            c = tidx + it * self.threads
            row = c // chunks
            if row < m:
                converted = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for i in cutlass.range_constexpr(4):
                    if cutlass.const_expr(self.swiglu):
                        converted[i] = _bf16x2_to_f16x2(staged[it, i], x_scale)
                    else:
                        converted[i] = staged[it, i]
                store_shared_u32x4_address(
                    x_base + (row * self.xrow + (c % chunks) * 8) * 2,
                    (converted[0], converted[1], converted[2], converted[3]),
                )
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        tile_local = warp // self.WK
        piece = warp % self.WK
        mbarrier_wait_address(bar_base + tile_local * 8, Int32(0))
        code_base = (
            w_base
            + tile_local * self.slab_codes
            + (piece * self.blocks * WARP_SIZE + lane) * 16
        )
        scale_base = (
            s_base
            + tile_local * self.slab_scales
            + (piece * WARP_SIZE + lane) * (2 * self.blocks)
        )
        x_lane = x_base + (g * self.xrow + piece * self.kw + q * 16) * 2
        acc = cute.make_rmem_tensor(cute.make_layout((ACCS, 4)), Float32)
        for a in cutlass.range_constexpr(ACCS):
            for i in cutlass.range_constexpr(4):
                acc[a, i] = Float32(0.0)
        for pair in cutlass.range_constexpr(self.blocks // 2):
            scale_word = cute.arch.load(
                shared_ptr(Uint32, scale_base + pair * 4), Uint32
            )
            for half_pair in cutlass.range_constexpr(2):
                b = pair * 2 + half_pair
                raw = load_shared_u32x4_address(code_base + b * WARP_SIZE * 16)
                x0 = load_shared_u32x4_address(x_lane + b * BLOCK_K * 2)
                x1 = load_shared_u32x4_address(x_lane + (b * BLOCK_K + 8) * 2)
                w = cute.make_rmem_tensor(cute.make_layout((2, 8)), Uint32)
                for h in cutlass.range_constexpr(2):
                    dq = _dequant_e2m1x16_f16(
                        raw[2 * h],
                        raw[2 * h + 1],
                        scale_word >> Uint32(8 * (half_pair * 2 + h)),
                    )
                    for i in cutlass.range_constexpr(8):
                        w[h, i] = dq[i]
                xw = (x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3])
                for j in cutlass.range_constexpr(4):
                    slot = (b * 4 + j) % ACCS
                    d = mma_m16n8k16_f16(
                        (w[0, 2 * j], w[1, 2 * j], w[0, 2 * j + 1], w[1, 2 * j + 1]),
                        xw[2 * j],
                        xw[2 * j + 1],
                        (acc[slot, 0], acc[slot, 1], acc[slot, 2], acc[slot, 3]),
                    )
                    for i in cutlass.range_constexpr(4):
                        acc[slot, i] = d[i]
        for i in cutlass.range_constexpr(4):
            total = acc[0, i]
            for a in cutlass.range_constexpr(1, ACCS):
                total = total + acc[a, i]
            feature = g + (i // 2) * 8
            row = q * 2 + (i % 2)
            offset = (
                (tile_local * self.WK + piece) * MAX_ROWS + row
            ) * TILE_N + feature
            cute.arch.store((partials + offset).llvm_ptr, total)
        cute.arch.barrier()

        # The WK pieces summed in order, sent to the tile's owner.
        if cutlass.const_expr(self.S > 1):
            cute.arch.cluster_wait()
        for it in cutlass.range_constexpr(cdiv(self.T * per_tile, self.threads)):
            idx = tidx + it * self.threads
            if idx < self.T * per_tile:
                tl = idx // per_tile
                rem = idx % per_tile
                value = Float32(0.0)
                for p in cutlass.range_constexpr(self.WK):
                    value = value + cute.arch.load(
                        (partials + (tl * self.WK + p) * per_tile + rem).llvm_ptr,
                        Float32,
                    )
                slot = ((split * owned + tl // self.S) * per_tile + rem) * 4
                if cutlass.const_expr(self.S > 1):
                    store_shared_cluster_f32(
                        map_shared_address_to_peer(cross_base + slot, tl % self.S),
                        value,
                    )
                else:
                    cute.arch.store((cross + slot // 4).llvm_ptr, value)
        if cutlass.const_expr(self.S > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # The owner: splits summed in rank order, scaled, cast, stored.
        outs = owned * MAX_ROWS * self.feats
        for it in cutlass.range_constexpr(cdiv(outs, self.threads)):
            idx = tidx + it * self.threads
            tl = idx // (MAX_ROWS * self.feats)
            rem = idx % (MAX_ROWS * self.feats)
            row = rem // self.feats
            f = rem % self.feats
            if idx < outs and row < m:
                lo = Float32(0.0)
                hi = Float32(0.0)
                for src in cutlass.range_constexpr(self.S):
                    base = (src * owned + tl) * per_tile + row * TILE_N + f
                    lo = lo + cute.arch.load((cross + base).llvm_ptr, Float32)
                    if cutlass.const_expr(self.swiglu):
                        hi = hi + cute.arch.load((cross + base + 8).llvm_ptr, Float32)
                value = lo * scale_a
                if cutlass.const_expr(self.swiglu):
                    value = silu_f32(value) * (hi * scale_b)
                value = value * out_scale
                tile = group * self.T + tl * self.S + split
                column = Int64(tile) * self.feats + Int64(f)
                address = Int64(
                    (out.iterator + (Int64(row) * self.n_out + column)).toint()
                )
                if cutlass.const_expr(self.swiglu):
                    cute.arch.store(
                        global_ptr(Uint16, address), Uint16(_f32_to_f16_bits(value))
                    )
                else:
                    cute.arch.store(
                        global_ptr(Uint16, address), Uint16(f32_to_bf16_bits(value))
                    )


@functools.cache
def compile_w4a16(
    n: int,
    k: int,
    shape: tuple[int, int, int],
    swiglu: bool,
    prefetch: int,
    enable_pdl: bool,
):
    """The compiled kernel of one shape (before CUDA graph capture)."""
    assert_not_capturing("the draft's W4A16 MLP")
    kernel = _W4A16Kernel(
        n=n, k=k, shape=shape, swiglu=swiglu, prefetch=prefetch, enable_pdl=enable_pdl
    )
    n_out = n // 2 if swiglu else n
    return cute.compile(
        kernel,
        make_fake_compact_tensor(cutlass.Uint8, (n * k // 2,), assumed_align=16),
        make_fake_compact_tensor(cutlass.Uint8, (n * k // 16,), assumed_align=16),
        make_fake_compact_tensor(Float32, (4,), assumed_align=16),
        make_fake_dynamic_compact_tensor(
            BFloat16 if swiglu else Float16, alignment=16, divisibility=k
        ),
        make_fake_dynamic_compact_tensor(
            Float16 if swiglu else BFloat16, alignment=16, divisibility=n_out
        ),
        make_fake_compact_tensor(cutlass.Uint8, (max(prefetch, 16),), assumed_align=16),
        Int32(1),
        current_cu_stream(),
    )


def pack_rows(
    codes: torch.Tensor, block_scales: torch.Tensor, shape: tuple[int, int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Codes `[N, K/2]` and linear E4M3 block scales `[N, K/16]` (uint8) in the
    kernel's order: per CTA (row group, split), per tile, per K piece, per
    64-K block, per lane 16 code bytes (rows g, g + 8 of scale group q), and
    per lane its `2 x blocks` scale bytes."""
    splits, tpc, wk = shape
    n, k = codes.shape[0], codes.shape[1] * 2
    blocks = k // splits // wk // BLOCK_K
    groups = n // TILE_N // tpc
    packed = codes.reshape(groups, tpc, 2, 8, splits, wk, blocks, 4, 8)
    packed = packed.permute(0, 4, 1, 5, 6, 3, 7, 2, 8)
    sf = block_scales.reshape(groups, tpc, 2, 8, splits, wk, blocks, 4)
    sf = sf.permute(0, 4, 1, 5, 3, 7, 6, 2)
    return packed.contiguous().view(-1), sf.contiguous().view(-1)


def interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Rows of `gate` and `up` (`[I, C]` each) as `[2 I, C]`: per 8 rows, 8 gate
    rows then the matching 8 up rows."""
    rows = gate.shape[0]
    return torch.stack(
        [gate.reshape(rows // 8, 8, -1), up.reshape(rows // 8, 8, -1)], 1
    ).reshape(2 * rows, -1)


def power_of_two_scale(amax: float) -> float:
    """The power of two (at most 1) that keeps `amax` below F16_HEADROOM."""
    if not math.isfinite(amax) or amax <= 0:
        raise ValueError(f"amax must be positive and finite, not {amax}")
    return 2.0 ** -max(0, math.ceil(math.log2(amax / F16_HEADROOM)))


class DraftW4A16Mlp:
    """A dense SwiGLU pair (gate_up `[2I, H]` = gate then up, down `[H, I]`,
    BF16) as NVFP4 weights (per-tensor global scales) run with F16
    activations: `run` takes contiguous BF16 rows `[m, H]`, 1 <= m <= 8, and
    returns BF16 `[m, H]` (one buffer, reused).
    `input_amax`, `inter_amax` set the power-of-two input and intermediate
    scales. Compiled at construction (before CUDA graph capture)."""

    @staticmethod
    def unsupported(hidden: int, inter: int) -> str | None:
        return _W4A16Kernel.unsupported(
            2 * inter, hidden, GATE_UP_SHAPE
        ) or _W4A16Kernel.unsupported(hidden, inter, DOWN_SHAPE)

    def __init__(
        self,
        gate_up: torch.Tensor,
        down: torch.Tensor,
        input_amax: float,
        inter_amax: float,
        max_tokens: int,
    ) -> None:
        two_i, hidden = gate_up.shape
        inter = two_i // 2
        if tuple(down.shape) != (hidden, inter):
            raise ValueError("Unexpected dense pair shapes.")
        reason = self.unsupported(hidden, inter)
        if reason is not None:
            raise ValueError(f"W4A16 draft MLP: {reason}")
        if not 1 <= max_tokens <= MAX_ROWS:
            raise ValueError(f"max_tokens must be 1..{MAX_ROWS}")
        from ...runtime.quantize_experts import quantize_nvfp4_weight

        device = gate_up.device
        q_gate, sf_gate, s_gate = quantize_nvfp4_weight(gate_up[:inter].contiguous())
        q_up, sf_up, s_up = quantize_nvfp4_weight(gate_up[inter:].contiguous())
        q_down, sf_down, s_down = quantize_nvfp4_weight(down.contiguous())
        sf_gate, sf_up, sf_down = (
            s.view(torch.uint8) for s in (sf_gate, sf_up, sf_down)
        )
        self.gate_up_codes, self.gate_up_scales = pack_rows(
            interleave_gate_up(q_gate, q_up),
            interleave_gate_up(sf_gate, sf_up),
            GATE_UP_SHAPE,
        )
        down_codes, down_scales = pack_rows(q_down, sf_down, DOWN_SHAPE)
        # One buffer, so the gate/up kernel prefetches all of it into L2.
        self.down_weight = torch.cat([down_codes, down_scales])
        self.down_codes = self.down_weight[: down_codes.numel()]
        self.down_scales = self.down_weight[down_codes.numel() :]
        x_scale = power_of_two_scale(input_amax)
        inter_scale = power_of_two_scale(inter_amax)
        # x is staged times x_scale: the gate/up scales divide it out before SiLU.
        self.gate_up_consts = torch.tensor(
            [s_gate.item() / x_scale, s_up.item() / x_scale, x_scale, inter_scale],
            dtype=torch.float32,
            device=device,
        )
        self.down_consts = torch.tensor(
            [s_down.item() / inter_scale, 0.0, 1.0, 1.0],
            dtype=torch.float32,
            device=device,
        )
        self.hidden, self.inter, self.max_tokens = hidden, inter, max_tokens
        pdl = pdl_supported()
        self.gate_up = compile_w4a16(
            two_i, hidden, GATE_UP_SHAPE, True, self.down_weight.numel(), pdl
        )
        self.down = compile_w4a16(hidden, inter, DOWN_SHAPE, False, 0, pdl)
        self.intermediate = torch.empty(
            (max_tokens, inter), dtype=torch.float16, device=device
        )
        self.out = torch.empty(
            (max_tokens, hidden), dtype=torch.bfloat16, device=device
        )
        self._views = (
            to_cute(self.gate_up_codes, 16),
            to_cute(self.gate_up_scales, 16),
            to_cute(self.gate_up_consts, 16),
            to_cute(self.down_weight, 16),
            to_cute(self.down_codes, 16),
            to_cute(self.down_scales, 16),
            to_cute(self.down_consts, 16),
        )

    def takes(self, x: torch.Tensor) -> bool:
        return (
            x.dim() == 2
            and 1 <= x.shape[0] <= self.max_tokens
            and x.shape[1] == self.hidden
            and x.dtype == torch.bfloat16
        )

    def run(self, x: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        x = x.contiguous()
        check_operand(x, (m, self.hidden), torch.bfloat16, 16, "x")
        codes1, scales1, consts1, down_weight, codes2, scales2, consts2 = self._views
        inter = self.intermediate[:m]
        out = self.out[:m]
        stream = current_cu_stream()
        self.gate_up(
            codes1,
            scales1,
            consts1,
            to_cute_dynamic(x.view(-1), 16, divisibility=self.hidden),
            to_cute_dynamic(inter.view(-1), 16, divisibility=self.inter),
            down_weight,
            Int32(m),
            stream,
        )
        self.down(
            codes2,
            scales2,
            consts2,
            to_cute_dynamic(inter.view(-1), 16, divisibility=self.inter),
            to_cute_dynamic(out.view(-1), 16, divisibility=self.hidden),
            down_weight,
            Int32(m),
            stream,
        )
        return out
