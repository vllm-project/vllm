# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dense (first ``first_k_dense_replace``) layers' MLP of a few decode tokens
as two CuTe kernels, in place of input quant, DeepGEMM gate_up, SwiGLU, quant and
DeepGEMM down.

A CTA owns a few 16-row tiles over the whole K, one warp per (tile, 512 of K):
one m16n8k32 FP8 mma.sync per 32 K with the tokens as the eight B columns. A lane
always reads the same 16 bytes of a row (and of a token), so its A and B
fragments come from plain 16-byte loads: within each 64-K step lane (g, q) holds
K 16q..16q+15 of rows g and g + 8, and the two mmas of the step take its first
and last eight bytes. Any K order is fine as long as A and B agree.

The CTA's weights stream into shared memory (cp.async, L2 evict-first) and,
for the first ``in_regs`` steps, into registers before the PDL wait, so a whole
layer's weights are in flight at launch. After the wait the tiles' warps of a K
slice split its tokens and quantize them once for the CTA (FP8 per 128 with the
stock scale rule): every CTA reads and quantizes the whole input, so fewer,
larger CTAs (one per SM) keep that from growing with the tokens. Each 128-K
group's four mmas sum in FP32 and take the weight and activation scales, and the
warps' partial sums meet in shared memory.

The gate_up CTA owns gate rows 8c..8c+7 (A rows g) and the matching up rows
(A rows g + 8), so its epilogue writes the clamped SwiGLU directly, rounding
like stock: BF16 gate/up, then a BF16 product. The down kernel's input is that
BF16 activation.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    copy_evict,
    div_rn,
    load_vector,
    load_words,
    mma,
    pack_fp8,
    store_words,
    unpack_bf16,
)

from .shared_expert import (
    GROUP,
    _group_scale,
    _load_scale,
    _round_bf16,
    _store_bf16_if,
    _store_scale,
    block_scales,
)

# K per warp: 32 lanes x 16 values to quantize, eight 64-K mma steps.
SLICE = 512
STEPS = SLICE // 64
# Tokens are the eight B columns.
MAX_TOKENS = 8


class _MmaGemv:
    """BF16 [tokens, rows] = FP8-quantized BF16 [tokens, k] input times the FP8
    [rows, k] weight; with ``swiglu`` the clamped SwiGLU [tokens, rows / 2] of the
    gate (first half) and up (second half) rows."""

    def __init__(
        self, rows: int, k: int, tokens: int, tiles: int, in_regs: int,
        ctas_per_sm: int, swiglu: bool, limit: float, ue8m0: bool,
    ):  # fmt: skip
        assert k % SLICE == 0 and rows % (16 * tiles) == 0 and 0 <= in_regs <= STEPS
        assert 1 <= tokens <= MAX_TOKENS
        self.rows, self.k, self.tokens, self.in_regs = rows, k, tokens, in_regs
        self.ctas_per_sm, self.swiglu, self.limit = ctas_per_sm, swiglu, limit
        self.ue8m0, self.tiles = ue8m0, tiles
        self.slices = k // SLICE  # warps per tile
        self.warps = self.slices * tiles
        self.threads = 32 * self.warps
        assert 32 * self.slices >= 16 * tokens and self.threads <= 1024
        self.ctas = rows // (16 * tiles)
        self.staged = STEPS - in_regs
        # Weights, then the quantized slices (reused for the partial sums).
        self.weight_words = self.warps * self.staged * 2 * 32 * 4
        self.act_words = max(tokens * k // 4, self.warps * 32 * 4)
        self.smem = (self.weight_words + self.act_words) * 4
        self.smem += tokens * (k // GROUP) * 4 + 64

    @cute.jit
    def __call__(
        self, w: cute.Tensor, s: cute.Tensor, x: cute.Tensor, out: cute.Tensor,
        st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(w, s, x, out).launch(
            grid=(self.ctas, 1, 1), block=(self.threads, 1, 1), smem=self.smem,
            min_blocks_per_mp=self.ctas_per_sm, stream=st, use_pdl=True,
        )  # fmt: skip

    def _row(self, tile, half, g):
        if self.swiglu:
            return tile * 8 + g + half * (self.rows // 2)
        return tile * 16 + half * 8 + g

    def _slot(self, staged, warp, h, step):
        """Shared words of step ``step``, half ``h`` of the warp's weights."""
        return staged + ((warp * self.staged + step - self.in_regs) * 2 + h) * 128

    @cute.kernel
    def kernel(self, w: cute.Tensor, s: cute.Tensor, x: cute.Tensor, out: cute.Tensor):
        k, tokens = self.k, self.tokens
        allocator = cutlass.memory.SmemAllocator()
        staged = allocator.allocate_array(
            Uint32, max(self.weight_words, 4), byte_alignment=16
        )
        act = allocator.allocate_array(Uint32, self.act_words, byte_alignment=16)
        x_scales = allocator.allocate_array(Float32, tokens * k // GROUP)
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        cta, _, _ = cute.arch.block_idx()
        g, q = lane // 4, lane % 4
        # Warp (tile i, slice j) of the CTA's tiles x K slices.
        tile, piece = warp // self.slices, warp % self.slices
        k0 = piece * SLICE
        rows = [Int64(self._row(cta * self.tiles + tile, h, g)) for h in range(2)]

        regs = cute.make_rmem_tensor(
            cute.make_layout((2, max(self.in_regs, 1), 4)), Uint32
        )
        for step in cutlass.range_constexpr(self.in_regs):
            for h in cutlass.range_constexpr(2):
                got = load_vector(
                    w.iterator + rows[h] * k + k0 + step * 64 + q * 16, cache="weight"
                )
                for i in cutlass.range_constexpr(4):
                    regs[h, step, i] = got[i]
        for step in cutlass.range_constexpr(self.in_regs, STEPS):
            for h in cutlass.range_constexpr(2):
                copy_evict(
                    self._slot(staged, warp, h, step) + lane * 4,
                    w.iterator + rows[h] * k + k0 + step * 64 + q * 16,
                )
        cute.arch.cp_async_commit_group()
        groups = SLICE // GROUP
        w_scales = cute.make_rmem_tensor(cute.make_layout((2, groups)), Float32)
        for h in cutlass.range_constexpr(2):
            for j in cutlass.range_constexpr(groups):
                w_scales[h, j] = s[rows[h] // GROUP, k0 // GROUP + j]

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        # The tiles' warps of a K slice share its quantized tokens: tile i's
        # warp takes tokens i, i + tiles, ... Lane l quantizes K k0 + 16l .. + 15;
        # 8 lanes per group.
        rounds = (tokens + self.tiles - 1) // self.tiles
        mine = act + piece * tokens * SLICE // 4
        owned = []
        for j in cutlass.range_constexpr(rounds):
            t = j * self.tiles + tile
            if t >= tokens:
                t = tokens - 1
            owned.append(t)
        loaded = [
            load_words(x.iterator + t * k + k0 + lane * 16, count=4, mutable=True)
            + load_words(x.iterator + t * k + k0 + lane * 16 + 8, count=4, mutable=True)
            for t in owned
        ]
        for j in cutlass.range_constexpr(rounds):
            t = owned[j]
            values = [v for word in loaded[j] for v in unpack_bf16(word)]
            amax = Float32(0.0)
            for v in values:
                amax = cute.arch.fmax(amax, cute.arch.fmax(v, -v))
            for shift in cutlass.range_constexpr(3):
                amax = cute.arch.fmax(
                    amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << shift)
                )
            scale, inverse = _group_scale(amax, self.ue8m0)
            packed = tuple(
                pack_fp8(tuple(values[4 * j + v] * inverse for v in range(4)))
                for j in range(4)
            )
            if j * self.tiles + tile < tokens:
                store_words(mine + t * SLICE // 4 + lane * 4, packed, shared=True)
                if lane % 8 == 0:
                    index = t * (k // GROUP) + k0 // GROUP + lane // 8
                    _store_scale(x_scales, index, scale)
        cute.arch.barrier()

        # B fragments: token g's 16 bytes of each step; columns past the tokens
        # repeat the last one and are never read.
        column = g
        if column >= tokens:
            column = tokens - 1
        b = cute.make_rmem_tensor(cute.make_layout((STEPS, 4)), Uint32)
        for step in cutlass.range_constexpr(STEPS):
            got = load_words(
                mine + column * SLICE // 4 + step * 16 + q * 4, count=4, shared=True
            )
            for i in cutlass.range_constexpr(4):
                b[step, i] = got[i]
        # Activation scales of C columns 2q and 2q + 1.
        x_scale = []
        for c in cutlass.range_constexpr(2):
            column = 2 * q + c
            if column >= tokens:
                column = tokens - 1
            base = column * (k // GROUP) + k0 // GROUP
            x_scale.append([_load_scale(x_scales, base + j) for j in range(groups)])
        cute.arch.cp_async_wait_group(0)

        total = [Float32(0.0)] * 4
        for j in cutlass.range_constexpr(groups):
            acc = (Float32(0.0),) * 4
            for step in cutlass.range_constexpr(2 * j, 2 * j + 2):
                a = []
                for h in cutlass.range_constexpr(2):
                    if cutlass.const_expr(step < self.in_regs):
                        a.append([regs[h, step, i] for i in range(4)])
                    else:
                        a.append(
                            load_words(
                                self._slot(staged, warp, h, step) + lane * 4,
                                count=4,
                                shared=True,
                            )
                        )
                for e in cutlass.range_constexpr(2):
                    acc = mma(
                        (a[0][2 * e], a[1][2 * e], a[0][2 * e + 1], a[1][2 * e + 1]),
                        (b[step, 2 * e], b[step, 2 * e + 1]),
                        acc,
                        fp8=True,
                    )
            for i in cutlass.range_constexpr(4):
                scale = w_scales[i // 2, j] * x_scale[i % 2][j]
                total[i] = total[i] + acc[i] * scale

        # Partial sums over the activation slices (every warp is past them).
        cute.arch.barrier()
        partial = act
        store_words(partial + (warp * 32 + lane) * 4, tuple(total), shared=True)
        cute.arch.barrier()
        self.epilogue(
            partial + tile * self.slices * 128, out, cta * self.tiles + tile,
            tid % (32 * self.slices),
        )  # fmt: skip

    def _summed(self, partial, src, i):
        """Partial sum ``i`` of lane ``src``, over the tile's warps."""
        value = Float32(0.0)
        for v in range(self.slices):
            value = value + _load_scale(partial, (v * 32 + src) * 4 + i)
        return value

    @cute.jit
    def epilogue(self, partial, out, cta, tid):
        """Tile ``cta``'s outputs from its warps' partial sums."""
        tokens = self.tokens
        if cutlass.const_expr(self.swiglu):
            # Element (token t, gate row g): C column t of lane 4g + t / 2.
            t, g = tid // 8, tid % 8
            src = g * 4 + t // 2
            if t < tokens:
                limit = Float32(self.limit)
                gate = cute.arch.fmin(
                    _round_bf16(self._summed(partial, src, t % 2)), limit
                )
                up = _round_bf16(self._summed(partial, src, 2 + t % 2))
                up = cute.arch.fmax(cute.arch.fmin(up, limit), -limit)
                exp = cute.math.exp(-gate, fastmath=False)
                silu = div_rn(gate, Float32(1.0) + exp)
                inter = self.rows // 2
                pointer = out.iterator + t * inter + cta * 8 + g
                _store_bf16_if(pointer, silu * up, 1)
        else:
            t, r = tid // 16, tid % 16
            src = r % 8 * 4 + t // 2
            if t < tokens:
                value = self._summed(partial, src, r // 8 * 2 + t % 2)
                pointer = out.iterator + t * self.rows + cta * 16 + r
                _store_bf16_if(pointer, value, 1)


# (16-row tiles per CTA, weight steps held in registers, CTAs per SM). More
# tiles per CTA share one quantized input among more rows: 128 one-per-SM
# CTAs for both kernels.
GATE_UP_SHAPE = (3, 4, 1)
DOWN_SHAPE = (2, 4, 1)


class DenseMlp:
    """FP8 block-scaled gate_up with the clamped SwiGLU, then the down projection
    of a TP shard of ``Glm5NextMLP`` (output still TP-partial)."""

    def __init__(self, w13, w13_scale, w2, w2_scale, limit: float, ue8m0: bool):
        self.w13, self.w2 = w13.contiguous(), w2.contiguous()
        self.n1, self.k = self.w13.shape
        self.n2, self.inter = self.w2.shape
        assert self.n1 == 2 * self.inter
        self.s13 = (
            w13_scale if w13_scale.dtype == torch.float32
            else block_scales(w13_scale, self.n1, self.k)
        )  # fmt: skip
        self.s2 = (
            w2_scale if w2_scale.dtype == torch.float32
            else block_scales(w2_scale, self.n2, self.inter)
        )  # fmt: skip
        self.limit, self.ue8m0 = limit, ue8m0
        self.compiled: dict = {}

    def prepare(self, tokens: list[int]) -> None:
        """Compiles for these token counts before graph capture."""
        for m in tokens:
            self(torch.zeros(m, self.k, dtype=torch.bfloat16, device=self.w13.device))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        act = torch.empty(m, self.inter, dtype=torch.bfloat16, device=x.device)
        out = torch.empty(m, self.n2, dtype=torch.bfloat16, device=x.device)
        first = (self.w13, self.s13, x, act)
        second = (self.w2, self.s2, act, out)
        st = stream(x.device)
        if m not in self.compiled:
            kernels = (
                _MmaGemv(
                    self.n1,
                    self.k,
                    m,
                    *GATE_UP_SHAPE,
                    True,
                    self.limit,
                    self.ue8m0,
                ),
                _MmaGemv(self.n2, self.inter, m, *DOWN_SHAPE, False, 0.0, self.ue8m0),
            )
            self.compiled[m] = tuple(
                cute.compile(kernel, *(to_cute(t, 16) for t in args), st)
                for kernel, args in zip(kernels, (first, second))
            )
        for compiled, args in zip(self.compiled[m], (first, second)):
            compiled(*(to_cute(t, 16) for t in args), st)
        return out
