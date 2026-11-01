# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared expert of a few decode tokens beside the mega MoE kernel.

The mega kernel's 16-CTA clusters take most of every GPC's SMs, and DeepGEMM's
large-shared-memory CTAs cannot share an SM with them, so the two slow each
other down. These two GEMVs use a few KB of shared memory, at most 128 registers
per thread and the largest carveout, so they run on SMs the clusters already
occupy. They read the DeepGEMM-prepared FP8 weights (UE8M0 block scales, in place)
and follow the stock rounding: FP8 activations per 128 with the stock scale rule,
FP32 sums, BF16 gate/up and SwiGLU outputs.

Each warp loads its weight rows into registers before the PDL wait. After it, each
CTA quantizes the activations once into shared memory, and lane l takes 16-byte
chunk l of every 512 bytes of a row. Products run as mixed-precision FMAs: FP8
weights and activations widen exactly to F16 pairs, which ``fma.rn.f32.f16``
multiplies exactly and adds in FP32, one instruction per product.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    _checked,
    resolve_kernel,
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    asm,
    bits_float,
    div_rn,
    float_bits,
    load_vector,
    load_words,
    pack_fp8,
    store_words,
    unpack_bf16,
    warp_max_abs,
)

GROUP = 128
WARPS = 4
THREADS = 32 * WARPS
FP8_MAX = 448.0
# Four CTAs per SM caps registers at 128 per thread, so a CTA fits beside a mega
# MoE CTA.
CTAS_PER_SM = 4


@asm((Uint32, Uint32), pure=True)
def _f16x4(word):
    """The four E4M3 values of ``word`` (low byte first) as two exact F16x2 words."""
    code = "{ .reg .b16 lo,hi; mov.b32 {lo,hi},$2; "
    code += "cvt.rn.f16x2.e4m3x2 $0,lo; cvt.rn.f16x2.e4m3x2 $1,hi; }"
    return code, "=r,=r,r", (word,)


@asm(Float32, pure=True)
def _fma_f16x2(a, b, c):
    """``c`` plus the two products of the F16x2 words ``a`` and ``b``: each product
    exact, each sum rounded in FP32."""
    code = "{ .reg .b16 al,ah,bl,bh; .reg .f32 t; mov.b32 {al,ah},$1; "
    code += "mov.b32 {bl,bh},$2; fma.rn.f32.f16 t,al,bl,$3; "
    code += "fma.rn.f32.f16 $0,ah,bh,t; }"
    return code, "=f,r,r,f", (a, b, c)


@asm(Float32, pure=True)
def _select(upper, high, low):
    code = "{ .reg .pred p; setp.ne.u32 p,$1,0; selp.f32 $0,$2,$3,p; }"
    return code, "=f,r,f,f", (upper, high, low)


@asm()
def _store_bf16_if(pointer, value, when):
    code = "{ .reg .pred p; .reg .b16 h; setp.ne.u32 p,$2,0; "
    code += "cvt.rn.bf16.f32 h,$1; @p st.global.b16 [$0],h; }"
    return code, "l,f,r", (pointer, value, Uint32(when))


@asm(Float32, pure=True)
def _round_bf16(value):
    code = "{ .reg .b16 h,z; mov.b16 z,0; cvt.rn.bf16.f32 h,$1; mov.b32 $0,{z,h}; }"
    return code, "=f,f", (value,)


def _group_scale(amax, ue8m0: bool):
    """Amax / 448 (amax at least 1e-10), rounded up to a power of two when
    ``ue8m0``, and its IEEE reciprocal."""
    scale = div_rn(cute.arch.fmax(amax, Float32(1e-10)), Float32(FP8_MAX))
    if ue8m0:
        bits = Uint32(float_bits(cute.arch.fmax(scale, Float32(1e-10))))
        scale = bits_float(
            ((bits >> 23) + Uint32((bits & Uint32(0x7FFFFF)) != 0)) << 23
        )
        # 2^-e exactly: the IEEE reciprocal of a power of two.
        return scale, bits_float((254 - (Uint32(float_bits(scale)) >> 23)) << 23)
    return scale, div_rn(Float32(1.0), scale)


def _dot(weights, values, total):
    """``total`` plus the dot product of two lists of F16x2 words."""
    for i in range(len(weights)):
        total = _fma_f16x2(weights[i], values[i], total)
    return total


def _reduce_rows(values, lane):
    """Warp sums of ``len(values)`` (a power of two up to 32) per-lane values:
    returns the sum of value ``(lane >> shift) % len(values)`` and ``shift``.
    Each halving step trades half the values with the partner lane."""
    count = len(values)
    offset = 16
    while count > 1:
        half = count // 2
        upper = lane & offset
        kept = []
        for i in range(half):
            send = _select(upper, values[i], values[i + half])
            keep = _select(upper, values[i + half], values[i])
            kept.append(keep + cute.arch.shuffle_sync_bfly(send, offset=offset))
        values, count, offset = kept, half, offset // 2
    total = values[0]
    shift = 0
    while (1 << shift) <= offset:
        shift += 1
    for s in range(shift):
        total = total + cute.arch.shuffle_sync_bfly(total, offset=offset >> s)
    return total, shift


def _store_scale(pointer, index, value):
    cute.arch.store((pointer + index).llvm_ptr, value)


def _load_scale(pointer, index):
    return cute.arch.load((pointer + index).llvm_ptr, Float32)


class _GateUpKernel:
    """BF16 [tokens, n] = FP8-quantized x times the FP8 [n, k] gate/up weight."""

    def __init__(self, n: int, k: int, tokens: int, rows: int, ue8m0: bool):
        assert k % 512 == 0 and n % (WARPS * rows) == 0
        assert GROUP % (WARPS * rows) == 0 and GROUP % (k // THREADS) == 0
        self.n, self.k, self.tokens, self.rows, self.ue8m0 = n, k, tokens, rows, ue8m0
        self.chunks = k // 512  # 16-byte chunks per lane per row
        self.ctas = n // (WARPS * rows)
        self.smem = tokens * (k + k // GROUP * 4)

    @cute.jit
    def __call__(
        self, w: cute.Tensor, s: cute.Tensor, x: cute.Tensor, out: cute.Tensor,
        st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(w, s, x, out).launch(
            grid=(self.ctas, 1, 1), block=(THREADS, 1, 1), smem=self.smem,
            min_blocks_per_mp=CTAS_PER_SM, stream=st, use_pdl=True,
        )  # fmt: skip

    @cute.kernel
    def kernel(self, w: cute.Tensor, s: cute.Tensor, x: cute.Tensor, out: cute.Tensor):
        k, tokens, rows = self.k, self.tokens, self.rows
        allocator = cutlass.memory.SmemAllocator()
        quantized = allocator.allocate_array(Uint32, tokens * k // 4, byte_alignment=16)
        x_scales = allocator.allocate_array(Float32, tokens * k // GROUP)
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        cta, _, _ = cute.arch.block_idx()
        first = (cta * WARPS + warp) * rows
        words = cute.make_rmem_tensor(cute.make_layout((rows, self.chunks, 4)), Uint32)
        scales = cute.make_rmem_tensor(cute.make_layout((self.chunks,)), Float32)
        for r in cutlass.range_constexpr(rows):
            row = Int64(first + r)
            for c in cutlass.range_constexpr(self.chunks):
                got = load_vector(
                    w.iterator + row * k + c * 512 + lane * 16, cache="weight"
                )
                for i in cutlass.range_constexpr(4):
                    words[r, c, i] = got[i]
        for c in cutlass.range_constexpr(self.chunks):
            scales[c] = s[first // GROUP, c * 4 + lane // 8]

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        # Each thread quantizes `per` values of every token once for the CTA.
        per = k // THREADS
        loaded = [
            [
                word
                for h in range(per // 8)
                for word in load_words(
                    x.iterator + t * k + tid * per + h * 8, count=4, mutable=True
                )
            ]
            for t in range(tokens)
        ]
        for t in cutlass.range_constexpr(tokens):
            values = [v for word in loaded[t] for v in unpack_bf16(word)]
            amax = Float32(0.0)
            for v in values:
                amax = cute.arch.fmax(amax, cute.arch.fmax(v, -v))
            for shift in cutlass.range_constexpr((GROUP // per).bit_length() - 1):
                amax = cute.arch.fmax(
                    amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << shift)
                )
            scale, inverse = _group_scale(amax, self.ue8m0)
            packed = [
                pack_fp8(tuple(values[4 * j + v] * inverse for v in range(4)))
                for j in range(per // 4)
            ]
            for j in cutlass.range_constexpr(per // 16):
                store_words(
                    quantized + (t * k + tid * per) // 4 + 4 * j,
                    packed[4 * j : 4 * j + 4],
                    shared=True,
                )
            if tid % (GROUP // per) == 0:
                _store_scale(x_scales, t * (k // GROUP) + tid // (GROUP // per), scale)
        cute.arch.barrier()

        acc = [[Float32(0.0)] * rows for _ in range(tokens)]
        for c in cutlass.range_constexpr(self.chunks):
            activations, combined = [], []
            for t in cutlass.range_constexpr(tokens):
                got = load_words(
                    quantized + t * k // 4 + c * 128 + lane * 4,
                    count=4, shared=True,
                )  # fmt: skip
                activations.append([h for word in got for h in _f16x4(word)])
                index = t * (k // GROUP) + c * 4 + lane // 8
                combined.append(_load_scale(x_scales, index) * scales[c])
            for r in cutlass.range_constexpr(rows):
                weights = [h for i in range(4) for h in _f16x4(words[r, c, i])]
                for t in cutlass.range_constexpr(tokens):
                    part = _dot(weights, activations[t], Float32(0.0))
                    acc[t][r] = acc[t][r] + part * combined[t]
        _store_rows([a for row in acc for a in row], lane, out, first, self.n, rows)


def _store_rows(values, lane, out, first, n, rows):
    """Warp-sums ``values`` (token-major, ``rows`` per token) and stores each as
    BF16 at ``out[token, first + row]``."""
    total, shift = _reduce_rows(values, lane)
    index = (lane >> shift) % len(values)
    pointer = out.iterator + index // rows * n + first + index % rows
    _store_bf16_if(pointer, total, lane % (1 << shift) == 0)


class _DownKernel:
    """BF16 [tokens, n] = clamped SwiGLU of the BF16 gate/up rows, FP8-quantized,
    times the FP8 [n, 512] down weight."""

    def __init__(
        self, n: int, inter: int, tokens: int, rows: int, limit: float, ue8m0: bool
    ):
        assert inter == 512 and n % (WARPS * rows) == 0
        assert GROUP % (WARPS * rows) == 0 and inter // THREADS * 32 == GROUP
        self.n, self.inter, self.tokens, self.rows = n, inter, tokens, rows
        self.limit, self.ue8m0 = limit, ue8m0
        self.ctas = n // (WARPS * rows)
        self.smem = tokens * (inter + inter // GROUP * 4)

    @cute.jit
    def __call__(
        self, w: cute.Tensor, s: cute.Tensor, gate_up: cute.Tensor, out: cute.Tensor,
        st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(w, s, gate_up, out).launch(
            grid=(self.ctas, 1, 1), block=(THREADS, 1, 1), smem=self.smem,
            min_blocks_per_mp=CTAS_PER_SM, stream=st, use_pdl=True,
        )  # fmt: skip

    @cute.kernel
    def kernel(
        self, w: cute.Tensor, s: cute.Tensor, gate_up: cute.Tensor, out: cute.Tensor
    ):
        inter, tokens, rows = self.inter, self.tokens, self.rows
        allocator = cutlass.memory.SmemAllocator()
        quantized = allocator.allocate_array(
            Uint32, tokens * inter // 4, byte_alignment=16
        )
        h_scales = allocator.allocate_array(Float32, tokens * inter // GROUP)
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        cta, _, _ = cute.arch.block_idx()
        first = (cta * WARPS + warp) * rows
        words = cute.make_rmem_tensor(cute.make_layout((rows, 4)), Uint32)
        for r in cutlass.range_constexpr(rows):
            got = load_vector(
                w.iterator + Int64(first + r) * inter + lane * 16, cache="weight"
            )
            for i in cutlass.range_constexpr(4):
                words[r, i] = got[i]
        w_scale = s[first // GROUP, lane // 8]

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        # Thread i computes and quantizes values 4i..4i+3; warp j owns group j.
        limit = Float32(self.limit)
        loaded = []
        for t in cutlass.range_constexpr(tokens):
            base = gate_up.iterator + t * 2 * inter + tid * 4
            loaded.append(
                (
                    load_words(base, count=2, mutable=True),
                    load_words(base + inter, count=2, mutable=True),
                )
            )
        for t in cutlass.range_constexpr(tokens):
            gates, ups = loaded[t]
            values = []
            for i in cutlass.range_constexpr(2):
                for gate, up in zip(unpack_bf16(gates[i]), unpack_bf16(ups[i])):
                    gate = cute.arch.fmin(gate, limit)
                    up = cute.arch.fmax(cute.arch.fmin(up, limit), -limit)
                    silu = div_rn(
                        gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)
                    )
                    values.append(_round_bf16(silu * up))
            amax = Float32(0.0)
            for v in values:
                amax = cute.arch.fmax(amax, cute.arch.fmax(v, -v))
            scale, inverse = _group_scale(warp_max_abs(amax), self.ue8m0)
            word = pack_fp8(tuple(v * inverse for v in values))
            store_words(quantized + t * inter // 4 + tid, (word,), shared=True)
            if lane == 0:
                _store_scale(h_scales, t * (inter // GROUP) + warp, scale)
        cute.arch.barrier()

        activations, combined = [], []
        for t in cutlass.range_constexpr(tokens):
            got = load_words(
                quantized + t * inter // 4 + lane * 4, count=4, shared=True
            )
            activations.append([h for word in got for h in _f16x4(word)])
            index = t * (inter // GROUP) + lane // 8
            combined.append(_load_scale(h_scales, index) * w_scale)
        partials = [[None] * rows for _ in range(tokens)]
        for r in cutlass.range_constexpr(rows):
            weights = [h for i in range(4) for h in _f16x4(words[r, i])]
            for t in cutlass.range_constexpr(tokens):
                part = _dot(weights, activations[t], Float32(0.0))
                partials[t][r] = part * combined[t]
        _store_rows(
            [p for row in partials for p in row], lane, out, first, self.n, rows
        )


def block_scales(packed: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """FP32 [n / 128, k / 128] block scales from DeepGEMM's packed UE8M0 layout
    (int32 [n, k / 512], byte g % 4 of column g // 4 holds K group g)."""
    groups = k // GROUP
    index = torch.arange(groups, device=packed.device)
    exponents = (packed[:, index // 4] >> (index % 4 * 8)) & 0xFF
    rows = exponents.view(n // GROUP, GROUP, groups)
    assert bool((rows == rows[:, :1]).all()), "scales vary inside a 128-row block"
    return (rows[:, 0].to(torch.int32) << 23).view(torch.float32).contiguous()


def _largest_carveout(compiled, device: int) -> None:
    """An SM configured for a small shared-memory carveout cannot take a mega MoE
    CTA until it drains: ask for the largest one."""
    _, function = resolve_kernel(compiled, device)
    attribute = cuda.CUfunction_attribute
    carveout = attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    _checked((cuda.cuFuncSetAttribute(function, carveout, 100)[0],))


class SharedExpert:
    """FP8 block-scaled gate_up GEMV, then clamped SwiGLU and the down GEMV."""

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
        gate_up = torch.empty(m, self.n1, dtype=torch.bfloat16, device=x.device)
        out = torch.empty(m, self.n2, dtype=torch.bfloat16, device=x.device)
        first = (self.w13, self.s13, x, gate_up)
        second = (self.w2, self.s2, gate_up, out)
        st = stream(x.device)
        if m not in self.compiled:
            kernels = (
                _GateUpKernel(self.n1, self.k, m, 2, self.ue8m0),
                _DownKernel(self.n2, self.inter, m, 8, self.limit, self.ue8m0),
            )
            self.compiled[m] = tuple(
                cute.compile(kernel, *(to_cute(t, 16) for t in args), st)
                for kernel, args in zip(kernels, (first, second))
            )
            for compiled in self.compiled[m]:
                _largest_carveout(compiled, x.device.index)
        for compiled, args in zip(self.compiled[m], (first, second)):
            compiled(*(to_cute(t, 16) for t in args), st)
        return out
