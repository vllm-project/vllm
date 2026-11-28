# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark router logits on tensor cores: the FP32 router weight as two
BF16 planes, hi = bf16(W) and lo = bf16(W - hi), on an mma.sync grid beside
the pooler GEMM.

`MmaRouter.partials(x)` is the FP32 K-split partials of the logits
x [m, 8192] @ W^T for 1 <= m <= 8, which the routing sums (`routing.route`
with `splits=SPLITS`): 32 MMA tiles x 4 K splits, 4 warps per CTA, 512 K per
warp. Tile
  t holds the hi plane of experts 8t..8t+7 in its rows 0..7 and their lo plane
  in rows 8..15, so a thread adds an expert's two products in FP32 (never
  rounded to BF16 apart). The weights load before `griddepcontrol.wait`, K
  block b at `PACE_NS` x b after launch (the 8 MB at once slows the all-reduce
  the front overlaps), the activation after the wait. Per thread the 4
  accumulator slots sum in order, then the 4 warps in warp order through
  shared memory, into FP32 split partials.

The logits are not bit-identical to the FP32 router kernels (the planes drop
the weight's last 8 mantissa bits and the sums run in another order); the top-8
selections they give are qualified separately.
"""

import functools
from collections.abc import Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..common.policy import POOLER
from ..common.primitives import (
    add_rn,
    current_cu_stream,
    load_global_u32x4,
    paced_ready,
    to_cute,
)
from ..draft.fp4_mlp import interleave_gate_up

HIDDEN, EXPERTS = 8192, 256
MAX_TOKENS = 8
TILES = 2 * EXPERTS // 16
# The pooler GEMM's K split, warp K and accumulators. Weight pacing is NOT
# aliased from the pooler: the phase-2 microbench (slop-loop iter7) measured
# pace_ns has no protective effect on the concurrent all-reduce (interference
# identical at 0 and 300) and costs ~2.4us/call of pure slowdown.
SPLITS, K_PER_WARP, ACCS = (
    POOLER.splits,
    POOLER.kw,
    POOLER.accs,
)
PACE_NS = 0
WARPS = HIDDEN // SPLITS // K_PER_WARP
BLOCKS = K_PER_WARP // 64
PARTIAL_ELEMENTS = SPLITS * MAX_TOKENS * EXPERTS


def _weight_unsupported(weight: torch.Tensor) -> str | None:
    if weight.shape != (EXPERTS, HIDDEN) or weight.dtype != torch.float32:
        return (
            f"router weight must be fp32 [{EXPERTS}, {HIDDEN}], got "
            f"{weight.dtype} {tuple(weight.shape)}"
        )
    return None


def prepack(weight: torch.Tensor) -> torch.Tensor:
    """The BF16 [512, 8192] bank of the FP32 router weight [256, 8192]: tile t
    (rows 16t..16t+15) is hi = bf16(W[8t + r]) in rows r = 0..7 and
    lo = bf16(W[8t + r] - hi) in rows 8..15."""
    reason = _weight_unsupported(weight)
    if reason is not None:
        raise ValueError(reason)
    hi = weight.to(torch.bfloat16)
    lo = (weight - hi.float()).to(torch.bfloat16)
    return interleave_gate_up(hi, lo)


class RouterFront:
    def __init__(self, enable_pdl: bool):
        self.enable_pdl = enable_pdl

    @cute.kernel
    def kernel(self, bank: cute.Tensor, x: cute.Tensor, part: cute.Tensor, m: Int32):
        tile, split, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        # MMA fragment coordinates: g the A row (feature) and the B column
        # (token), q the K pair.
        g, q = lane // 4, lane % 4
        smem = cutlass.memory.SmemAllocator()
        red = smem.allocate_array(Float32, WARPS * MAX_TOKENS * 16)
        k_lane = (
            Int64(split) * (HIDDEN // SPLITS) + Int64(warp) * K_PER_WARP + Int64(q * 16)
        )
        start = cute.arch.globaltimer()
        w = cute.make_rmem_tensor(cute.make_layout((BLOCKS, 2, 8)), Uint32)
        for b in cutlass.range_constexpr(BLOCKS):
            ready = paced_ready(start, b, PACE_NS)
            for h in cutlass.range_constexpr(2):
                row = Int64(tile * 16 + h * 8) + Int64(g)
                for v in cutlass.range_constexpr(2):
                    words = load_global_u32x4(
                        _gmem_bf16(bank, row * HIDDEN + k_lane + (b * 64 + v * 8)),
                        ready,
                    )
                    for i in cutlass.range_constexpr(4):
                        w[b, h, v * 4 + i] = words[i]
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        xs = cute.make_rmem_tensor(cute.make_layout((BLOCKS, 8)), Uint32)
        for b in cutlass.range_constexpr(BLOCKS):
            for i in cutlass.range_constexpr(8):
                xs[b, i] = Uint32(0)
        if g < m:
            for b in cutlass.range_constexpr(BLOCKS):
                for v in cutlass.range_constexpr(2):
                    words = load_global_u32x4(
                        _gmem_bf16(x, Int64(g) * HIDDEN + k_lane + (b * 64 + v * 8))
                    )
                    for i in cutlass.range_constexpr(4):
                        xs[b, v * 4 + i] = words[i]
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        acc = cute.make_rmem_tensor(cute.make_layout((ACCS, 4)), Float32)
        for a in cutlass.range_constexpr(ACCS):
            for i in cutlass.range_constexpr(4):
                acc[a, i] = Float32(0.0)
        # The fragments are filled by hand: K is permuted per lane (q * 16).
        mma = cute.make_mma_atom(
            cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        )
        tiled = cute.make_tiled_mma(mma)
        a = cute.make_rmem_tensor(tiled.partition_shape_A((16, 16)), BFloat16)
        b_frag = cute.make_rmem_tensor(tiled.partition_shape_B((8, 16)), BFloat16)
        c = cute.make_rmem_tensor(tiled.partition_shape_C((16, 8)), Float32)
        a_words = cute.recast_tensor(a, Uint32)
        b_words = cute.recast_tensor(b_frag, Uint32)
        for b in cutlass.range_constexpr(BLOCKS):
            for j in cutlass.range_constexpr(4):
                s = (b * 4 + j) % ACCS
                a_words[0] = w[b, 0, 2 * j]
                a_words[1] = w[b, 1, 2 * j]
                a_words[2] = w[b, 0, 2 * j + 1]
                a_words[3] = w[b, 1, 2 * j + 1]
                b_words[0] = xs[b, 2 * j]
                b_words[1] = xs[b, 2 * j + 1]
                for i in cutlass.range_constexpr(4):
                    c[i] = acc[s, i]
                cute.gemm(mma, c, a, b_frag, c)
                for i in cutlass.range_constexpr(4):
                    acc[s, i] = c[i]
        total = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        for i in cutlass.range_constexpr(4):
            t = acc[0, i]
            for a in cutlass.range_constexpr(1, ACCS):
                t = add_rn(t, acc[a, i])
            total[i] = t
        # Values 0, 1 are feature g (hi), 2, 3 feature g + 8 (its lo).
        total[0] = add_rn(total[0], total[2])
        total[1] = add_rn(total[1], total[3])
        for i in cutlass.range_constexpr(4):
            feature = g + (i // 2) * 8
            token = q * 2 + (i % 2)
            cute.arch.store(
                (red + (warp * MAX_TOKENS + token) * 16 + feature).llvm_ptr, total[i]
            )
        cute.arch.barrier()
        token = tidx // 16
        feature = tidx % 16
        v = Float32(0.0)
        for wi in cutlass.range_constexpr(WARPS):
            v = add_rn(
                v,
                cute.arch.load(
                    (red + (wi * MAX_TOKENS + token) * 16 + feature).llvm_ptr, Float32
                ),
            )
        if feature < 8:
            part[Int64(split * MAX_TOKENS + token) * EXPERTS + tile * 8 + feature] = v

    @cute.jit
    def launch(
        self,
        bank: cute.Tensor,
        x: cute.Tensor,
        part: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(bank, x, part, m).launch(
            grid=(TILES, SPLITS, 1),
            block=(WARPS * 32, 1, 1),
            smem=WARPS * MAX_TOKENS * 16 * 4,
            stream=stream,
            use_pdl=self.enable_pdl,
        )


@cute.jit
def _gmem_bf16(base: cute.Tensor, offset: Int64):
    return cute.make_ptr(
        BFloat16,
        (base.iterator + offset).llvm_ptr,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )


def _flat(tensor: torch.Tensor) -> cute.Tensor:
    return to_cute(tensor.view(-1), 16)


@functools.cache
def _compiled(enable_pdl: bool):
    """The front, once per process: shapes and m are runtime arguments."""
    device = torch.accelerator.current_device_index()
    bank = torch.zeros(2 * EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device)
    x = torch.zeros(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
    part = torch.zeros(PARTIAL_ELEMENTS, dtype=torch.float32, device=device)
    return cute.compile(
        RouterFront(enable_pdl).launch,
        _flat(bank), _flat(x), _flat(part), Int32(MAX_TOKENS), current_cu_stream(),
    )  # fmt: skip


class MmaRouter:
    """One MoE layer's router on tensor cores, prepared at build (before CUDA
    graph capture): the prepacked bank, the `splits` K-split partials and the
    front compiled. `partials` is graph-capturable and allocates nothing.
    """

    splits = SPLITS

    def __init__(self, weight: torch.Tensor, *, enable_pdl: bool = True) -> None:
        if torch.cuda.get_device_capability(weight.device)[0] != 10:
            raise RuntimeError("the MMA router needs an SM100-family GPU")
        self.bank = prepack(weight)
        device = weight.device
        self.part = torch.empty(PARTIAL_ELEMENTS, dtype=torch.float32, device=device)
        self._views = (_flat(self.bank), _flat(self.part))
        self._front = _compiled(enable_pdl)

    def partials(self, x: torch.Tensor) -> torch.Tensor:
        """The FP32 split partials of the BF16 `x` [m, 8192] (1 <= m <= 8, a
        contiguous 16-byte aligned tensor) as `[0, :m]` of their
        `[SPLITS, 8, 256]` buffer, for `routing.route(..., splits=SPLITS)`,
        which sums them from 0.0 in split order; valid until this router's next
        run."""
        m = int(x.shape[0])
        if (
            not 1 <= m <= MAX_TOKENS
            or tuple(x.shape) != (m, HIDDEN)
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or x.data_ptr() % 16
        ):
            raise ValueError(
                f"x must be contiguous 16-byte aligned bf16 [1..{MAX_TOKENS}, "
                f"{HIDDEN}], got {x.dtype} {tuple(x.shape)}"
            )
        bank, part = self._views
        self._front(bank, _flat(x), part, Int32(m), current_cu_stream())
        return self.split_partials()[0, :m]

    def split_partials(self) -> torch.Tensor:
        """The whole `[SPLITS, 8, 256]` partials buffer."""
        return self.part.view(SPLITS, MAX_TOKENS, EXPERTS)

    @staticmethod
    def unsupported(weight: torch.Tensor, rows: Sequence[int]) -> str | None:
        """Why the MMA router cannot take `weight` for the batches `rows`, or
        None."""
        reason = _weight_unsupported(weight)
        if reason is not None:
            return reason
        if any(not 1 <= m <= MAX_TOKENS for m in rows):
            return f"batches {tuple(rows)} outside 1..{MAX_TOKENS}"
        return None
