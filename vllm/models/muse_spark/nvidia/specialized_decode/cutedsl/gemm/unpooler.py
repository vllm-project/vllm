# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
# The zero-copy epilogue's publish step (the stage read, the slot offset, the
# multicast store of the sanitized vector and the stage-word store) is adapted
# from FlashInfer's Lamport publish (flashinfer/comm/mnnvl_cutedsl/kernel_ll/
# device_kernels.py and kernel_bt/device_kernels.py,
# flashinfer-ai/flashinfer@14ba7aaa84718abe7a5648184fa9ad15362e2880,
# Apache-2.0); the GEMM itself is not.
# FlashInfer NOTICE: "FlashInfer / Copyright 2025-2026 NVIDIA / Copyright
# 2023-2026 FlashInfer community (https://flashinfer.ai/)".

"""The Muse Spark unpooler GEMM for small M: this rank's column block
y[M, N] = x[M, K] @ W[N, K]^T of the MoE unpooler (N = hidden / tp, K = 4096),
BF16 in and out, FP32 accumulate, 1 <= M <= 16. An optional epilogue also
publishes y into the next boundary's Lamport mailbox (zero-copy publish).

Grid (N / 16, S), cluster (1, S, 1) when S > 1. CTA (t, s) owns output features
[16 t, 16 t + 16) and the K slice [s K / S, (s + 1) K / S); each of its warps
owns K / (S * warps) consecutive K. Operands are loaded straight into registers
and fed to mma.sync.m16n8k16 (A = 16 weight rows, B = 8 tokens).

Weights are loaded before griddepcontrol.wait, activations after it. Each warp's
FP32 partial goes to cluster rank 0's shared memory, which sums the partials in
a fixed order, rounds to BF16 and stores y. The summation order depends only on
the launch shape, so runs are bitwise reproducible.

Zero-copy epilogue (``zerocopy``): rank 0 also multicasts y (-0.0 stored as
+0.0) into this rank's slot of the next boundary's LamportMailbox at stage
NEXT_STAGE and sets ACTIVE_STAGE, the same bytes and stage word the mailbox's
own publisher would write.

Ticket stage (``ticket_stage``, with ``zerocopy``): the stage is instead the
upper 16 bits of the draft boundary's ticket word, and no stage word is stored;
the consuming boundary advances the stage itself.
"""

from collections.abc import Sequence
from functools import cache
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor

from ..common.host import assert_not_capturing
from ..common.mailbox import LamportMailbox
from ..common.policy import UNPOOLER, UnpoolerShape
from ..common.primitives import (
    ACTIVE_STAGE,
    NEXT_STAGE,
    WARP_SIZE,
    current_cu_stream,
    load_global_u32x4,
    load_volatile_u32,
    make_fake_dynamic_compact_tensor,
    map_shared_to_peer,
    mma_m16n8k16_bf16,
    pack_bf16x2_rn,
    sanitize_negative_zero_u32x4,
    shuffle_sync_idx_u32,
    stmc_bf16x8,
    store_global_u32,
    store_global_u32x4,
    store_shared_cluster_f32,
    to_cute,
    to_cute_dynamic,
)

TILE_N = 16  # output features per CTA (the MMA's M)
BLOCK_K = 64  # K per register block: 4 MMA k-steps of 16
MAX_M = 16
TOKEN_TILES = (8, 16)


def launch_unsupported(n: int, k: int, splits: int, kw: int, accs: int) -> str | None:
    """Why the kernel has no launch for an [n, k] weight with K split over
    ``splits`` CTAs, ``kw`` K per warp and ``accs`` accumulators, or None."""
    if n <= 0 or n % TILE_N or splits not in (1, 2, 4, 8) or accs not in (1, 2, 4):
        return f"unsupported shape or tactic n={n} splits={splits} accs={accs}"
    if k % (splits * kw) or kw % BLOCK_K:
        return f"K={k} does not split into {splits} x warps x {kw}"
    if not 1 <= k // (splits * kw) <= 32:
        return f"{k // (splits * kw)} warps per CTA"
    return None


def warps_for(k: int, splits: int, kw: int) -> int:
    if k % (splits * kw) or kw % BLOCK_K:
        raise ValueError(f"K={k} does not split into {splits} x warps x {kw}")
    return k // (splits * kw)


# Numerics: BF16 in, FP32 accumulate, BF16 RN out. Differs from the stock
# unpooler GEMM in summation order only, so it is not bitwise equal to it. The
# zero-copy mailbox bytes are bitwise those of the shared-only publisher.
class UnpoolerKernel:
    def __init__(
        self,
        *,
        n: int,
        k: int,
        splits: int,
        token_tile: int,
        kw: int,
        accs: int,
        enable_pdl: bool = True,
        zerocopy: bool = False,
        tp: int = 1,
        rank: int = 0,
        capacity_m: int = 1,
        hidden_local: int = 0,
        ticket_stage: bool = False,
    ) -> None:
        reason = launch_unsupported(n, k, splits, kw, accs)
        if reason is None and token_tile not in TOKEN_TILES:
            reason = f"unsupported token tile {token_tile}"
        if reason is not None:
            raise ValueError(reason)
        self.n, self.k, self.splits, self.tt = n, k, splits, token_tile
        self.kw, self.accs = kw, accs
        self.warps = warps_for(k, splits, kw)
        self.threads = self.warps * WARP_SIZE
        self.ks = k // splits
        self.blocks = kw // BLOCK_K
        self.n_iters = token_tile // 8
        self.parts = splits * self.warps
        self.enable_pdl = enable_pdl
        self.zerocopy = zerocopy
        if zerocopy and (hidden_local != n or capacity_m < 1 or not 0 <= rank < tp):
            raise ValueError("zero-copy publish needs hidden_local == n and a rank")
        self.tp, self.rank = tp, rank
        self.capacity_m, self.hidden_local = capacity_m, hidden_local
        if ticket_stage and not zerocopy:
            raise ValueError("a ticket stage needs the zero-copy publish")
        self.ticket_stage = ticket_stage

    def smem_bytes(self) -> int:
        return self.parts * self.tt * TILE_N * 4

    @cute.jit
    def __call__(
        self,
        weight: cute.Tensor,
        x: cute.Tensor,
        out: cute.Tensor,
        stage_state: cute.Tensor,
        mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        kernel = self.kernel(weight, x, out, stage_state, mailbox_multicast_address, m)
        if cutlass.const_expr(self.splits > 1):
            kernel.launch(
                grid=(self.n // TILE_N, self.splits, 1),
                block=(self.threads, 1, 1),
                cluster=(1, self.splits, 1),
                smem=self.smem_bytes(),
                stream=stream,
                use_pdl=self.enable_pdl,
            )
        else:
            kernel.launch(
                grid=(self.n // TILE_N, 1, 1),
                block=(self.threads, 1, 1),
                smem=self.smem_bytes(),
                stream=stream,
                use_pdl=self.enable_pdl,
            )

    @cute.kernel
    def kernel(
        self,
        weight: cute.Tensor,
        x: cute.Tensor,
        out: cute.Tensor,
        stage_state: cute.Tensor,
        mailbox_multicast_address: Int64,
        m: Int32,
    ) -> None:
        tile, split, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        g = lane // 4
        q = lane % 4

        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_array(Float32, self.parts * self.tt * TILE_N)
        if cutlass.const_expr(self.splits > 1):
            # Peers store into rank 0's shared memory; the matching wait precedes
            # the first remote store.
            cute.arch.cluster_arrive_relaxed()

        k_lane = Int64(split) * self.ks + Int64(warp) * self.kw + Int64(q * 16)

        # Weight rows g and g + 8 of this tile, 32 bytes per block: independent of
        # the predecessor grid.
        wreg = cute.make_rmem_tensor(cute.make_layout((self.blocks, 2, 8)), Uint32)
        for b in cutlass.range_constexpr(self.blocks):
            for h in cutlass.range_constexpr(2):
                row = Int64(tile * TILE_N + h * 8) + Int64(g)
                for v in cutlass.range_constexpr(2):
                    element = row * self.k + k_lane + (b * BLOCK_K + v * 8)
                    wp = cute.make_ptr(
                        BFloat16,
                        (weight.iterator + element).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    words = load_global_u32x4(wp)
                    for i in cutlass.range_constexpr(4):
                        wreg[b, h, v * 4 + i] = words[i]

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        stage = Uint32(0)
        # Zero-copy: the publishing threads read the stage after the wait; it is
        # used only in the epilogue.
        if cutlass.const_expr(self.zerocopy) and split == 0:
            if cutlass.const_expr(self.ticket_stage):
                stage = load_volatile_u32(stage_state.iterator) >> Uint32(16)
            else:
                stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)

        xreg = cute.make_rmem_tensor(
            cute.make_layout((self.n_iters, self.blocks, 8)), Uint32
        )
        for ni in cutlass.range_constexpr(self.n_iters):
            for b in cutlass.range_constexpr(self.blocks):
                for i in cutlass.range_constexpr(8):
                    xreg[ni, b, i] = Uint32(0)
        for ni in cutlass.range_constexpr(self.n_iters):
            token = g + ni * 8
            if token < m:
                for b in cutlass.range_constexpr(self.blocks):
                    for v in cutlass.range_constexpr(2):
                        element = Int64(token) * self.k + k_lane + (b * BLOCK_K + v * 8)
                        xp = cute.make_ptr(
                            BFloat16,
                            (x.iterator + element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        words = load_global_u32x4(xp)
                        for i in cutlass.range_constexpr(4):
                            xreg[ni, b, v * 4 + i] = words[i]
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        acc = cute.make_rmem_tensor(
            cute.make_layout((self.n_iters, self.accs, 4)), Float32
        )
        for ni in cutlass.range_constexpr(self.n_iters):
            for a in cutlass.range_constexpr(self.accs):
                for i in cutlass.range_constexpr(4):
                    acc[ni, a, i] = Float32(0.0)
        for b in cutlass.range_constexpr(self.blocks):
            for j in cutlass.range_constexpr(4):
                a_slot = (b * 4 + j) % self.accs
                for ni in cutlass.range_constexpr(self.n_iters):
                    d = mma_m16n8k16_bf16(
                        (
                            wreg[b, 0, 2 * j],
                            wreg[b, 1, 2 * j],
                            wreg[b, 0, 2 * j + 1],
                            wreg[b, 1, 2 * j + 1],
                        ),
                        xreg[ni, b, 2 * j],
                        xreg[ni, b, 2 * j + 1],
                        (
                            acc[ni, a_slot, 0],
                            acc[ni, a_slot, 1],
                            acc[ni, a_slot, 2],
                            acc[ni, a_slot, 3],
                        ),
                    )
                    for i in cutlass.range_constexpr(4):
                        acc[ni, a_slot, i] = d[i]

        # This warp's partial into slot (split * warps + warp) of rank 0:
        # [token][feature], FP32.
        part = split * self.warps + warp
        if cutlass.const_expr(self.splits > 1):
            cute.arch.cluster_wait()
            remote = map_shared_to_peer(partials, Int32(0))
        for ni in cutlass.range_constexpr(self.n_iters):
            for i in cutlass.range_constexpr(4):
                total = acc[ni, 0, i]
                for a in cutlass.range_constexpr(1, self.accs):
                    total = total + acc[ni, a, i]
                feature = g + (i // 2) * 8
                token = q * 2 + (i % 2) + ni * 8
                offset = (part * self.tt + token) * TILE_N + feature
                if cutlass.const_expr(self.splits > 1):
                    store_shared_cluster_f32(remote + offset * 4, total)
                else:
                    cute.arch.store((partials + offset).llvm_ptr, total)
        if cutlass.const_expr(self.splits > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        if split == 0:
            if (
                cutlass.const_expr(self.zerocopy and not self.ticket_stage)
                and tile == 0
                and tidx == 0
            ):
                store_global_u32(stage_state.iterator + ACTIVE_STAGE, stage)
            # Lane i of a round owns token idx // 8, features 2 (idx % 8) and
            # 2 (idx % 8) + 1; lanes 4u..4u+3 hold one token's 8-feature half,
            # gathered into lane 4u for one 16-byte store.
            pairs = self.tt * (TILE_N // 2)
            for rnd in cutlass.range_constexpr(
                (pairs + self.threads - 1) // self.threads
            ):
                idx = tidx + rnd * self.threads
                token = idx // 8
                j = idx % 8
                word = Uint32(0)
                if token < self.tt:
                    lo = Float32(0.0)
                    hi = Float32(0.0)
                    for p in cutlass.range_constexpr(self.parts):
                        base = (p * self.tt + token) * TILE_N + j * 2
                        lo = lo + cute.arch.load((partials + base).llvm_ptr, Float32)
                        hi = hi + cute.arch.load(
                            (partials + base + 1).llvm_ptr, Float32
                        )
                    word = pack_bf16x2_rn(lo, hi)
                lead = lane - q
                packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for i in cutlass.range_constexpr(4):
                    packed[i] = shuffle_sync_idx_u32(word, lead + i)
                if q == 0 and token < m:
                    column = Int64(tile * TILE_N) + Int64((j // 4) * 8)
                    destination = out.iterator + (Int64(token) * self.n + column)
                    store_global_u32x4(Int64(destination.toint()), packed.load())
                    if cutlass.const_expr(self.zerocopy):
                        element = (
                            (Int64(stage) * self.tp + self.rank) * self.capacity_m
                            + Int64(token)
                        ) * self.hidden_local + column
                        stmc_bf16x8(
                            mailbox_multicast_address + element * 2,
                            sanitize_negative_zero_u32x4(packed.load()),
                        )


@cache
def compile_unpooler(
    n: int,
    k: int,
    token_tile: int,
    *,
    splits: int,
    kw: int,
    accs: int,
    enable_pdl: bool = True,
    zerocopy: bool = False,
    tp: int = 1,
    rank: int = 0,
    capacity_m: int = 1,
    hidden_local: int = 0,
    ticket_stage: bool = False,
):
    """The compiled kernel for weight [n, k] and one token tile; called as
    fn(weight, x, out, stage_state, mailbox_multicast_address, m, stream) with
    flat views (``Unpooler`` binds them; ``stage_state`` is the ticket word
    with ``ticket_stage``). Shared by every layer of one shape."""
    assert_not_capturing("the unpooler GEMM")
    kernel = UnpoolerKernel(
        n=n,
        k=k,
        splits=splits,
        token_tile=token_tile,
        kw=kw,
        accs=accs,
        enable_pdl=enable_pdl,
        zerocopy=zerocopy,
        tp=tp,
        rank=rank,
        capacity_m=capacity_m,
        hidden_local=hidden_local,
        ticket_stage=ticket_stage,
    )
    return cute.compile(
        kernel,
        make_fake_compact_tensor(BFloat16, (n * k,), assumed_align=16),
        make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=k),
        make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=n),
        make_fake_compact_tensor(Int32, (1 if ticket_stage else 2,), assumed_align=4),
        Int64(0),
        Int32(1),
        current_cu_stream(),
    )


class Unpooler:
    """One MoE layer's unpooler column block, prepared at build (before CUDA
    graph capture): the [N, K] BF16 weight shard (this rank's rows of the
    replicated unpooler weight) and the kernels of both token tiles compiled.

    With ``mailbox`` (the next boundary's LamportMailbox, width N) the epilogue
    also publishes y into it (zero-copy); without, y is only stored and the
    boundary publishes it. With ``ticket`` (int32 [1], stage << 16: the draft
    boundary's ticket on that mailbox) the slot's stage is the ticket's, and the
    mailbox's own stage state is not used. With ``rows`` only the token tiles
    those row counts take are compiled at build (default: every row count); a
    call with another row count compiles its tile then, which must not happen
    under CUDA graph capture. ``run`` is graph-capturable and allocates
    nothing.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        mailbox: LamportMailbox | None = None,
        *,
        enable_pdl: bool = True,
        shape: UnpoolerShape = UNPOOLER,
        ticket: torch.Tensor | None = None,
        rows: Sequence[int] | None = None,
    ) -> None:
        reason = self.unsupported(weight, shape)
        if reason is not None:
            raise ValueError(reason)
        n, k = map(int, weight.shape)
        if weight.is_cuda and torch.cuda.get_device_capability(weight.device)[0] != 10:
            raise RuntimeError("the unpooler GEMM needs an SM100-family GPU")
        self.n, self.k, self.weight = n, k, weight
        self.mailbox = mailbox
        options: dict[str, Any] = dict(
            splits=shape.splits, kw=shape.kw, accs=shape.accs, enable_pdl=enable_pdl
        )
        if mailbox is not None:
            if mailbox.width != n:
                raise ValueError(f"mailbox width {mailbox.width} != unpooler N {n}")
            options.update(
                zerocopy=True,
                tp=mailbox.tp,
                rank=mailbox.rank,
                capacity_m=mailbox.capacity,
                hidden_local=mailbox.width,
            )
            self.max_m = min(MAX_M, mailbox.capacity)
            self.stage_state = mailbox.stage_state
            self.address = Int64(mailbox.multicast)
            if ticket is not None:
                if (
                    ticket.dtype != torch.int32
                    or tuple(ticket.shape) != (1,)
                    or ticket.device != mailbox.tensor.device
                ):
                    raise ValueError("the ticket must be int32 [1] on the mailbox")
                options.update(ticket_stage=True)
                self.stage_state = ticket
        elif ticket is not None:
            raise ValueError("a ticket stage needs a mailbox")
        else:
            self.max_m = MAX_M
            # Never read: the plain kernel takes no stage.
            self.stage_state = torch.zeros(2, dtype=torch.int32, device=weight.device)
            self.address = Int64(0)
        self.token_tile = {
            m: tile for m, tile in shape.token_tile.items() if 1 <= m <= self.max_m
        }
        if sorted(self.token_tile) != list(range(1, self.max_m + 1)) or any(
            tile not in TOKEN_TILES or tile < m for m, tile in self.token_tile.items()
        ):
            raise ValueError(
                f"token tiles {shape.token_tile} do not cover M 1..{self.max_m}"
            )
        if rows is None:
            rows = tuple(self.token_tile)
        elif any(m not in self.token_tile for m in rows):
            raise ValueError(f"rows {tuple(rows)} outside 1..{self.max_m}")
        self._options = options
        self.kernels = {
            tile: compile_unpooler(n, k, tile, **options)
            for tile in sorted({self.token_tile[m] for m in rows})
        }
        self.weight_view = to_cute(weight.view(-1), 16)
        self.stage_view = to_cute(self.stage_state, 4)

    @staticmethod
    def unsupported(
        weight: torch.Tensor, shape: UnpoolerShape = UNPOOLER
    ) -> str | None:
        """Why the GEMM cannot take ``weight`` (an [N, K] shard) at launch
        shape ``shape``, or None: the constructor's checks of both, which it
        raises as ValueError."""
        if weight.dim() != 2:
            return f"unpooler shard must be 2-D, got {tuple(weight.shape)}"
        n, k = map(int, weight.shape)
        if (
            weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
            or weight.data_ptr() % 16
            or n % TILE_N
        ):
            return (
                f"unpooler shard {tuple(weight.shape)} {weight.dtype}: needs a "
                f"contiguous, 16-byte aligned BF16 [N, K] with N % {TILE_N} == 0"
            )
        return launch_unsupported(n, k, shape.splits, shape.kw, shape.accs)

    @classmethod
    def from_moe(
        cls,
        moe: torch.nn.Module,
        tp: int,
        rank: int,
        mailbox: LamportMailbox | None = None,
        **options: Any,
    ) -> "Unpooler":
        """The unpooler of a MuseSpark1xMoE: rows [rank N, (rank + 1) N) of its
        replicated [hidden, K] weight, N = hidden / tp (as the stock
        _project_local_columns takes them)."""
        weight = moe.unpooler.weight
        n = weight.shape[0] // tp
        return cls(weight.narrow(0, rank * n, n), mailbox, **options)

    def run(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """out[m, N] = x[m, K] @ W^T (and its publish with a mailbox)."""
        m = int(x.shape[0])
        if not 1 <= m <= self.max_m or tuple(x.shape) != (m, self.k):
            raise ValueError(f"x {tuple(x.shape)}: needs [1..{self.max_m}, {self.k}]")
        if tuple(out.shape) != (m, self.n):
            raise ValueError(f"out {tuple(out.shape)} != ({m}, {self.n})")
        for tensor in (x, out):
            if (
                tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError("x and out must be contiguous 16-byte aligned BF16")
        tile = self.token_tile[m]
        kernel = self.kernels.get(tile)
        if kernel is None:
            kernel = compile_unpooler(self.n, self.k, tile, **self._options)
            self.kernels[tile] = kernel
        kernel(
            self.weight_view,
            to_cute_dynamic(x.view(-1), 16, divisibility=self.k),
            to_cute_dynamic(out.view(-1), 16, divisibility=self.n),
            self.stage_view,
            self.address,
            Int32(m),
            current_cu_stream(),
        )
        return out
