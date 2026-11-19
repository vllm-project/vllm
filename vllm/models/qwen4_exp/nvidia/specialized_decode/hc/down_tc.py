# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Down projection of a decode hyper-connection read on the tensor cores
(tcgen05), for up to 128 token rows.

`.down.hc_down` serves 1-64 rows with warp MMAs: each of its 21
weight-row tiles re-reads the normalized streams `xn [rows, 10240]` into
registers, which bounds it at 64 rows. `hc_down_tc` computes the same
outputs (`low = bf16(silu(bf16(xn @ W[:320].T) / 4))` and the injection
logits `bf16(xn @ W[320:324].T)`, the stock `hc_down_silu` epilogue and
rounding) for up to 128 rows with one UMMA per 16-value K step:

* A is `xn` (M = 128 token rows; TMA zero-fills rows past `rows`), B is a
  tile of `ntile` weight rows (N = `ntile`), D accumulates in tensor
  memory: lane `t` holds token `t`'s outputs of the tile.
* CTA `(tile, split)` owns weight rows `[ntile tile, ntile tile + ntile)`
  over the K range `[640 split, 640 split + 640)` (16 splits); the 16 splits
  of a tile form a thread-block cluster.
* Before `griddepcontrol.wait` each CTA loads its whole weight share into
  shared memory (TMA, L2 evict-first) and triggers its dependents; the
  previous kernel (the combine-norm or the boundary kernel) still runs. After
  the wait, warp 0 streams `xn` through a ring of shared-memory stages and
  warp 1 issues the UMMAs.
* Split-K reduce-scatter: thread `t` stores its token's FP32 partial row
  into the shared memory of the cluster's CTA `t // 8` (`st.async`,
  completing on that CTA's mbarrier); CTA `r` then sums the 16 partials of
  its 8 tokens in cluster-rank order and applies the epilogue.

Every sum has a fixed order, so equal inputs give equal bits. Only the FP32
accumulation order of the projection differs from `hc_down` (and from the
stock kernels), so `bf16(acc)` can differ in its last bit.

Shared memory: the weight share (`ntile` x 640 BF16), `stages` xn stages
of 16 KiB and the partials received (128 x `ntile` FP32), at most 184 KiB
with the defaults, so that CTAs of the previous kernel still fit beside a CTA
launched early.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from .. import HC_HIDDEN, HC_LOW_RANK, HC_STREAMS
from ..boundary._ptx import load_shared_u32
from . import _tc
from ._ptx import evict_first_policy, load_shared_f32, store_bf16

STREAMS = HC_STREAMS
WIDTH = STREAMS * HC_HIDDEN
LOW_RANK = HC_LOW_RANK
MAX_ROWS = 128

TOKENS = 128  # MMA M: token rows of a call
ATOM = 64  # K values of one 128-byte swizzle atom
MMA_K = 16  # K values of one UMMA
THREADS = 128
# Default tiling (module docstring).
NTILE = 48
SPLITS = 16
STAGES = 6
# Shared memory a CTA may take (bytes, beside the barriers): the budget with
# a separate buffer for the received partials, the limit when they reuse the
# weight share.
SMEM_BUDGET = 190 * 1024
SMEM_LIMIT = 225 * 1024
# Shared memory beside the operand buffers: barriers, the tensor-memory
# address and the 1024-byte alignment of the swizzled buffers.
SMEM_SLACK = 2048


def _sigmoid(x: Float32) -> Float32:
    """`1 / (1 + exp(-x))` as the stock `hc_down_silu` computes it."""
    return cute.math.rcp(1.0 + cute.math.exp(-x), approx=True)


def sw128_layout(rows: int, stages: int) -> cute.ComposedLayout:
    """Shared-memory layout of `stages` K-major operand tiles of `rows`
    rows by one 128-byte swizzle atom (`_tc` docstring)."""
    return cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3),
        0,
        cute.make_layout((rows, ATOM, stages), stride=(ATOM, 1, rows * ATOM)),
    )


def tma_load(
    tma: cpasync.TmaInfo,
    tile: cute.Tensor,
    stage: cute.Tensor,
    barrier: cute.Pointer,
    policy: Int64 | None = None,
) -> None:
    """Load one tile of `tma`'s tensor into a shared-memory stage,
    completing on `barrier`, with the L2 `policy` if given. A whole warp
    calls it (the copy elects its issuing thread)."""
    smem, gmem = cpasync.tma_partition(
        tma.atom,
        0,
        cute.make_layout(1),
        cute.group_modes(stage, 0),
        cute.group_modes(tile, 0),
    )
    if policy is None:
        cute.copy(tma.atom, gmem, smem, tma_bar_ptr=barrier)
    else:
        cute.copy(tma.atom, gmem, smem, tma_bar_ptr=barrier, cache_policy=policy)


class HcDownTc:
    """The down projection of a hyper-connection read for 1-128 rows.

    Args:
        low_rank: Low-rank columns (SiLU epilogue).
        injection: Injection-logit columns after them (passthrough epilogue):
            one per stream at a combining read, none at the final mixer.
        ntile: Weight rows of a CTA (the UMMA's N, a multiple of 16).
        splits: K splits of a tile (the cluster size; 128 divisible by it).
        stages: Shared-memory stages of `xn` (at most the K atoms of a
            split, and what `smem_budget` leaves).
        smem_budget: Shared memory a CTA may take (bytes).

    """

    def __init__(
        self,
        low_rank: int,
        injection: int,
        ntile: int = NTILE,
        splits: int = SPLITS,
        stages: int = STAGES,
        smem_budget: int = SMEM_BUDGET,
    ) -> None:
        assert ntile % 16 == 0 and 16 <= ntile <= 256
        assert TOKENS % splits == 0 and WIDTH % (splits * ATOM) == 0
        self.low_rank = low_rank
        self.outputs = low_rank + injection
        self.ntile = ntile
        self.tiles = (self.outputs + ntile - 1) // ntile
        self.splits = splits
        self.atoms = WIDTH // splits // ATOM  # K atoms of a CTA
        self.owned = TOKENS // splits  # tokens a CTA reduces
        self.w_bytes = ntile * ATOM * 2  # one weight atom
        self.x_bytes = TOKENS * ATOM * 2  # one xn stage
        weights = self.atoms * self.w_bytes
        received = TOKENS * ntile * 4
        # The received partials get their own buffer when it fits the budget
        # beside two xn stages; otherwise they reuse the weight share once
        # every CTA of the cluster finished its UMMAs (a cluster barrier).
        self.alias = weights + received + 2 * self.x_bytes > smem_budget
        assert not self.alias or received <= weights
        room = smem_budget if not self.alias else SMEM_LIMIT
        room -= weights + (0 if self.alias else received)
        self.stages = min(stages, self.atoms, room // self.x_bytes)
        assert self.stages >= 2, "the weight share leaves no room for xn"
        # The dynamic shared memory a CTA takes, at most.
        self.smem_bytes = (
            weights
            + self.stages * self.x_bytes
            + (0 if self.alias else received)
            + SMEM_SLACK
        )
        self.tmem_columns = max(32, 1 << (ntile - 1).bit_length())
        self.idesc = _tc.mma_idesc(TOKENS, ntile)

    @cute.jit
    def __call__(
        self,
        xn: cute.Tensor,
        weight: cute.Tensor,
        out: Int64,
        out_stride: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on `stream`: `xn [rows, 10240]`, `weight [n, 10240]`
        (rows past `outputs` ignored), `out [rows, out_stride]` BF16."""
        op = cpasync.CopyBulkTensorTileG2SOp()
        x_tma = cpasync.make_tiled_tma_atom(
            op, xn, sw128_layout(TOKENS, self.stages), (TOKENS, ATOM)
        )
        w_tma = cpasync.make_tiled_tma_atom(
            op, weight, sw128_layout(self.ntile, self.atoms), (self.ntile, ATOM)
        )
        self.kernel(x_tma, w_tma, out, out_stride, xn.shape[0]).launch(
            grid=(self.tiles, self.splits, 1),
            block=(THREADS, 1, 1),
            cluster=(1, self.splits, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        x_tma: cpasync.TmaInfo,
        w_tma: cpasync.TmaInfo,
        out: Int64,
        out_stride: Int32,
        rows: Int32,
    ):
        """One CTA: `ntile` weight rows over one K split, then the
        reduction of its `owned` tokens over the cluster."""
        ntile = self.ntile
        atoms = self.atoms
        stages = self.stages
        owned = self.owned
        tid, _, _ = cute.arch.thread_idx()
        tile, split, _ = cute.arch.block_idx()  # split == rank in the cluster
        warp = cute.arch.make_warp_uniform(tid // 32)

        allocator = cutlass.utils.SmemAllocator()
        s_w = allocator.allocate_tensor(
            BFloat16,
            w_tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=w_tma.smem_layout.inner,
        )
        s_x = allocator.allocate_tensor(
            BFloat16,
            x_tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=x_tma.smem_layout.inner,
        )
        # [source rank][owned token][weight row] FP32: the partial rows the
        # cluster sends this CTA (in the weight share's buffer when aliased).
        received = s_w.iterator.toint()
        if cutlass.const_expr(not self.alias):
            received = allocator.allocate_array(
                Float32, TOKENS * ntile, byte_alignment=16
            ).toint()
        # mbarriers: weights, xn stages full and empty, accumulator, partials.
        bars = allocator.allocate_array(Int64, 3 + 2 * stages, byte_alignment=8)
        tmem_slot = allocator.allocate_array(Int32, 1, byte_alignment=4)
        w_full = bars
        acc_full = bars + 1
        received_full = bars + 2
        x_full = bars + 3
        x_empty = bars + 3 + stages

        if warp == 0:
            with cute.arch.elect_one():
                _tc.mbarrier_init(w_full.toint(), 1)
                _tc.mbarrier_init(acc_full.toint(), 1)
                _tc.mbarrier_init(received_full.toint(), 1)
                for s in cutlass.range_constexpr(stages):
                    _tc.mbarrier_init((x_full + s).toint(), 1)
                    _tc.mbarrier_init((x_empty + s).toint(), 1)
                _tc.fence_mbarrier_init()
        if warp == 1:
            _tc.tmem_alloc(tmem_slot.toint(), self.tmem_columns)
            _tc.fence_before_sync()
        cute.arch.sync_threads()
        _tc.fence_after_sync()
        tmem = Int32(load_shared_u32(tmem_slot.toint()))

        # The weight share, before the grid-dependency wait.
        g_w = cute.zipped_divide(w_tma.tma_tensor, (ntile, ATOM))
        if warp == 0:
            with cute.arch.elect_one():
                _tc.mbarrier_arrive_expect(w_full.toint(), atoms * self.w_bytes)
            policy = evict_first_policy()
            for a in cutlass.range_constexpr(atoms):
                tma_load(
                    w_tma,
                    g_w[None, (tile, split * atoms + a)],
                    s_w[None, None, a],
                    w_full,
                    policy,
                )
        # Peers store into this CTA's shared memory only after its barriers
        # are initialized (the matching wait precedes the stores).
        cute.arch.cluster_arrive_relaxed()
        # Dependents may launch now: they read this kernel's output only after
        # their own grid-dependency wait.
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()

        if warp == 0:
            # xn of the K split, through the stage ring.
            g_x = cute.zipped_divide(x_tma.tma_tensor, (TOKENS, ATOM))
            for a in cutlass.range_constexpr(atoms):
                s = a % stages
                if cutlass.const_expr(a >= stages):
                    _tc.mbarrier_wait((x_empty + s).toint(), (a // stages - 1) % 2)
                with cute.arch.elect_one():
                    _tc.mbarrier_arrive_expect((x_full + s).toint(), self.x_bytes)
                tma_load(
                    x_tma,
                    g_x[None, (0, split * atoms + a)],
                    s_x[None, None, s],
                    x_full + s,
                )
        elif warp == 1:
            w_base = s_w.iterator.toint()
            x_base = s_x.iterator.toint()
            _tc.mbarrier_wait(w_full.toint(), 0)
            for a in cutlass.range_constexpr(atoms):
                s = a % stages
                _tc.mbarrier_wait((x_full + s).toint(), (a // stages) % 2)
                _tc.fence_after_sync()
                with cute.arch.elect_one():
                    for k in cutlass.range_constexpr(ATOM // MMA_K):
                        _tc.mma(
                            tmem,
                            _tc.sw128_desc(x_base + s * self.x_bytes + 32 * k),
                            _tc.sw128_desc(w_base + a * self.w_bytes + 32 * k),
                            Int32(self.idesc),
                            Int32(1 if a > 0 or k > 0 else 0),
                        )
                    _tc.mma_commit((x_empty + s).toint())
            with cute.arch.elect_one():
                _tc.mma_commit(acc_full.toint())

        # This thread's token row of the partial tile (lane `tid`).
        _tc.mbarrier_wait(acc_full.toint(), 0)
        _tc.fence_after_sync()
        lane_address = tmem + ((32 * warp) << 16)
        partial = []
        for c in cutlass.range_constexpr(0, ntile, 16):
            partial.extend(_tc.tmem_load(lane_address + c, 16))
        _tc.tmem_wait_load()
        _tc.fence_before_sync()

        # Send it to the CTA that owns the token, in this CTA's source slot.
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.alias):
            # The peers' weight shares take the partials: wait until every
            # CTA of the cluster saw its UMMAs complete.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        mine = cute.max(cute.min(rows - split * owned, owned), 0)
        if tid == 0:
            _tc.mbarrier_arrive_expect(
                received_full.toint(), self.splits * mine * ntile * 4
            )
        if tid < rows:
            slot = received + ((split * owned + tid % owned) * ntile) * 4
            for v in cutlass.range_constexpr(0, ntile, 4):
                _tc.store_async_peer(
                    slot + 4 * v,
                    (partial[v], partial[v + 1], partial[v + 2], partial[v + 3]),
                    received_full.toint(),
                    tid // owned,
                )

        cute.arch.sync_threads()
        if warp == 1:
            _tc.fence_after_sync()
            _tc.tmem_dealloc(tmem, self.tmem_columns)

        # The owned tokens' outputs: the cluster's partials in rank order.
        _tc.mbarrier_wait_cluster(received_full.toint(), 0)
        per_source = owned * ntile
        for step in cutlass.range_constexpr((per_source + THREADS - 1) // THREADS):
            i = step * THREADS + tid
            token = split * owned + i // ntile
            column = tile * ntile + i % ntile
            if (i < per_source) & (token < rows) & (column < self.outputs):
                element = received + i * 4
                total = load_shared_f32(element)
                for q in cutlass.range_constexpr(1, self.splits):
                    total = total + load_shared_f32(element + q * per_source * 4)
                address = out + Int64(token * out_stride + column) * 2
                # The stock rounding boundary: the GEMM result is BF16.
                value = total.to(BFloat16).to(Float32)
                if column < self.low_rank:
                    z = value * (1.0 / STREAMS)
                    store_bf16(address, z * _sigmoid(z))
                else:
                    store_bf16(address, value)


@cache
def _compiled(
    device_index: int,
    low_rank: int,
    injection: int,
    ntile: int = NTILE,
    splits: int = SPLITS,
    stages: int = STAGES,
    smem_budget: int = SMEM_BUDGET,
):
    """`HcDownTc` compiled for a device (any row count, any weight
    row count)."""
    rows = cute.sym_int()
    weight_rows = cute.sym_int()
    xn = make_fake_tensor(BFloat16, (rows, WIDTH), (WIDTH, 1), assumed_align=16)
    weight = make_fake_tensor(
        BFloat16, (weight_rows, WIDTH), (WIDTH, 1), assumed_align=16
    )
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            HcDownTc(low_rank, injection, ntile, splits, stages, smem_budget),
            xn,
            weight,
            Int64(0),
            Int32(1),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def hc_down_tc(
    xn: torch.Tensor,
    weight: torch.Tensor,
    low_rank: int = LOW_RANK,
    injection: int = STREAMS,
    **tiling: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hyper-connection down projection with its activation, for 1-128 rows
    (the TP1 read takes it from `read.TC_MIN_ROWS` rows).

    Args:
        xn: Normalized streams, contiguous BF16 `[rows, 10240]`.
        weight: Merged down+inject weight, contiguous BF16 `[n, 10240]` with
            `n >= low_rank + injection`; rows past those are ignored.
        low_rank: Low-rank output columns.
        injection: Injection-logit output columns (`STREAMS` or 0).
        tiling: `ntile`, `splits`, `stages`, `smem_budget` of
            `HcDownTc`.

    Returns:
        `silu(bf16(xn @ weight[:low_rank].T) / 4)` as BF16
        `[rows, low_rank]` and the injection logits
        `bf16(xn @ weight[low_rank:low_rank + injection].T)` as BF16
        `[rows, injection]`: column views of one `[rows, n]` buffer, as
        `hc_down_silu` returns them.

    """
    rows = xn.shape[0]
    assert 1 <= rows <= MAX_ROWS, f"hc_down_tc serves 1-{MAX_ROWS} rows, got {rows}"
    assert xn.dtype == weight.dtype == torch.bfloat16
    assert xn.shape[1] == WIDTH and xn.is_contiguous()
    assert weight.shape[1] == WIDTH and weight.is_contiguous()
    assert weight.shape[0] >= low_rank + injection
    out = torch.empty(rows, weight.shape[0], dtype=torch.bfloat16, device=xn.device)
    kernel = _compiled(xn.device.index, low_rank, injection, **tiling)
    kernel(xn, weight, out.data_ptr(), out.stride(0))
    return out[:, :low_rank], out[:, low_rank : low_rank + injection]
