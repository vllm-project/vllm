# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Down projection of a decode hyper-connection read in CuTe DSL.

The stock read projects the normalized streams `xn [rows, 10240]` onto the
merged down+inject weight `[336, 10240]` (320 low-rank rows, 4 injection
rows, 12 zero padding rows) with `hc_down_silu`. At decode sizes that kernel
streams 6.6 MB of weight after its grid-dependency wait, so the weight read
sits on the critical path behind the previous kernel. `hc_down` computes
the same outputs with the weight already on chip when the activations arrive:

* every CTA loads its whole weight share into registers before
  `griddepcontrol.wait`, while the previous kernel (the combine-norm) still
  runs, and triggers its dependents as soon as those loads are issued;
* after the wait only the activations (L2-resident) and the reductions remain.

Work split. CTA `(tile, split)` owns the 16 weight rows `[16 tile,
16 tile + 16)` over the k range `[1280 split, 1280 split + 1280)`; each of
its four warps takes the 32-wide chunks `w, w + 4, ..., w + 36` of that
range (ten chunks, 40 KB of weight per CTA, 21 x 8 = 168 CTAs). Every lane
holds 160 B of weight, two rows by ten chunks. The MMA is `m16n8k16`
with A = 16 weight rows and B = 8 token rows (one eight-row group), so the
eight CTAs of a tile, one per k split, form a thread-block cluster:

1. each warp multiplies its chunks for every group of the call, two groups'
   activations loaded before either is multiplied;
2. the four warps' partial tiles meet in shared memory and are added in warp
   order;
3. after a cluster barrier, CTA `split` reads its eighth of the tile's
   outputs from the eight CTAs' shared memory (distributed shared memory) and
   adds them in cluster-rank order.

Every sum has a fixed order, so equal inputs give equal bits. The epilogue
keeps the stock rounding: `x = bf16(acc)`, `low = bf16(silu(x / 4))` for
the 320 low-rank columns and `inject = x` for the 4 injection columns, in a
`[rows, 336]` output whose padding columns are not written (as
`hc_down_silu`). The k summation order differs from the stock kernels, so
`acc` (not the operation sequence after it) can differ in the last bits.

Scope: 1-64 token rows (eight groups). A group with fewer than eight rows
repeats its last row in the empty MMA columns, whose outputs are dropped.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream

from .. import HC_HIDDEN, HC_LOW_RANK, HC_STREAMS
from ._ptx import (
    evict_first_policy,
    load_cluster_f32,
    load_shared_f32,
    load_weight_x4,
    load_x4,
    mma_chunk,
    store_bf16,
    store_shared_f32,
    store_shared_f32x4,
)

STREAMS = HC_STREAMS
WIDTH = STREAMS * HC_HIDDEN
LOW_RANK = HC_LOW_RANK
MAX_ROWS = 64

SPLITS = 8  # k splits of a tile: the cluster size
SPLIT_WIDTH = WIDTH // SPLITS  # 1280
WARPS = 4
THREADS = 32 * WARPS
CHUNKS = SPLIT_WIDTH // (32 * WARPS)  # 32-wide chunks per warp: 10
TILE_ROWS = 16
GROUP_ROWS = 8  # token rows of one MMA (its N)
PAIR = 2  # groups whose activations are loaded together
# Shared memory: a warp's partial tile is 32 lanes x 4 FP32 values.
PARTIAL_BYTES = 32 * 4 * 4
assert CHUNKS * 32 * WARPS * SPLITS == WIDTH


def _sigmoid(x: Float32) -> Float32:
    """`1 / (1 + exp(-x))` as the stock `hc_down_silu` computes it."""
    return cute.math.rcp(1.0 + cute.math.exp(-x), approx=True)


class HcDown:
    """The down projection of a hyper-connection read for one group count.

    Args:
        groups: Eight-row token groups of a call (1 to 8); the row count is a
            runtime argument within `[8 groups - 7, 8 groups]`.
        low_rank: Low-rank columns (SiLU epilogue).
        injection: Injection-logit columns after them (passthrough epilogue):
            one per stream at a combining read, none at the final mixer.

    """

    def __init__(self, groups: int, low_rank: int, injection: int) -> None:
        assert 1 <= groups <= MAX_ROWS // GROUP_ROWS
        self.groups = groups
        self.low_rank = low_rank
        self.outputs = low_rank + injection
        self.tiles = (self.outputs + TILE_ROWS - 1) // TILE_ROWS

    @cute.jit
    def __call__(
        self,
        x: Int64,
        weight: Int64,
        weight_rows: Int32,
        out: Int64,
        out_stride: Int32,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on `stream`: `x [rows, 10240]`, `weight [weight_rows,
        10240]` (rows past `outputs` ignored), `out [rows, out_stride]`
        BF16 addresses."""
        self.kernel(x, weight, weight_rows, out, out_stride, rows).launch(
            grid=(self.tiles, SPLITS, 1),
            block=(THREADS, 1, 1),
            cluster=(1, SPLITS, 1),
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=3,
        )

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        weight: Int64,
        weight_rows: Int32,
        out: Int64,
        out_stride: Int32,
        rows: Int32,
    ):
        """One CTA: 16 weight rows over one k split, for every token row."""
        groups = self.groups
        tid, _, _ = cute.arch.thread_idx()
        tile, split, _ = cute.arch.block_idx()  # split == rank in the cluster
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        g = lane // 4
        t = lane % 4
        # Chunk j of this lane: k = k_lane + 128 j + [0, 8).
        k_lane = split * SPLIT_WIDTH + 32 * warp + 8 * t
        chunk_bytes = 32 * WARPS * 2

        # The weight share, before the grid-dependency wait. A ragged last
        # tile repeats the last weight row; those outputs are dropped.
        policy = evict_first_policy()
        row_lo = cute.min(TILE_ROWS * tile + g, weight_rows - 1)
        row_hi = cute.min(TILE_ROWS * tile + g + 8, weight_rows - 1)
        address_lo = weight + Int64(row_lo * WIDTH + k_lane) * 2
        address_hi = weight + Int64(row_hi * WIDTH + k_lane) * 2
        lo = []
        hi = []
        for j in cutlass.range_constexpr(CHUNKS):
            lo.append(load_weight_x4(address_lo + j * chunk_bytes, policy))
            hi.append(load_weight_x4(address_hi + j * chunk_bytes, policy))

        allocator = cutlass.utils.SmemAllocator()
        # [group][warp][lane][4]: the warps' MMA tiles.
        partial = allocator.allocate_array(
            Float32, groups * WARPS * 32 * 4, byte_alignment=16
        ).toint()
        # [group][token][weight row]: this CTA's sums over its warps.
        reduced = allocator.allocate_array(
            Float32, groups * GROUP_ROWS * TILE_ROWS, byte_alignment=16
        ).toint()

        # Dependents may launch now: they read this kernel's output only after
        # their own grid-dependency wait. Later kernels in the chain are covered
        # too: each kernel ends only after its own wait returned, so a
        # completed kernel implies completed predecessors. The trigger follows
        # the weight loads on purpose: placed before them, ptxas schedules the
        # wait right after the trigger and sinks the non-coherent weight loads
        # below it.
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()

        # 1. Partial tiles of every group (B column g = token 8 f + g).
        for first in cutlass.range_constexpr(0, groups, PAIR):
            count = min(PAIR, groups - first)
            activations = []
            for u in cutlass.range_constexpr(count):
                token = cute.min(GROUP_ROWS * (first + u) + g, rows - 1)
                address = x + Int64(token * WIDTH + k_lane) * 2
                chunks = []
                for j in cutlass.range_constexpr(CHUNKS):
                    chunks.append(load_x4(address + j * chunk_bytes))
                activations.append(chunks)
            for u in cutlass.range_constexpr(count):
                acc = (Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0))
                for j in cutlass.range_constexpr(CHUNKS):
                    acc = mma_chunk(acc, lo[j], hi[j], activations[u][j])
                store_shared_f32x4(
                    partial + ((first + u) * WARPS + warp) * PARTIAL_BYTES + lane * 16,
                    acc,
                )
        cute.arch.sync_threads()

        # 2. Sum the warps in warp order. Lane (g, t) of a warp holds
        # D[g][2t], D[g][2t+1], D[g+8][2t], D[g+8][2t+1] (weight row, token);
        # thread `tid` sums element (token, r) of every group.
        token = (tid >> 4) & 7
        r = tid & 15
        source = (r & 7) * 4 + (token >> 1)
        slot = (r >> 3) * 2 + (token & 1)
        for f in cutlass.range_constexpr(groups):
            element = partial + f * WARPS * PARTIAL_BYTES + source * 16 + slot * 4
            total = load_shared_f32(element)
            for w in cutlass.range_constexpr(1, WARPS):
                total = total + load_shared_f32(element + w * PARTIAL_BYTES)
            store_shared_f32(reduced + (f * THREADS + tid) * 4, total)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        # 3. CTA `split` owns outputs [split * per, split * per + per) of the
        # tile: the eight CTAs' sums in cluster-rank order, then the epilogue.
        per = groups * TILE_ROWS
        if tid < per:
            i = split * per + tid
            total = load_cluster_f32(reduced + i * 4, Int32(0))
            for q in cutlass.range_constexpr(1, SPLITS):
                total = total + load_cluster_f32(reduced + i * 4, Int32(q))
            row = i >> 4
            column = TILE_ROWS * tile + (i & 15)
            if (row < rows) & (column < self.outputs):
                address = out + Int64(row * out_stride + column) * 2
                # The stock rounding boundary: the GEMM result is BF16.
                value = total.to(BFloat16).to(Float32)
                if column < self.low_rank:
                    z = value * (1.0 / STREAMS)
                    store_bf16(address, z * _sigmoid(z))
                else:
                    store_bf16(address, value)
        # Peers read this CTA's shared memory until they pass this barrier.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()


@cache
def _compiled(device_index: int, groups: int, low_rank: int, injection: int):
    """`HcDown` compiled for a device and a group count."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            HcDown(groups, low_rank, injection),
            Int64(0),
            Int64(0),
            Int32(1),
            Int64(0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


def hc_down(
    xn: torch.Tensor,
    weight: torch.Tensor,
    low_rank: int = LOW_RANK,
    injection: int = STREAMS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hyper-connection down projection with its activation, for 1-64 rows.

    Args:
        xn: Normalized streams, contiguous BF16 `[rows, 10240]`.
        weight: Merged down+inject weight, contiguous BF16 `[n, 10240]` with
            `n >= low_rank + injection`; rows past those are ignored.
        low_rank: Low-rank output columns.
        injection: Injection-logit output columns (`STREAMS` or 0).

    Returns:
        `silu(bf16(xn @ weight[:low_rank].T) / 4)` as BF16
        `[rows, low_rank]` and the injection logits
        `bf16(xn @ weight[low_rank:low_rank + injection].T)` as BF16
        `[rows, injection]`: column views of one `[rows, n]` buffer, as
        `hc_down_silu` returns them.

    """
    rows = xn.shape[0]
    assert 1 <= rows <= MAX_ROWS, f"hc_down serves 1-{MAX_ROWS} rows, got {rows}"
    assert xn.dtype == weight.dtype == torch.bfloat16
    assert xn.shape[1] == WIDTH and xn.is_contiguous()
    assert weight.shape[1] == WIDTH and weight.is_contiguous()
    assert weight.shape[0] >= low_rank + injection
    out = torch.empty(rows, weight.shape[0], dtype=torch.bfloat16, device=xn.device)
    groups = (rows + GROUP_ROWS - 1) // GROUP_ROWS
    _compiled(xn.device.index, groups, low_rank, injection)(
        Int64(xn.data_ptr()),
        Int64(weight.data_ptr()),
        Int32(weight.shape[0]),
        Int64(out.data_ptr()),
        Int32(out.stride(0)),
        Int32(rows),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out[:, :low_rank], out[:, low_rank : low_rank + injection]
