# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN decode core of a speculative verification step in one CuTe DSL kernel.

Each of Qwen4Exp's gated delta net (GDN) layers projects its input into the
packed `qkvz` rows (queries, keys, values and the output gate `z`) and the
`ba` rows (the per-head `beta` and decay logits), then runs its core: a
width-4 causal convolution with SiLU over the q, k and v channels, the L2
normalization of q and k, the gated delta rule on each value head's 128 x 128
recurrent state, and the gated RMSNorm of the result. In a decode step with MTP
each request verifies `q` rows (the draft tokens plus one). Stock vLLM runs
this core as a zero fill of the output, the Triton `causal_conv1d_update` and
the CUDA `gdn_decode_post_conv_mtp` kernel, whose single CTA per (request,
value head) walks the state in four serial chunks. `gdn_decode_core`
does all of it in one kernel that splits each request's state over many CTAs
and loads it before the grid-dependency wait, so the state streams in while
the `qkvz` projection still runs (programmatic dependent launch).

State protocol (vLLM's speculative-decode cache semantics, `slots`): the
slot table `slots [requests, >= q]` names each request's state slots; the
recurrent state is read from `slots[b, accepted[b] - 1]` and the state after
token `t` is written to `slots[b, t]` (slots <= 0 are not written). The
convolution state lives in `slots[b, 0]` as `[channel, q + 2]` BF16
values whose taps for this step start at position `accepted[b] - 1`; it is
rewritten in place to `[c[accepted], c[accepted + 1], x_0, ..., x_{n-1}]`
for the request's `n` rows. A request whose state slot or convolution slot
is <= 0 (a padded request) writes zero output rows and no state. With the
`records` protocol (`commit.py`) the kernel writes the same outputs and
convolution state but no recurrent state: after the token loop each CTA
writes its rows' verify records (normalized keys, decays, delta-rule
corrections) into its key head's band of the request's two copy slots (the
first two non-source speculative slots, `slots[b, t] > 0`, that cannot be
the `align` boundary row's: `commit.py`), and the cluster's first CTA of
the first key head the request's header, for the commit after sampling; a
padded request writes a header of no rows. The `slots`
protocol compiles to its own kernel entry (`kernel`), whose code the
`records` entry (`kernel_records`) leaves as it was.

Work split. The `3 * key_heads` value heads share the q and k channels of
their key head (grouped queries, three value heads per key head). A thread
block cluster owns one (request, key head): `S` CTAs, each with the value
rows `[p R, p R + R)` (`R = 128 / partitions`) of `heads_per_cta` of the
three value heads. Every CTA computes the convolution and normalization of all
256 q and k channels of its key head (cheaper than exchanging them), and the
convolution of its own v channels. A warp updates `W = R / (warps /
heads_per_cta)` state rows; each lane holds four consecutive keys of every
row. The warp sums each row's dot products over its lanes with a
reduce-scatter butterfly: every lane ends with the sum of one row (`W + W/2
+ ... + 1` shuffles for the `W` rows instead of `5 W`), computes that
row's delta-rule correction, and the warp exchanges the corrections through
shared memory.

Sequence of one CTA:

1. before the wait: the slot table, the convolution taps and weights, the
   gate parameters, the norm weight and the CTA's state rows (all written by
   earlier graphs or loaded weights), then the dependents' trigger;
2. after the wait: the `qkvz` and `ba` rows of the request (written by
   this step's projections), the convolutions, the gates, the q and k norms;
3. the delta rule over the q tokens, writing each token's state rows and its
   BF16 output rows to shared memory;
4. a cluster barrier; the gated RMSNorm of each (token, value head) from the
   cluster's shared memory (distributed shared memory), each CTA writing its
   own rows; the convolution state (q and k channels split over the cluster:
   all CTAs read their taps before the barrier); a final cluster barrier
   keeps every CTA's shared memory alive for its peers.

With one partition a CTA owns whole value heads: its norm reads only its own
shared memory (a CTA barrier replaces the cluster barrier), and the cluster
barrier that orders the convolution state's reads before its writes is split,
arriving after the convolutions and waiting just before the writes.

Programmatic dependent launch (PDL). The kernel is launched with PDL right
after the `qkvz` projection on the same stream (the staged-weight projection
of `qkvz` and `ba` in the staged layer, which triggers before its own
wait), so its CTAs may start while the projection, and possibly kernels
before it, still run. Each CTA also
triggers its own dependents (`griddepcontrol.launch_dependents`) before its
`griddepcontrol.wait`, so the next kernel may start before this one has
passed its wait. Two invariants make this safe:

- Before its wait a CTA reads only memory that no kernel of the step can
  still be writing:
  * the slot table, the accepted counts and `cu_seqlens` (the step's
    metadata buffers): the model runner writes them before it launches the
    step's graph, and nothing in the graph writes them;
  * the convolution weight, `A_log`, `dt_bias` and the norm weight:
    loaded weights, never written afterwards;
  * this layer's convolution taps and recurrent state rows: only this
    layer's core writes them (this kernel, or the stock layer when the step
    does not take the fused core), so they were written by earlier steps (earlier
    graph replays or eager forwards), which completed before this step's graph
    started. Each layer has its own caches. Inside this kernel, the cluster
    barriers above order the reads before the writes; the `records`
    protocol's records go only to non-source slots, which no CTA reads.
  Everything the step's projections write (`qkvz`, `ba`) is read after
  the wait, and the kernel writes nothing before it.
- Every dependent reads this kernel's output only after its own wait. A
  dependent's `griddepcontrol.wait` returns only once this grid has completed
  and its writes are visible. This grid completes only after its own wait,
  that is after the `qkvz` projection completed. So completion stays
  transitive along the stream, although the trigger comes early. In the
  executor the dependent is the first kernel of the layer's `out_proj`:
  * for an NVFP4 `out_proj`, vLLM's activation quantization
    (`cvt_fp16_to_fp4`), whose first statement is
    `cudaGridDependencySynchronize()`, so it neither reads nor writes before
    its wait;
  * for an unquantized one, a cuBLAS GEMM, which launches without PDL and so
    takes a full dependency, or, at the row counts that the GPU's plan table
    or the executor's decode GEMM plans (`runtime/gemm.py`) list, the skinny
    CuTe GEMM, which launches with PDL but before its wait only loads weight
    tiles and writes nothing;
  * in the staged layer (`runtime/gdn.py`, small tensor-parallel steps),
    the staged-weight projection (`proj.py`), which before its wait only
    copies its weight into its own shared memory.
  The `pad` path triggers and waits in the same order and writes only after
  its wait.

Numerics. Every value follows the stock path's operation sequence and
rounding: the convolution as the Triton kernel computes it (BF16 products, FP32
sums, its SiLU), rounded to BF16 like the convolution's output; the rest as the
CUDA kernel (built with fast math) computes it, with the same per-row key
reduction (four keys per lane, then the butterfly's pairwise sums over the
warp) and the same BF16 rounding of the recurrence output before the norm
(packed `.f32x2` instructions round each element as the scalar ones do).
The states therefore match the stock path bit for bit when `a` and `b` come
from the same projection; `log1p` (in the softplus) is the one libdevice
function and is compiled without flush-to-zero here. The outputs match it bit
for bit when the stock kernel normalizes each row in one CTA; at small batches
it splits each head's values over two or four CTAs and adds their sums of
squares across the cluster, so its norm's scale, and the outputs, can differ in
the last bit.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream

from ..hc._ptx import (
    load_bf16,
    load_cluster_f32,
    load_shared_f32,
    store_bf16,
    store_shared_f32,
)
from ._ptx import (
    add_ftz,
    add_rn,
    conv_silu,
    exp_ftz,
    fma_ftz,
    fma_ftz_x2,
    load_bf16x4,
    load_f32,
    load_s32,
    load_shared_f32x4,
    load_state_x4,
    mul_bf16,
    mul_ftz,
    mul_ftz_x2,
    rcp_ftz,
    round_bf16,
    rsqrt_ftz,
    select_f32,
    select_s32,
    softplus_select,
    store_s32,
    store_state_x4,
    store_stream_f32,
    store_stream_x4,
    sub_ftz,
)
from .commit import RecordLayout

# State protocols of the core (``commit.py``): ``slots`` (vLLM's: the state
# after every verified row to its slot) and ``records`` (no recurrent state
# writes; per-row records for the commit after sampling).
PROTOCOLS = ("slots", "records")

HEAD_DIM = 128
GQA = 3  # value heads per key head
QK_CHANNELS = 2 * HEAD_DIM  # the q and k channels of one key head
TAPS = 4  # convolution width
WARP = 32
QK_EPS = 1.0e-6  # the stock q / k L2-norm epsilon


@dataclass(frozen=True)
class Schedule:
    """How one (request, key head) is split over a thread block cluster.

    Attributes:
        heads_per_cta: Value heads of a CTA: 3 (all of the key head's) or 1.
        partitions: CTAs per value head, each with `128 / partitions` value
            rows of the state.
        warps: Warps per CTA; each updates an equal share of the CTA's rows.
        ctas_per_sm: CTAs the register budget must fit on one SM (0 leaves
            the register count to the compiler).

    """

    heads_per_cta: int
    partitions: int
    warps: int
    ctas_per_sm: int = 0

    @property
    def cluster(self) -> int:
        """CTAs per (request, key head)."""
        return self.partitions * GQA // self.heads_per_cta

    def validate(self) -> None:
        """Raise if the split is not one the kernel supports."""
        if self.heads_per_cta not in (1, GQA):
            raise ValueError(f"heads_per_cta must be 1 or 3: {self}")
        if HEAD_DIM % self.partitions or self.cluster > 16:
            raise ValueError(f"unsupported partitions: {self}")
        rows = HEAD_DIM // self.partitions
        warps_per_head = self.warps // self.heads_per_cta
        if self.warps % self.heads_per_cta or rows % warps_per_head:
            raise ValueError(f"warps must split every head's rows evenly: {self}")
        warp_rows = rows // warps_per_head
        if warp_rows > WARP or warp_rows & (warp_rows - 1):
            raise ValueError(f"rows per warp must be a power of two <= 32: {self}")


# Schedules by step size, from GB300 measurements with 4 and 16 key heads (the
# fastest of several splits at 1, 4, 8, 16, 32 and 64 requests). A single
# request spreads its state over as many SMs as possible; larger steps use
# smaller clusters (fewer CTAs to co-schedule) and, from 32 (request, key head)
# pairs on, CTAs that fit two per SM and stream the state at full bandwidth.
SINGLE = Schedule(heads_per_cta=GQA, partitions=4, warps=12)
FEW = Schedule(heads_per_cta=1, partitions=2, warps=8)
MANY = Schedule(heads_per_cta=1, partitions=1, warps=8, ctas_per_sm=2)


def default_schedule(key_heads: int, requests: int) -> Schedule:
    """The schedule for `requests` requests over `key_heads` key heads."""
    if requests == 1:
        return SINGLE
    return FEW if requests * key_heads <= 16 else MANY


class GdnDecodeCore:
    """The fused GDN decode core for one deployment shape and schedule.

    Args:
        key_heads: Key heads of the rank (value heads are three times as many).
        tokens: Rows per request at most (`num_speculative_tokens + 1`).
        schedule: The cluster split.
        protocol: The state protocol: `slots` (vLLM's) or `records`
            (`commit.py`).
        layout: The records layout (`records` only).
        align: `records` in prefix-caching `align` mode: the copies avoid
            the slot of the row that can be the commit's boundary row.

    """

    def __init__(
        self,
        key_heads: int,
        tokens: int,
        schedule: Schedule,
        protocol: str = "slots",
        layout: RecordLayout | None = None,
        align: bool = False,
    ) -> None:
        schedule.validate()
        assert protocol in PROTOCOLS
        assert (protocol == "slots") == (layout is None)
        assert protocol == "records" or not align
        self.protocol = protocol
        self.layout = layout
        self.align = align
        self.key_heads = key_heads
        self.value_heads = GQA * key_heads
        self.tokens = tokens
        self.schedule = schedule
        self.hpc = schedule.heads_per_cta
        self.partitions = schedule.partitions
        self.cluster = schedule.cluster
        self.rows = HEAD_DIM // schedule.partitions  # value rows per head and CTA
        self.cta_rows = self.hpc * self.rows
        self.warps = schedule.warps
        self.threads = WARP * schedule.warps
        self.warp_rows = self.rows // (schedule.warps // self.hpc)
        # Column offsets in a qkvz row: q, k (per key head), v, z (per value head).
        self.k_offset = key_heads * HEAD_DIM
        self.v_offset = 2 * key_heads * HEAD_DIM
        self.z_offset = self.v_offset + self.value_heads * HEAD_DIM
        # Convolution tasks: the 256 q / k channels, then the CTA's v channels
        # (on the threads after the first 256 when the CTA has enough).
        self.qk_iters = -(-QK_CHANNELS // self.threads)
        if self.threads >= QK_CHANNELS + self.cta_rows:
            self.v_first, self.v_iters = QK_CHANNELS, 1
        else:
            self.v_first, self.v_iters = 0, -(-self.cta_rows // self.threads)
        self.z_iters = -(-(tokens * self.cta_rows) // self.threads)
        assert tokens * self.hpc <= self.threads

    @property
    def records_layout(self) -> RecordLayout:
        """The records layout of the `records` protocol."""
        assert self.layout is not None
        return self.layout

    @cute.jit
    def __call__(
        self,
        qkvz: Int64,
        qkvz_stride: Int32,
        ba: Int64,
        ba_stride: Int32,
        conv_weight: Int64,
        conv_state: Int64,
        conv_slot_stride: Int64,
        conv_channel_stride: Int32,
        conv_position_stride: Int32,
        state: Int64,
        state_slot_stride: Int64,
        a_log: Int64,
        dt_bias: Int64,
        norm_weight: Int64,
        slots: Int64,
        slots_stride: Int32,
        accepted: Int64,
        cu_seqlens: Int64,
        out: Int64,
        out_stride: Int32,
        eps: Float32,
        records: Int64,
        seq_lens: Int64,
        block_size: Int32,
        requests: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on `stream` for `requests` requests (addresses and
        element strides of the tensors `gdn_decode_core` documents;
        `records` is unused by the `slots` protocol, `seq_lens` and
        `block_size` outside `align` mode)."""
        launch = dict(
            grid=(self.cluster, self.key_heads, requests),
            block=(self.threads, 1, 1),
            cluster=(self.cluster, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.schedule.ctas_per_sm,
            use_pdl=True,
        )
        if cutlass.const_expr(self.protocol == "slots"):
            self.kernel(
                qkvz,
                qkvz_stride,
                ba,
                ba_stride,
                conv_weight,
                conv_state,
                conv_slot_stride,
                conv_channel_stride,
                conv_position_stride,
                state,
                state_slot_stride,
                a_log,
                dt_bias,
                norm_weight,
                slots,
                slots_stride,
                accepted,
                cu_seqlens,
                out,
                out_stride,
                eps,
            ).launch(**launch)
        else:
            self.kernel_records(
                qkvz,
                qkvz_stride,
                ba,
                ba_stride,
                conv_weight,
                conv_state,
                conv_slot_stride,
                conv_channel_stride,
                conv_position_stride,
                state,
                state_slot_stride,
                a_log,
                dt_bias,
                norm_weight,
                slots,
                slots_stride,
                accepted,
                cu_seqlens,
                out,
                out_stride,
                eps,
                records,
                seq_lens,
                block_size,
            ).launch(**launch)

    @cute.kernel
    def kernel(
        self,
        qkvz: Int64,
        qkvz_stride: Int32,
        ba: Int64,
        ba_stride: Int32,
        conv_weight: Int64,
        conv_state: Int64,
        conv_slot_stride: Int64,
        conv_channel_stride: Int32,
        conv_position_stride: Int32,
        state: Int64,
        state_slot_stride: Int64,
        a_log: Int64,
        dt_bias: Int64,
        norm_weight: Int64,
        slots: Int64,
        slots_stride: Int32,
        accepted: Int64,
        cu_seqlens: Int64,
        out: Int64,
        out_stride: Int32,
        eps: Float32,
    ):
        """One CTA of the `slots` protocol (`body`)."""
        self.body(
            qkvz,
            qkvz_stride,
            ba,
            ba_stride,
            conv_weight,
            conv_state,
            conv_slot_stride,
            conv_channel_stride,
            conv_position_stride,
            state,
            state_slot_stride,
            a_log,
            dt_bias,
            norm_weight,
            slots,
            slots_stride,
            accepted,
            cu_seqlens,
            out,
            out_stride,
            eps,
            Int64(0),
            Int64(0),
            Int32(0),
        )

    @cute.kernel
    def kernel_records(
        self,
        qkvz: Int64,
        qkvz_stride: Int32,
        ba: Int64,
        ba_stride: Int32,
        conv_weight: Int64,
        conv_state: Int64,
        conv_slot_stride: Int64,
        conv_channel_stride: Int32,
        conv_position_stride: Int32,
        state: Int64,
        state_slot_stride: Int64,
        a_log: Int64,
        dt_bias: Int64,
        norm_weight: Int64,
        slots: Int64,
        slots_stride: Int32,
        accepted: Int64,
        cu_seqlens: Int64,
        out: Int64,
        out_stride: Int32,
        eps: Float32,
        records: Int64,
        seq_lens: Int64,
        block_size: Int32,
    ):
        """One CTA of the `records` protocol (`body`)."""
        self.body(
            qkvz,
            qkvz_stride,
            ba,
            ba_stride,
            conv_weight,
            conv_state,
            conv_slot_stride,
            conv_channel_stride,
            conv_position_stride,
            state,
            state_slot_stride,
            a_log,
            dt_bias,
            norm_weight,
            slots,
            slots_stride,
            accepted,
            cu_seqlens,
            out,
            out_stride,
            eps,
            records,
            seq_lens,
            block_size,
        )

    @cute.jit
    def body(
        self,
        qkvz: Int64,
        qkvz_stride: Int32,
        ba: Int64,
        ba_stride: Int32,
        conv_weight: Int64,
        conv_state: Int64,
        conv_slot_stride: Int64,
        conv_channel_stride: Int32,
        conv_position_stride: Int32,
        state: Int64,
        state_slot_stride: Int64,
        a_log: Int64,
        dt_bias: Int64,
        norm_weight: Int64,
        slots: Int64,
        slots_stride: Int32,
        accepted: Int64,
        cu_seqlens: Int64,
        out: Int64,
        out_stride: Int32,
        eps: Float32,
        records: Int64,
        seq_lens: Int64,
        block_size: Int32,
    ):
        """One CTA: `cta_rows` value rows of one (request, key head)."""
        Q = self.tokens
        R = self.rows
        ROWS = self.cta_rows
        tid, _, _ = cute.arch.thread_idx()
        rank, key_head, request = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // WARP)
        group = rank // self.partitions
        v0 = (rank % self.partitions) * R
        head0 = key_head * GQA + group * self.hpc  # the CTA's first value head

        allocator = cutlass.memory.SmemAllocator()
        # Convolution outputs (BF16 values as FP32): q and k as [token][256]
        # (normalized in place), v and sigmoid(z) as [token][head][row].
        qk_smem = allocator.allocate_array(
            Float32, Q * QK_CHANNELS, byte_alignment=16
        ).toint()
        v_smem = allocator.allocate_array(Float32, Q * ROWS, byte_alignment=16).toint()
        gate_smem = allocator.allocate_array(
            Float32, Q * ROWS, byte_alignment=16
        ).toint()
        # The recurrence outputs (BF16 values as FP32), [token][head][row]:
        # read by the whole cluster for the norm.
        o_smem = allocator.allocate_array(Float32, Q * ROWS, byte_alignment=16).toint()
        # Each row's delta-rule correction, [token][head][row]: one lane of the
        # warp computes it, every lane applies it.
        delta_smem = allocator.allocate_array(
            Float32, Q * ROWS, byte_alignment=16
        ).toint()
        decay_smem = allocator.allocate_array(Float32, Q * self.hpc).toint()
        beta_smem = allocator.allocate_array(Float32, Q * self.hpc).toint()
        smem = (qk_smem, v_smem, gate_smem, o_smem, delta_smem, decay_smem, beta_smem)

        # The request's rows and slots (host-prepared before the graph).
        bos = load_s32(cu_seqlens + Int64(request) * 4)
        tokens = load_s32(cu_seqlens + Int64(request) * 4 + 4) - bos
        accepted_count = load_s32(accepted + Int64(request) * 4)
        offset = cute.min(cute.max(accepted_count, 1), Q) - 1
        slot_row = slots + Int64(request * slots_stride) * 4
        source = load_s32(slot_row + Int64(offset) * 4)
        conv_slot = load_s32(slot_row)
        live = (
            (tokens > 0)
            & (accepted_count >= 1)
            & (accepted_count <= Q)
            & (source > 0)
            & (conv_slot > 0)
        )
        args = (
            qkvz,
            qkvz_stride,
            ba,
            ba_stride,
            conv_weight,
            conv_state,
            conv_slot_stride,
            conv_channel_stride,
            conv_position_stride,
            state,
            state_slot_stride,
            a_log,
            dt_bias,
            norm_weight,
            out,
            out_stride,
            eps,
            records,
            seq_lens,
            block_size,
        )
        ctx = (tid, rank, key_head, warp, group, v0, head0, bos, tokens, request)
        if live:
            self.decode(args, ctx, smem, (offset, slot_row, source, conv_slot))
        else:
            self.pad(out, out_stride, records, ctx)

    @cute.jit
    def decode(self, args, ctx, smem, request_slots):
        """The CTA's work for a request with usable caches."""
        (
            qkvz,
            qkvz_stride,
            ba,
            ba_stride,
            conv_weight,
            conv_state,
            conv_slot_stride,
            conv_channel_stride,
            conv_position_stride,
            state,
            state_slot_stride,
            a_log,
            dt_bias,
            norm_weight,
            out,
            out_stride,
            eps,
            records,
            seq_lens,
            block_size,
        ) = args
        tid, rank, key_head, warp, group, v0, head0, bos, tokens, request = ctx
        qk_smem, v_smem, gate_smem, o_smem, delta_smem, decay_smem, beta_smem = smem
        offset, slot_row, source, conv_slot = request_slots
        Q = self.tokens
        R = self.rows
        ROWS = self.cta_rows
        THREADS = self.threads
        lane = tid % WARP

        destinations = [load_s32(slot_row + t * 4) for t in range(Q)]
        conv_base = conv_state + Int64(conv_slot) * conv_slot_stride * 2
        channel_stride = Int64(conv_channel_stride)
        position_stride = Int64(conv_position_stride)

        # 1. Everything earlier graphs wrote, before the wait.
        qk_tasks = []
        for k in cutlass.range_constexpr(self.qk_iters):
            c_qk = cute.min(tid + k * THREADS, QK_CHANNELS - 1)
            col_qk = (
                (c_qk // HEAD_DIM) * self.k_offset
                + key_head * HEAD_DIM
                + c_qk % HEAD_DIM
            )
            qk_tasks.append(
                self.conv_operands(
                    conv_base,
                    conv_weight,
                    col_qk,
                    offset,
                    channel_stride,
                    position_stride,
                )
            )
        v_tasks = []
        for k in cutlass.range_constexpr(self.v_iters):
            l_v = cute.min(cute.max(tid + (k * THREADS - self.v_first), 0), ROWS - 1)
            col_v = self.v_offset + (head0 + l_v // R) * HEAD_DIM + v0 + l_v % R
            v_tasks.append(
                self.conv_operands(
                    conv_base,
                    conv_weight,
                    col_v,
                    offset,
                    channel_stride,
                    position_stride,
                )
            )
        gate_task = cute.min(tid, Q * self.hpc - 1)
        gate_head = head0 + gate_task % self.hpc
        a_log_value = load_f32(a_log + Int64(gate_head) * 4)
        dt_value = load_bf16(dt_bias + Int64(gate_head) * 2)
        norm = [load_bf16(norm_weight + (lane + WARP * i) * 2) for i in range(4)]

        warp_head = warp // (self.warps // self.hpc)
        row0 = v0 + (warp % (self.warps // self.hpc)) * self.warp_rows
        state_lane = Int64(
            ((head0 + warp_head) * HEAD_DIM + row0) * HEAD_DIM + lane * 4
        )
        source_rows = state + (Int64(source) * state_slot_stride + state_lane) * 4
        h = [
            list(load_state_x4(source_rows + r * HEAD_DIM * 4))
            for r in range(self.warp_rows)
        ]
        # Dependents may launch now: they read this kernel's output after their
        # own wait. The trigger follows the loads so that they are issued
        # before the wait.
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()

        # 2. This step's projections. Every load first (one round trip to L2),
        # then the convolutions, the gates and the q / k norms.
        last = tokens - 1
        row_bases = [
            qkvz + Int64(bos + cute.min(t, last)) * qkvz_stride * 2 for t in range(Q)
        ]
        qk_inputs = [
            [load_bf16(row_bases[t] + Int64(qk_tasks[k][0]) * 2) for t in range(Q)]
            for k in range(self.qk_iters)
        ]
        v_inputs = [
            [load_bf16(row_bases[t] + Int64(v_tasks[k][0]) * 2) for t in range(Q)]
            for k in range(self.v_iters)
        ]
        z_values = []
        for k in cutlass.range_constexpr(self.z_iters):
            z_clamped = cute.min(tid + k * THREADS, Q * ROWS - 1)
            z_local = z_clamped % ROWS
            z_column = (
                self.z_offset + (head0 + z_local // R) * HEAD_DIM + v0 + z_local % R
            )
            z_row = bos + cute.min(z_clamped // ROWS, last)
            z_values.append(
                load_bf16(qkvz + (Int64(z_row) * qkvz_stride + z_column) * 2)
            )
        g_row = bos + cute.min(gate_task // self.hpc, last)
        g_ba = ba + Int64(g_row) * ba_stride * 2
        g_b = load_bf16(g_ba + Int64(gate_head) * 2)
        g_a = load_bf16(g_ba + Int64(self.value_heads + gate_head) * 2)

        for k in cutlass.range_constexpr(self.qk_iters):
            c_in = tid + k * THREADS
            col_in, taps_in, weights_in = qk_tasks[k]
            qk_tasks[k] = (col_in, taps_in, qk_inputs[k])
            y_qk = self.convolve(taps_in, weights_in, qk_inputs[k])
            if c_in < QK_CHANNELS:
                for tt in cutlass.range_constexpr(Q):
                    store_shared_f32(qk_smem + (tt * QK_CHANNELS + c_in) * 4, y_qk[tt])
        for k in cutlass.range_constexpr(self.v_iters):
            l_in = tid + (k * THREADS - self.v_first)
            col_vin, taps_vin, weights_vin = v_tasks[k]
            v_tasks[k] = (col_vin, taps_vin, v_inputs[k])
            y_v = self.convolve(taps_vin, weights_vin, v_inputs[k])
            if (l_in >= 0) & (l_in < ROWS):
                for tt in cutlass.range_constexpr(Q):
                    store_shared_f32(v_smem + (tt * ROWS + l_in) * 4, y_v[tt])
        for k in cutlass.range_constexpr(self.z_iters):
            z_index = tid + k * THREADS
            if z_index < Q * ROWS:
                # sigmoid(z) as the stock norm computes its gate.
                z_gate = rcp_ftz(add_ftz(Float32(1.0), exp_ftz(-z_values[k])))
                store_shared_f32(gate_smem + z_index * 4, z_gate)
        if tid < Q * self.hpc:
            g_x = add_ftz(g_a, dt_value)
            g_softplus = softplus_select(
                g_x, cute.math.log1p(exp_ftz(g_x), fastmath=False)
            )
            g_log = mul_ftz(-exp_ftz(a_log_value), g_softplus)
            store_shared_f32(decay_smem + gate_task * 4, exp_ftz(g_log))
            g_beta = rcp_ftz(add_ftz(Float32(1.0), exp_ftz(-g_b)))
            store_shared_f32(beta_smem + gate_task * 4, g_beta)
        cute.arch.sync_threads()
        if cutlass.const_expr(self.partitions == 1):
            # The convolutions consumed this CTA's q / k taps: peers may
            # rewrite the convolution state once they pass the matching wait.
            cute.arch.cluster_arrive_relaxed()

        for k in cutlass.range_constexpr(-(-2 * Q // self.warps)):
            n_task = warp + k * self.warps
            if n_task < 2 * Q:
                n_base = (
                    qk_smem
                    + ((n_task % Q) * QK_CHANNELS + (n_task // Q) * HEAD_DIM) * 4
                )
                n_values = [
                    load_shared_f32(n_base + (lane + WARP * i) * 4) for i in range(4)
                ]
                n_total = self.warp_sum(self.square_sum(n_values))
                n_factor = rsqrt_ftz(add_ftz(n_total, Float32(QK_EPS)))
                # q also takes the attention scale (an exact x1 for k).
                n_scale = select_f32(
                    Int32(n_task < Q), Float32(HEAD_DIM**-0.5), Float32(1.0)
                )
                n_factor = mul_ftz(n_factor, n_scale)
                for ii in cutlass.range_constexpr(4):
                    store_shared_f32(
                        n_base + (lane + WARP * ii) * 4, mul_ftz(n_values[ii], n_factor)
                    )
        cute.arch.sync_threads()

        # 3. The delta rule, token by token; each token's state rows go to its
        # slot, its outputs to shared memory.
        # The warp's row sums are reduce-scattered: lane L ends with the sum of
        # row L / (32 / W), shared by 32 / W lanes, the first of which stores it.
        W = self.warp_rows
        local_row = warp_head * R + row0 - v0
        lane_row = local_row + lane // (WARP // W)
        writer = lane % (WARP // W) == 0
        for t in cutlass.range_constexpr(Q):
            q_vec = load_shared_f32x4(qk_smem + (t * QK_CHANNELS + lane * 4) * 4)
            k_vec = load_shared_f32x4(
                qk_smem + (t * QK_CHANNELS + HEAD_DIM + lane * 4) * 4
            )
            decay = load_shared_f32(decay_smem + (t * self.hpc + warp_head) * 4)
            beta = load_shared_f32(beta_smem + (t * self.hpc + warp_head) * 4)
            value = load_shared_f32(v_smem + (t * ROWS + lane_row) * 4)
            key_dots = []
            for r in cutlass.range_constexpr(W):
                h01 = mul_ftz_x2((h[r][0], h[r][1]), (decay, decay))
                h23 = mul_ftz_x2((h[r][2], h[r][3]), (decay, decay))
                h[r] = [*h01, *h23]
                key_dots.append(self.dot(h[r], k_vec))
            delta = mul_ftz(sub_ftz(value, self.reduce_scatter(key_dots, lane)), beta)
            if writer:
                store_shared_f32(delta_smem + (t * ROWS + lane_row) * 4, delta)
            cute.arch.sync_warp()
            deltas = self.load_rows(delta_smem + (t * ROWS + local_row) * 4)
            query_dots = []
            for r in cutlass.range_constexpr(W):
                pair = (deltas[r], deltas[r])
                h01 = fma_ftz_x2((k_vec[0], k_vec[1]), pair, (h[r][0], h[r][1]))
                h23 = fma_ftz_x2((k_vec[2], k_vec[3]), pair, (h[r][2], h[r][3]))
                h[r] = [*h01, *h23]
                query_dots.append(self.dot(h[r], q_vec))
            query_dot = self.reduce_scatter(query_dots, lane)
            if writer:
                store_shared_f32(
                    o_smem + (t * ROWS + lane_row) * 4, round_bf16(query_dot)
                )
            # A compile-time condition around a runtime one: not one `if`.
            if cutlass.const_expr(self.protocol == "slots"):  # noqa: SIM102
                if (t < tokens) & (destinations[t] > 0):
                    dst_rows = (
                        state
                        + (Int64(destinations[t]) * state_slot_stride + state_lane) * 4
                    )
                    for r in cutlass.range_constexpr(self.warp_rows):
                        store_state_x4(dst_rows + r * HEAD_DIM * 4, tuple(h[r]))
        if cutlass.const_expr(self.partitions == 1):
            cute.arch.sync_threads()
        else:
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        # 4. The gated RMSNorm of each (token, head) over the cluster's rows
        # (this CTA's own with a single partition); this CTA writes its own.
        for k in cutlass.range_constexpr(-(-Q * self.hpc // self.warps)):
            o_task = warp + k * self.warps
            if o_task < Q * self.hpc:
                o_token = o_task // self.hpc
                o_head = o_task % self.hpc
                o_values = []
                for ii in cutlass.range_constexpr(4):
                    o_v = lane + WARP * ii
                    o_owner = group * self.partitions + o_v // R
                    o_address = o_smem + (o_token * ROWS + o_head * R + o_v % R) * 4
                    if cutlass.const_expr(self.partitions == 1):
                        o_values.append(load_shared_f32(o_address))
                    else:
                        o_values.append(load_cluster_f32(o_address, o_owner))
                o_total = self.warp_sum(self.square_sum(o_values))
                o_rstd = rsqrt_ftz(fma_ftz(o_total, Float32(1.0 / HEAD_DIM), eps))
                o_row = (
                    out
                    + (Int64(bos + o_token) * out_stride + (head0 + o_head) * HEAD_DIM)
                    * 2
                )
                for ii in cutlass.range_constexpr(4):
                    o_col = lane + WARP * ii
                    if (o_col >= v0) & (o_col < v0 + R) & (o_token < tokens):
                        o_gate = load_shared_f32(
                            gate_smem + (o_token * ROWS + o_head * R + o_col - v0) * 4
                        )
                        o_result = mul_ftz(
                            mul_ftz(mul_ftz(o_values[ii], o_rstd), norm[ii]), o_gate
                        )
                        store_bf16(o_row + o_col * 2, o_result)

        # The convolution state: [c[accepted], c[accepted + 1], x_0, ...]. Every
        # CTA of the cluster read its q / k taps before the barrier.
        if cutlass.const_expr(self.partitions == 1):
            cute.arch.cluster_wait()
        for k in cutlass.range_constexpr(self.qk_iters):
            c_w = tid + k * THREADS
            if (c_w < QK_CHANNELS) & (c_w % self.cluster == rank):
                self.store_conv_state(
                    conv_base, qk_tasks[k], tokens, channel_stride, position_stride
                )
        for k in cutlass.range_constexpr(self.v_iters):
            l_w = tid + (k * THREADS - self.v_first)
            if (l_w >= 0) & (l_w < ROWS):
                self.store_conv_state(
                    conv_base, v_tasks[k], tokens, channel_stride, position_stride
                )
        if cutlass.const_expr(self.protocol != "slots"):
            self.store_records(
                (records, seq_lens, block_size),
                ctx,
                (qk_smem, decay_smem, delta_smem),
                source,
                destinations,
                state,
                state_slot_stride,
            )
        if cutlass.const_expr(self.partitions > 1):
            # Peers read this CTA's outputs until they pass this barrier.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.jit
    def store_records(
        self, inputs, ctx, smem, source, destinations, state, state_slot_stride
    ):
        """Write the verify records of this CTA (`commit.py`) into the key
        head's band of the request's two copy slots: the cluster's first CTA
        the key head's normalized keys, the first CTA of each value-head group
        its decays, every CTA the deltas of its rows; the cluster's first CTA
        of the first key head also the request's header. `inputs`: the
        layer's header buffer, and in `align` mode the step's `seq_lens`
        (int32, by request row) and the mamba block size. Everything else is
        read from shared memory after the token loop. The records are stored
        as streaming data (evict-first): the commit reads them after every
        layer's verify, by when the state traffic has long evicted them."""
        records, seq_lens, block_size = inputs
        tid, rank, key_head, _, _, v0, head0, _, tokens, request = ctx
        qk_smem, decay_smem, delta_smem = smem
        Q = self.tokens
        R = self.rows
        ROWS = self.cta_rows
        THREADS = self.threads
        lay = self.records_layout
        # The copies (gdn_records.copy_slots): the first two slots that are not
        # null, not the source and, in align mode, not a row whose state ends
        # at a block boundary (the state after row c holds the computed tokens
        # before the step, seq_lens - tokens, plus c + 1), which the commit
        # may write as the boundary row (c <= tokens - 2).
        before = Int32(0)
        if cutlass.const_expr(self.align):
            before = load_s32(seq_lens + Int64(request) * 4) - tokens
        taken = Int32(0)
        copies = []
        for c in cutlass.range_constexpr(Q):
            usable = (destinations[c] > 0) & (destinations[c] != source)
            if cutlass.const_expr(self.align and c <= Q - 2):
                usable = usable & (
                    (tokens - 2 < c) | ((before + (c + 1)) % block_size != 0)
                )
            copies.append(usable & (taken < 2))
            taken = taken + Int32(usable)
        bands = [
            state
            + (
                Int64(destinations[c]) * state_slot_stride
                + Int64(key_head * lay.band_stride)
            )
            * 4
            for c in range(Q)
        ]
        group_head = head0 - key_head * GQA  # the CTA's first head in the band
        if rank == 0:
            if tid < Q * WARP:
                k_token = tid // WARP
                k_col = (tid % WARP) * 4
                if k_token < tokens:
                    key = load_shared_f32x4(
                        qk_smem + (k_token * QK_CHANNELS + HEAD_DIM + k_col) * 4
                    )
                    for c in cutlass.range_constexpr(Q):
                        if copies[c]:
                            store_stream_x4(
                                bands[c]
                                + Int64(lay.keys + k_token * HEAD_DIM + k_col) * 4,
                                key,
                            )
            if (key_head == 0) & (tid < 3 + Q):
                # (rows, source, slot 0, ..., slot q - 1, copies mask)
                word = select_s32(Int32(tid == 0), tokens, source)
                for t in cutlass.range_constexpr(Q):
                    word = select_s32(Int32(tid == 2 + t), destinations[t], word)
                mask = Int32(0)
                for t in cutlass.range_constexpr(Q):
                    mask = mask | select_s32(Int32(copies[t]), Int32(1 << t), Int32(0))
                word = select_s32(Int32(tid == 2 + Q), mask, word)
                store_s32(
                    records + (Int64(request) * lay.header_stride + tid) * 4, word
                )
        if (v0 == 0) & (tid < Q * self.hpc):
            d_token = tid // self.hpc
            if d_token < tokens:
                decay = load_shared_f32(decay_smem + tid * 4)
                for c in cutlass.range_constexpr(Q):
                    if copies[c]:
                        store_stream_f32(
                            bands[c]
                            + Int64(
                                lay.decays + d_token * GQA + group_head + tid % self.hpc
                            )
                            * 4,
                            decay,
                        )
        for k in cutlass.range_constexpr(-(-(Q * ROWS // 4) // THREADS)):
            quad = tid + k * THREADS
            if quad < Q * ROWS // 4:
                q_token = (quad * 4) // ROWS
                q_local = (quad * 4) % ROWS
                if q_token < tokens:
                    delta = load_shared_f32x4(delta_smem + quad * 16)
                    for c in cutlass.range_constexpr(Q):
                        if copies[c]:
                            store_stream_x4(
                                bands[c]
                                + Int64(
                                    lay.deltas
                                    + (q_token * GQA + group_head + q_local // R)
                                    * HEAD_DIM
                                    + v0
                                    + q_local % R
                                )
                                * 4,
                                delta,
                            )

    @cute.jit
    def pad(self, out, out_stride, records, ctx):
        """A request without usable caches: zero output rows, no state (and a
        header that records no rows)."""
        tid, rank, key_head, _, _, v0, head0, bos, tokens, request = ctx
        Q = self.tokens
        R = self.rows
        ROWS = self.cta_rows
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        # A compile-time condition around a runtime one: not one `if`.
        if cutlass.const_expr(self.protocol != "slots"):  # noqa: SIM102
            if (rank == 0) & (key_head == 0) & (tid == 0):
                store_s32(
                    records + Int64(request) * self.records_layout.header_stride * 4,
                    Int32(0),
                )
        for k in cutlass.range_constexpr(self.z_iters):
            p_index = tid + k * self.threads
            p_token = p_index // ROWS
            p_local = p_index % ROWS
            if (p_index < Q * ROWS) & (p_token < tokens):
                p_column = (head0 + p_local // R) * HEAD_DIM + v0 + p_local % R
                store_bf16(
                    out + (Int64(bos + p_token) * out_stride + p_column) * 2,
                    Float32(0.0),
                )

    def conv_operands(
        self, conv_base, conv_weight, column, offset, channel_stride, position_stride
    ):
        """A channel's three taps of this step and its four weights."""
        channel = conv_base + Int64(column) * channel_stride * 2
        taps = [
            load_bf16(channel + Int64(offset + i) * position_stride * 2)
            for i in range(TAPS - 1)
        ]
        return column, taps, list(load_bf16x4(conv_weight + Int64(column) * TAPS * 2))

    def convolve(self, taps, weights, inputs):
        """The `q` SiLU convolution outputs of one channel, rounded to BF16
        (the Triton kernel's sequence: BF16 products, FP32 sums from zero)."""
        window = taps + inputs
        outputs = []
        for t in range(self.tokens):
            acc = add_rn(mul_bf16(window[t], weights[0]), Float32(0.0))
            for i in range(1, TAPS):
                acc = add_rn(acc, mul_bf16(window[t + i], weights[i]))
            outputs.append(round_bf16(conv_silu(acc)))
        return outputs

    @cute.jit
    def store_conv_state(
        self, conv_base, task, tokens, channel_stride, position_stride
    ):
        """Rewrite one channel's convolution state after this step."""
        column, taps, inputs = task
        channel = conv_base + Int64(column) * channel_stride * 2
        store_bf16(channel, taps[1])
        store_bf16(channel + position_stride * 2, taps[2])
        for t in cutlass.range_constexpr(self.tokens):
            if t < tokens:
                store_bf16(channel + Int64(t + 2) * position_stride * 2, inputs[t])

    @staticmethod
    def square_sum(values):
        """`sum(v * v)` over a lane's four values, in order (contracted)."""
        total = fma_ftz(values[0], values[0], Float32(0.0))
        for value in values[1:]:
            total = fma_ftz(value, value, total)
        return total

    @staticmethod
    def dot(row, vector):
        """A lane's four-key dot product, in order (contracted)."""
        total = fma_ftz(row[0], vector[0], Float32(0.0))
        for e in range(1, 4):
            total = fma_ftz(row[e], vector[e], total)
        return total

    @staticmethod
    def warp_sum(value):
        """Butterfly sum over the warp (every lane gets the same bits)."""
        for offset in (16, 8, 4, 2, 1):
            value = add_ftz(value, cute.arch.shuffle_sync_bfly(value, offset))
        return value

    @staticmethod
    def reduce_scatter(values, lane):
        """The warp sums of `W` (a power of two) per-lane values, scattered:
        lane `L` returns the sum of value `L / (32 / W)`.

        Each butterfly step halves the values a lane carries: the lane keeps
        the half its partner does not and adds the partner's partial sums of
        it. Every sum is formed by the same pairwise additions as a full
        butterfly (`warp_sum`) per value, with `W + W / 2 + ... + 1`
        shuffles in place of `5 W`.
        """
        values = list(values)
        for offset in (16, 8, 4, 2, 1):
            half = len(values) // 2
            if half == 0:
                values = [
                    add_ftz(values[0], cute.arch.shuffle_sync_bfly(values[0], offset))
                ]
            else:
                upper = (lane // offset) % 2
                kept = [
                    select_f32(upper, values[i + half], values[i]) for i in range(half)
                ]
                sent = [
                    select_f32(upper, values[i], values[i + half]) for i in range(half)
                ]
                values = [
                    add_ftz(k, cute.arch.shuffle_sync_bfly(s, offset))
                    for k, s in zip(kept, sent)
                ]
        return values[0]

    def load_rows(self, address):
        """The `W` FP32 values of a warp's rows from shared memory."""
        if self.warp_rows < 4:
            return [load_shared_f32(address + r * 4) for r in range(self.warp_rows)]
        return [
            value
            for i in range(self.warp_rows // 4)
            for value in load_shared_f32x4(address + i * 16)
        ]


@cache
def _compiled(
    device_index: int,
    key_heads: int,
    tokens: int,
    schedule: Schedule,
    protocol: str = "slots",
    layout: RecordLayout | None = None,
    align: bool = False,
):
    """`GdnDecodeCore` compiled for a device, shape, schedule and state
    protocol (and `align` mode of the `records` protocol)."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            GdnDecodeCore(key_heads, tokens, schedule, protocol, layout, align),
            Int64(0),
            Int32(1),
            Int64(0),
            Int32(1),
            Int64(0),
            Int64(0),
            Int64(1),
            Int32(1),
            Int32(1),
            Int64(0),
            Int64(1),
            Int64(0),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(1),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(1),
            Float32(1e-6),
            Int64(0),
            Int64(0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


def gdn_decode_core(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    slots: torch.Tensor,
    accepted: torch.Tensor,
    cu_seqlens: torch.Tensor,
    requests: int,
    out: torch.Tensor,
    eps: float,
    schedule: Schedule | None = None,
    records: torch.Tensor | None = None,
    layout: RecordLayout | None = None,
    seq_lens: torch.Tensor | None = None,
    block_size: int = 0,
) -> torch.Tensor:
    """The GDN core of a decode step with speculative verification rows: the
    stock `causal_conv1d_update` + `fused_gdn_decode_post_conv_mtp` (with
    the sigmoid output gate) in one kernel, updating both caches in place.

    With `records` and `layout` the core writes no recurrent state but the
    verify records of `commit.py` (protocol `records`), which the commit
    after sampling folds into the accepted row's slot: the records into two
    of the request's non-source speculative slots of `state`, the header
    into `records`. With `seq_lens` (prefix-caching `align` mode) the
    two avoid the slot of a row that can be the commit's boundary row.

    Args:
        qkvz: The packed projection, BF16 `[rows, (2 KH + 2 HV) 128]` (q, k
            per key head, then v, z per value head), unit column stride.
        ba: The `b` and `a` logits, BF16 `[rows, 2 HV]` (b first).
        conv_weight: The convolution weight, contiguous BF16 `[conv_dim, 4]`
            (`conv_dim = (2 KH + HV) 128`).
        conv_state: The convolution cache as `[slots, conv_dim, q + 2]` BF16
            (any strides: `kv_cache[0]` or its transpose, as the stock path
            passes it).
        state: The recurrent cache, FP32 `[slots, HV, 128, 128]` with
            contiguous slots.
        a_log: FP32 `[HV]`.
        dt_bias: BF16 `[HV]`.
        norm_weight: The gated RMSNorm weight, BF16 `[128]`.
        slots: Each request's state slots, int32 `[>= requests, >= q]`.
        accepted: Accepted tokens of each request's previous step, int32.
        cu_seqlens: Row offsets of the requests, int32 `[>= requests + 1]`.
        requests: Requests of the step (the grid's last dimension).
        out: The output, BF16 `[rows, HV 128]` with unit column stride.
        eps: The gated RMSNorm epsilon.
        schedule: The cluster split; `default_schedule` when None.
        records: The layer's record header buffer (`commit.RecordLayout`),
            or None for vLLM's protocol (the state after every row to its
            slot).
        layout: The records layout (with `records`).
        seq_lens: `records` in `align` mode: each request's computed
            tokens before the step plus its rows, int32 `[>= requests]` (the
            runner's `seq_lens`).
        block_size: With `seq_lens`: the mamba block size, at least `q -
            1`.

    Returns:
        `out`.

    """
    value_heads = state.shape[1]
    key_heads = value_heads // GQA
    tokens = slots.shape[1]
    assert value_heads == GQA * key_heads and state.shape[2:] == (HEAD_DIM, HEAD_DIM)
    assert state.dtype == torch.float32 and state.stride()[1:] == (
        HEAD_DIM * HEAD_DIM,
        HEAD_DIM,
        1,
    )
    assert qkvz.dtype == ba.dtype == conv_state.dtype == torch.bfloat16
    assert qkvz.stride(1) == 1 and ba.stride(1) == 1 and out.stride(1) == 1
    assert qkvz.shape[1] == 2 * (key_heads + value_heads) * HEAD_DIM
    assert (
        conv_weight.is_contiguous()
        and conv_weight.numel() == conv_state.shape[1] * TAPS
    )
    assert conv_state.shape[2] == tokens + TAPS - 2
    assert a_log.dtype == torch.float32 and dt_bias.dtype == torch.bfloat16
    assert norm_weight.dtype == torch.bfloat16 and norm_weight.numel() == HEAD_DIM
    assert slots.dtype == accepted.dtype == cu_seqlens.dtype == torch.int32
    assert (
        slots.stride(1) == 1 and accepted.is_contiguous() and cu_seqlens.is_contiguous()
    )
    if schedule is None:
        schedule = default_schedule(key_heads, requests)
    protocol = "slots"
    if records is not None:
        assert layout is not None and layout.tokens == tokens
        assert layout.key_heads == key_heads and requests <= layout.capacity
        assert records.dtype == torch.int32 and records.numel() >= layout.words
        assert layout.band_words <= layout.band_stride
        protocol = "records"
    align = seq_lens is not None
    if seq_lens is not None:
        assert records is not None and block_size >= max(tokens - 1, 1)
        assert seq_lens.dtype == torch.int32 and seq_lens.is_contiguous()
        assert seq_lens.numel() >= requests
    _compiled(qkvz.device.index, key_heads, tokens, schedule, protocol, layout, align)(
        Int64(qkvz.data_ptr()),
        Int32(qkvz.stride(0)),
        Int64(ba.data_ptr()),
        Int32(ba.stride(0)),
        Int64(conv_weight.data_ptr()),
        Int64(conv_state.data_ptr()),
        Int64(conv_state.stride(0)),
        Int32(conv_state.stride(1)),
        Int32(conv_state.stride(2)),
        Int64(state.data_ptr()),
        Int64(state.stride(0)),
        Int64(a_log.data_ptr()),
        Int64(dt_bias.data_ptr()),
        Int64(norm_weight.data_ptr()),
        Int64(slots.data_ptr()),
        Int32(slots.stride(0)),
        Int64(accepted.data_ptr()),
        Int64(cu_seqlens.data_ptr()),
        Int64(out.data_ptr()),
        Int32(out.stride(0)),
        Float32(eps),
        Int64(0 if records is None else records.data_ptr()),
        Int64(0 if seq_lens is None else seq_lens.data_ptr()),
        Int32(block_size),
        Int32(requests),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out


def warm_up_gdn_decode_core(
    key_heads: int,
    tokens: int,
    schedules: Iterable[Schedule],
    device: torch.device,
    layout: RecordLayout | None = None,
    align: bool = False,
) -> None:
    """Compile `GdnDecodeCore` for each of `schedules` and run it once
    on scratch caches of one request, so that no CUDA graph capture compiles or
    loads it.

    Args:
        key_heads: Key heads of the rank.
        tokens: Rows per request (`num_speculative_tokens + 1`).
        schedules: The schedules the rank's decode steps use.
        device: The rank's GPU.
        layout: The records layout of the `records` protocol, or None for
            the `slots` protocol.
        align: The `records` protocol in prefix-caching `align` mode.

    """
    value_heads = GQA * key_heads
    conv_dim = (2 * key_heads + value_heads) * HEAD_DIM

    def zeros(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """A scratch tensor of zeros on `device`."""
        return torch.zeros(*shape, dtype=dtype, device=device)

    out = zeros(tokens, value_heads * HEAD_DIM)
    caches = (
        zeros(tokens, 2 * (key_heads + value_heads) * HEAD_DIM),
        zeros(tokens, 2 * value_heads),
        zeros(conv_dim, TAPS),
        # Two slots: 0 (never written) and the request's 1.
        zeros(2, tokens + TAPS - 2, conv_dim).transpose(-1, -2),
        zeros(2, value_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32),
        zeros(value_heads, dtype=torch.float32),
        zeros(value_heads),
        zeros(HEAD_DIM),
        torch.ones(1, tokens, dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.int32, device=device),
        torch.tensor([0, tokens], dtype=torch.int32, device=device),
    )
    records = None if layout is None else layout.allocate(device)[0]
    # In align mode: the request's computed tokens before the step plus its
    # rows (its only slot is the source: no copies).
    seq_lens = (
        torch.full((1,), tokens, dtype=torch.int32, device=device) if align else None
    )
    for schedule in schedules:
        gdn_decode_core(
            *caches,
            1,
            out,
            QK_EPS,
            schedule,
            records=records,
            layout=layout,
            seq_lens=seq_lens,
            block_size=tokens if align else 0,
        )
    torch.accelerator.synchronize(device)
