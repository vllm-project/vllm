# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native-score DoSA with call-local bootstrap and carry."""

import math
import threading
import weakref
from contextlib import contextmanager

import torch
from torch.cuda import CUDAGraph

from vllm import _custom_ops as stock_ops
from vllm.model_executor.layers import dosa_ops as ops
from vllm.model_executor.layers.indexer_topk import as_fp32_logits
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, get_mqa_logits_metadata

SEED = 12288
TOPK = 2048
RECENT = 1536
RECORD_CAPACITY = 114688
MIN_KEYS = 131072


def _candidate_page_budget(rows: int, keys: int) -> int:
    # Bound the shared128-ID pool from the causal visibility geometry.
    causal = (rows // 4) * ((keys - SEED + 2 * (rows - 1) + 127) // 128 + 4)
    return max(256 * rows, causal)


def _query_slab_rows(rows: int, keys: int) -> int:
    """Bound physical Q slabs by the fixed per-row record and candidate budget."""
    rows = min(triton.cdiv(rows, 8) * 8, 16384)
    page_bytes = (128 + 2 + 2) * 4
    budget = rows * (65536 * 5 + 256 * page_bytes)
    while (
        rows > 8
        and rows * RECORD_CAPACITY * 5 + _candidate_page_budget(rows, keys) * page_bytes
        > budget
    ):
        rows = max(8, rows // 16 * 8)
    return rows


def _destroy_stream(runtime, raw, device: torch.device):
    with torch.accelerator.device_index(device.index):
        runtime.cudaStreamDestroy(raw)


class _Graph(torch.cuda.CUDAGraph):
    capture_stream = None
    # Python reset rearms retained resources; finalization only resets native nodes.
    _native_reset = torch._C._CUDAGraph.reset

    def __del__(self):
        # Reset native nodes before releasing the unpooled capture stream.
        stream = self.capture_stream
        try:
            super().__del__()
        finally:
            self._native_reset()
            _ = stream


@contextmanager
def capture(device: torch.device, *, keep_graph=True):
    """Use an unpooled stream so graph reset cannot clear another BLAS cache."""
    from cuda.bindings import runtime

    with torch.accelerator.device_index(device.index):
        status, raw = runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
        if status != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA capture stream creation failed: {status}")
        stream = torch.cuda.ExternalStream(int(raw), device=device)
    cleanup = weakref.finalize(stream, _destroy_stream, runtime, raw, device)
    object.__setattr__(cleanup, "atexit", False)
    caller = torch.cuda.current_stream(device)
    stream.wait_stream(caller)
    graph = _Graph(keep_graph=keep_graph)
    graph.capture_stream = stream
    torch.accelerator.synchronize(device)
    with torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode="global")
        try:
            yield graph
        finally:
            graph.capture_end()
    caller.wait_stream(stream)


class Recovery:
    """Raw capacity recovery requires SAFE; other failures use native MX/BF16.

    Stream-ordered children share scratch and publish permuted K IDs. Invalid
    intervals stay empty with sticky bit2; the owner retains all graph resources.
    """

    def __init__(self, owner: "DoSAWorkspace") -> None:
        self.capacity = min(owner.rows, 512)
        q, h = self.capacity, owner.heads
        self.rows = owner._empty((q,), torch.int32)
        self.state = owner._empty((2,), torch.int32)
        self.packed_q = owner._empty((q, h, 128), torch.float8_e4m3fn)
        self.packed_w = owner._empty((q, h))
        self.lengths = owner._empty((q,), torch.int32)
        self.selected = owner._empty((q, TOPK), torch.int32)
        self.zero = torch.zeros(q, device=owner.device, dtype=torch.int32)

    def prepare_native(self, owner: "DoSAWorkspace") -> None:
        self.unit_q = owner._empty(self.packed_w.shape, torch.int32).fill_(0x7F7F7F7F)
        self.weights = owner._empty(self.packed_w.shape, torch.bfloat16)
        self.starts = torch.empty_like(self.lengths)
        self.ends = torch.empty_like(self.starts)
        self.packed = owner._empty((owner.max_k,), torch.int32)

    def __call__(self, owner: "DoSAWorkspace", begin: int, end: int, *, native=False):
        n, rows = end - begin, slice(begin, end)
        status = (owner.slow_error if native else owner.fast_error)[rows]
        if native:
            self.weights[:n].copy_(owner.w[rows])
            ops.recovery_intervals[(triton.cdiv(n, 128),)](
                owner.starts[rows],
                owner.ends[rows],
                status,
                self.starts,
                self.ends,
                self.lengths,
                owner.live_nk,
                n,
                owner.max_k,
            )
            metadata = get_mqa_logits_metadata(
                self.starts[:n], self.ends[:n], owner.max_k, owner.heads
            )
            logits = fp8_fp4_mqa_logits(
                (owner.q[rows].view(torch.float8_e4m3fn), self.unit_q[:n]),
                (owner.pk, self.packed),
                self.weights[:n],
                self.starts[:n],
                self.ends[:n],
                owner.max_k,
                schedule_meta=metadata,
            )
            scores = as_fp32_logits(logits, self.lengths[:n])
            stock_ops.top_k_per_row_prefill(
                scores,
                self.zero[:n],
                self.lengths[:n],
                self.selected[:n],
                n,
                scores.stride(0),
                scores.stride(1),
                TOPK,
            )
            origin = self.starts[:n]
        else:
            ops.recovery_row_map[(1,)](
                status,
                owner.ends[rows],
                self.rows,
                self.state,
                self.lengths,
                n,
                triton.next_power_of_2(self.capacity),
            )
            ops.recovery_gather_rows[(self.capacity,)](
                owner.q[rows],
                owner.w[rows],
                self.rows,
                self.state,
                self.packed_q.view(torch.uint8),
                self.packed_w,
                owner.heads,
            )
            owner._op(
                "sample_recover_score",
                self.packed_q,
                owner.pk,
                owner.ps,
                self.packed_w,
                owner.replay_scores[:n],
                self.state,
                self.rows,
                owner.starts[rows],
                owner.ends[rows],
                owner.live_nk,
                owner.sms,
            )
            owner._topk(
                owner.replay_scores[:n],
                self.zero[:n],
                self.lengths[:n],
                self.selected[:n],
            )
            origin = self.zero[:n]
        ops.recovery_publish[(n, TOPK // 256)](
            self.selected[:n],
            status,
            owner.order[rows],
            origin,
            owner.output,
            TOPK,
            owner.output.stride(0),
        )


class DoSAWorkspace:
    """Fixed-capacity scratch for one serialized WorkspaceManager execution slot.

    max_queries bounds logical calls; keys support H32/H64, D128 and K<=2**20.
    prepare() captures every branch before KV sizing. Physical
    slabs repeat the last active query in padding, without output or carry votes.
    """

    def __init__(
        self, device: torch.device, *, max_queries: int, max_keys: int, heads: int
    ) -> None:
        if heads not in (32, 64) or max_queries <= 0 or not SEED <= max_keys <= 1048576:
            raise ValueError(
                "DoSA requires H32/H64, positive Q capacity and K12288..2**20"
            )
        import vllm._dosa_C  # noqa: F401

        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.accelerator.current_device_index())
        self.heads = heads
        self.max_queries = max_queries
        self.max_k = triton.cdiv(max_keys, 128) * 128
        self.max_q = _query_slab_rows(max_queries, self.max_k)
        self.rows = self.max_q
        self.max_pages = _candidate_page_budget(self.rows, self.max_k)
        self.high_paid = 28 if heads == 64 else 20
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.graph: CUDAGraph | None = None
        self.recovery_graphs: tuple[CUDAGraph, CUDAGraph] | None = None
        self.native_recovery_graphs: tuple[CUDAGraph, CUDAGraph] | None = None
        self.high_graph: CUDAGraph | None = None
        self.plan_epoch = 0
        self.lock = threading.RLock()
        self._allocate()
        self.done = torch.cuda.Event()
        self.done.record(torch.cuda.current_stream(self.device))

    def _empty(
        self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.empty(shape, device=self.device, dtype=dtype)

    def _kernel(self, name: str):
        return getattr(torch.ops._C, f"dosa_h{self.heads}_{name}")

    def _op(self, name: str, *args):
        return self._kernel(name)(*args)

    def _allocate(self) -> None:
        q, k, h = self.max_q, self.max_k, self.heads
        self.q = self._empty((q, h, 128), torch.uint8)
        self.w = self._empty((q, h), torch.float32)
        self.starts = self._empty((q,), torch.int32)
        self.ends = self._empty((q,), torch.int32)
        self.budget8 = self._empty((q // 8,), torch.int32)
        self.budget4 = self._empty((q // 4,), torch.int32)
        self.capture = self._empty((q, 129), torch.int64 if h == 64 else torch.int32)
        self.safe_tau = self._empty((q,), torch.float32)
        self.seed_tau = self._empty((q,), torch.float32)
        self.error = self._empty((q,), torch.int32)
        self.fast_error = self._empty((q,), torch.int32)
        self.slow_error = self._empty((q,), torch.int32)
        self.seed_status = self._empty((q,), torch.int32)
        self.values = self._empty((q, RECORD_CAPACITY), torch.uint8)
        self.ids = self._empty((q, RECORD_CAPACITY), torch.int32)
        self.counts = self._empty((q,), torch.int32)
        self.order = self._empty((q,), torch.int64)
        self.partition = self._empty((q,), torch.int32)
        self.output = self._empty((q, TOPK), torch.int32)
        self.error_original = self._empty((q,), torch.int32)
        self.counts_original = self._empty((q,), torch.int32)
        self.zero = torch.zeros(q, device=self.device, dtype=torch.int32)
        self.replay_scores = self._empty((min(q, 512), k))
        self.recovery = Recovery(self)
        self.q_input = self._empty((q, h, 128), torch.float8_e4m3fn)
        self.w_input = self._empty((q, h))
        self.start_input = self._empty((q,), torch.int32)
        self.end_input = self._empty((q,), torch.int32)
        # All TMA-read padding is initialized, including unused physical K tiles.
        self.pk = self._empty((k, 128), torch.float8_e4m3fn).zero_()
        self.ps = self._empty((k,)).zero_()
        self.rho = self._empty((k,), torch.int32).zero_()
        self.live_nk = self._empty((1,), torch.int32)
        self.input_scale = self._empty((k,)).zero_()
        self.sample_ids = (
            32 + torch.arange(128, device=self.device, dtype=torch.int64) * 64
        )
        self.sample_k = self._empty((128, 128), torch.uint8)
        self.plans = {}
        for paid in (14, 16, self.high_paid):
            self.plans[paid] = [
                self._empty((q, paid + 2, 128), torch.uint8),
                self._empty((q, paid + 2)),
                self._empty((q, 64 if paid <= 16 else 96)),
                self._empty((q,)),
                self._empty((q,)),
                self._empty((q,), torch.int32),
            ]
        self.keep_words = self._empty(
            (q // 4, triton.cdiv(k - SEED, 32), 4), torch.int32
        )
        self.id_pool = self._empty((self.max_pages, 128), torch.int32)
        chunk_capacity = max(1, self.max_pages // 32)
        self.solo_tasks = self._empty((chunk_capacity, 3), torch.int32)
        self.common_tasks = self._empty((chunk_capacity, 3), torch.int32)
        self.pool_head = torch.zeros(1, device=self.device, dtype=torch.int64)
        self.solo_state = torch.zeros(2, device=self.device, dtype=torch.int32)
        self.common_state = torch.zeros(2, device=self.device, dtype=torch.int32)
        self.schedule = {
            paid: (
                self._empty((q // 8,), torch.int32),
                self._empty((q // 8,), torch.int32),
                self._empty((2,), torch.int32),
            )
            for paid in (14, 16)
        }
        words = self._op("high_packed_record_words")
        self.packed_record = self._empty((q // 4, words), torch.int32)
        self.high_budget = self._empty((q // 4,), torch.int32)
        workers = min(self.sms, triton.cdiv(k, 1024))
        self.key_max = self._empty((workers,))
        self.key_error = self._empty((workers,), torch.int32)
        self.scale_max = self._empty((1,))
        self.scale_error = self._empty((1,), torch.int32)
        self.pair_rank = self._empty((k,), torch.int32)
        self.seed_logits = self._empty((q, SEED))
        self.hot_epoch = torch.zeros(k, device=self.device, dtype=torch.int32)
        self.permutation = torch.arange(k, device=self.device, dtype=torch.int32)
        self.swap_a = self._empty((SEED,), torch.int32)
        self.swap_b = self._empty((SEED,), torch.int32)
        self.plan_counts = torch.zeros(3, device=self.device, dtype=torch.int32)
        self.hot = self._empty((SEED,), torch.int64)
        self.cold_q = self._empty((4, h, 128), torch.uint8)
        self.cold_w = self._empty((4, h))
        self.cold_scores = self._empty((4, k))
        self.cold_ids = self._empty((4, TOPK), torch.int32)
        self.cold_zero = torch.zeros(4, device=self.device, dtype=torch.int32)
        self.cold_ends = self._empty((4,), torch.int32)
        self.votes = torch.zeros(k, device=self.device, dtype=torch.int32)
        self.carry_partial = self._empty((128, RECENT + 1), torch.int16)
        self.carry_state = torch.zeros(136, device=self.device, dtype=torch.int32)
        self.topk_state = self._empty((1024 * 1024,), torch.uint8).zero_()

    def _topk(self, scores, offsets, lengths, output):
        self.topk.radix_topk_ragged_transform(
            scores,
            output,
            offsets,
            lengths,
            self.topk_state,
            TOPK,
            False,
            0,
            True,
            row_starts=None,
        )
        return output

    def _seed(self, q):
        self._op(
            "sample_dense_score",
            q,
            self.pk[:SEED],
            self.ps[:SEED],
            self.w,
            self.seed_logits,
            self.sms,
        )
        self._op(
            "select_prepare5",
            self.seed_logits,
            self.values,
            self.ids,
            self.counts,
            self.seed_tau,
            self.error,
            self.seed_status,
        )

    def _metadata(self):
        workers = len(self.key_max)
        ops.key_parts_live[(workers,)](
            self.ps,
            self.rho,
            self.key_max,
            self.key_error,
            self.live_nk,
            self.max_k,
            SEED,
            workers,
            1024,
            num_warps=4,
        )
        ops.key_finish[(1,)](
            self.key_max,
            self.key_error,
            self.scale_max,
            self.scale_error,
            workers,
            triton.next_power_of_2(workers),
            num_warps=4,
        )
        dot = math.nextafter(128 * 2.0**-23 / (1 - 128 * 2.0**-23), math.inf)
        steps = self.heads // 4 + 3
        g = math.nextafter(steps * 2.0**-24 / (1 - steps * 2.0**-24), math.inf)
        bf16_factor = 1.0
        for _ in range(steps):
            bf16_factor = math.nextafter(bf16_factor * (1 + 2.0**-8), math.inf)
        ops.query_check[(self.rows,)](
            self.w,
            self.plans[14][5],
            self.plans[16][5],
            self.plans[self.high_paid][5],
            self.budget8,
            self.scale_max,
            self.scale_error,
            self.seed_status,
            self.seed_tau,
            self.error,
            self.safe_tau,
            8,
            dot,
            g,
            bf16_factor,
            self.heads,
            self.high_paid,
            num_warps=1,
            enable_fp_fusion=False,
        )

    def _emit(self, paid, x):
        args = (
            x,
            self.error,
            SEED,
            self.keep_words,
        )
        if paid in (14, 16):
            ids, inverse, config = self.schedule[paid]
            groups = self.rows // 8
            grid = (2 if paid == 14 else 1) * max(groups, 8 * self.sms)
            ops.prepare_live[(1,)](
                self.budget4,
                self.live_nk,
                ids[:groups],
                inverse[:groups],
                config,
                groups,
                grid,
                self.sms,
                self.max_k,
                SEED,
                256 if paid == 14 else 128,
                paid,
                triton.next_power_of_2(groups),
                num_warps=4,
            )
            method = (
                self._kernel("low_emit_h16_indexed_trusted_live_k")
                if paid == 14
                else self._kernel("low_emit_h18_indexed_trusted_live_k")
            )
            method(*args, ids[:groups], inverse[:groups], config, grid, self.live_nk)
            return
        record = self.packed_record[: self.rows // 4]
        admitted = self.high_budget[: self.rows // 4]
        self._kernel("high_prepare_fp16_h22_live_k")(
            x, self.error, record, admitted, self.keep_words, SEED, self.live_nk
        )
        packed_x = list(x)
        packed_x[13] = admitted
        self.high_args = (packed_x, *args[1:], record, self.live_nk)
        if self.high_graph is not None and torch.cuda.is_current_stream_capturing():
            self._kernel("select_attach_if_any")(
                admitted, 14, self.high_graph.raw_cuda_graph()
            )
        else:
            self._kernel("high_emit_fp16_candidates_live_k")(*self.high_args)

    def _recover_rows(self, native):
        if native:
            ops.recovery_pack_scales[(triton.cdiv(self.max_k, 256),)](
                self.ps, self.recovery.packed, self.max_k, 256
            )
        for begin in range(0, self.rows, 512):
            self.recovery(self, begin, min(self.rows, begin + 512), native=native)

    def _recover(self):
        ops.recovery_classify_errors[(triton.cdiv(self.rows, 128),)](
            self.safe_tau,
            self.error,
            self.starts,
            self.ends,
            self.live_nk,
            self.fast_error,
            self.slow_error,
            self.rows,
            self.max_k,
        )
        for graphs, native, status in (
            (self.recovery_graphs, False, self.fast_error),
            (self.native_recovery_graphs, True, self.slow_error),
        ):
            if graphs is None:
                self._recover_rows(native)
            elif torch.cuda.is_current_stream_capturing():
                self._kernel("select_attach_if_any")(
                    status, 0, graphs[0].raw_cuda_graph()
                )
            else:
                graphs[1].replay()
        ops.recovery_merge_errors[(triton.cdiv(self.rows, 128),)](
            self.error,
            self.fast_error,
            self.slow_error,
            self.rows,
        )

    def _body(self):
        for value in (self.error, self.pool_head, self.solo_state, self.common_state):
            value.zero_()
        ops.check_live_extent[(triton.cdiv(self.rows, 256),)](
            self.error, self.live_nk, self.rows, self.max_k, SEED, 256
        )
        positive = (self.w_input[: self.rows] > 0).sum(1)
        tags = (
            (positive > 14).long()
            + (positive > 16).long()
            + (positive > self.high_paid).long()
        )
        self.order.copy_(tags.argsort(stable=True))
        for source, target in (
            (self.q_input.view(torch.uint8), self.q),
            (self.w_input, self.w),
            (self.start_input, self.starts),
            (self.end_input, self.ends),
        ):
            torch.index_select(source[: self.rows], 0, self.order, out=target)
        maximum = positive.index_select(0, self.order).reshape(-1, 8).max(1).values
        self.budget8.copy_(
            torch.where(
                maximum <= 14, 14, torch.where(maximum <= 16, 16, self.high_paid)
            )
        )
        self.budget4.copy_(self.budget8.repeat_interleave(2))
        q = self.q.view(torch.float8_e4m3fn)
        self._seed(q)
        torch.index_select(
            self.pk.view(torch.uint8), 0, self.sample_ids, out=self.sample_k
        )
        self._kernel("sample_sign_masks")(
            q,
            self.sample_k.view(torch.float8_e4m3fn),
            self.capture[:, :128],
            self.sms,
        )
        self.capture[:, 128].copy_(self.error)
        for paid in (14, 16, self.high_paid):
            planner = "direct_abs" if paid in (14, 16) else "planner"
            self._kernel(f"{planner}_plan128")(
                paid, q, self.w, self.capture, self.plans[paid], self.budget8
            )
        self._metadata()
        for paid in (14, 16, self.high_paid):
            fields = self.plans[paid]
            x = [
                self.q,
                self.pk.view(torch.uint8),
                self.ps,
                self.rho,
                self.w,
                *fields[:6],
                self.starts,
                self.ends,
                self.budget4,
                self.safe_tau,
            ]
            self._emit(paid, x)
        self._kernel("compact_compact_spans_live_nk")(
            self.id_pool,
            self.pool_head,
            self.solo_tasks,
            self.common_tasks,
            self.solo_state,
            self.common_state,
            self.error,
            self.keep_words,
            self.budget4,
            SEED,
            self.max_k,
            self.starts,
            self.ends,
            self.live_nk,
        )
        self._kernel("refine_validate_seed5_live_nk")(
            self.ids,
            self.counts,
            self.seed_tau,
            self.safe_tau,
            self.starts,
            self.ends,
            self.error,
            self.max_k,
            self.live_nk,
            SEED,
        )
        for mode, tasks, state in (
            ("solo", self.solo_tasks, self.solo_state),
            ("common", self.common_tasks, self.common_state),
        ):
            self._kernel(f"refine_exact_span_{mode}_live_nk")(
                q,
                self.pk,
                self.ps,
                self.w,
                self.safe_tau,
                self.id_pool,
                tasks,
                state,
                self.values,
                self.ids,
                self.counts,
                0,
                self.sms,
                self.starts,
                self.ends,
                self.error,
                self.live_nk,
                SEED,
            )
        self._kernel("select_select_dense5")(
            self.values,
            self.ids,
            self.counts,
            self.starts,
            self.ends,
            self.error,
            self.output,
            self.order,
            self.permutation,
            self.partition,
            self.max_k,
        )
        self._recover()
        self.counts_original.index_copy_(0, self.order, self.counts)
        self.error_original.index_copy_(0, self.order, self.error)

    def prepare(self) -> None:
        from flashinfer.topk import get_topk_module

        if self.graph is not None:
            return
        self.topk = get_topk_module()
        self.recovery.prepare_native(self)
        ops.decode_ue8m0[(triton.cdiv(self.max_k, 256),)](
            self.ps.view(torch.uint8), self.input_scale, self.max_k, 256
        )
        self.ps[:SEED].fill_(1)
        self.q_input.zero_()
        self.w_input.zero_()
        self.start_input.zero_()
        self.end_input.fill_(SEED)
        self.live_nk.fill_(SEED)
        for weights in (self.w_input, self.w_input.to(torch.bfloat16)):
            self._bootstrap(self.q_input, weights, self.pk[:SEED], self.ps[:SEED], SEED)
            self._stage(self.q_input, weights, self.start_input, self.end_input, 0)
        # Profiling uses disjoint seed inputs, as every real gather does.
        self._gather(self.pk[:SEED].clone(), self.ps[:SEED].clone(), SEED, SEED)
        self._body()
        branches = []
        for native, status in ((True, self.slow_error), (False, self.fast_error)):
            with capture(self.device) as child:
                self._recover_rows(native)
            with capture(self.device, keep_graph=False) as dispatch:
                self._op("select_attach_if_any", status, 0, child.raw_cuda_graph())
            branches.append((child, dispatch))
        self.native_recovery_graphs, self.recovery_graphs = branches
        with capture(self.device) as high_graph:
            self._op("high_emit_fp16_candidates_live_k", *self.high_args)
        self.high_graph = high_graph
        with capture(self.device, keep_graph=False) as graph:
            self.graph = graph
            self._body()
        graph.replay()
        recent = min(self.max_q, RECENT)
        self._map(self.rows, SEED, self.votes[:SEED], recent)
        self._carry(self.votes[:SEED], recent)
        self.done.record(torch.cuda.current_stream(self.device))

    def _bootstrap(self, q, weights, keys, scales, common_end: int) -> None:
        ops._spread_queries[(4,)](
            q.view(torch.uint8),
            weights,
            self.cold_q,
            self.cold_w,
            len(q),
            self.heads,
            num_warps=4,
        )
        self._op(
            "sample_dense_score",
            self.cold_q.view(torch.float8_e4m3fn),
            keys,
            scales,
            self.cold_w,
            self.cold_scores,
            self.sms,
        )
        self.cold_ends.fill_(common_end)
        ids = self._topk(
            self.cold_scores, self.cold_zero, self.cold_ends, self.cold_ids
        )
        votes = self.votes[:common_end]
        votes.zero_()
        ops._add_votes[(triton.cdiv(4 * TOPK, 256),)](
            ids, votes, 4 * TOPK, common_end, 256
        )
        self._carry(votes, 4)

    def _gather(self, keys, scales, count: int, common_end: int) -> None:
        """Publish current permutation from disjoint pair ranks before K gather."""
        if self.plan_epoch + 1 >= 1 << 30:
            self.hot_epoch.zero_()
            self.plan_epoch = 0
        self.plan_epoch += 1
        state = (
            self.hot_epoch[:count],
            self.pair_rank[:count],
            self.permutation[:count],
            self.swap_a,
            self.swap_b,
            self.plan_counts,
        )
        with torch.accelerator.device_index(self.device.index):
            grid = (triton.cdiv(SEED, 256),)
            ops._mark_hot[grid](
                self.hot,
                state,
                SEED,
                common_end,
                self.plan_epoch,
                256,
                num_warps=8,
            )
            ops._compact_pairs[grid](
                self.hot,
                state,
                SEED,
                common_end,
                self.plan_epoch,
                256,
                num_warps=8,
            )
            ops._publish_gather[(triton.cdiv(count, 32),)](
                keys[:count].view(-1).view(torch.uint8),
                scales[:count],
                self.pk[:count].view(-1).view(torch.uint8),
                self.ps[:count],
                self.rho[:count],
                state,
                count,
                SEED,
                self.plan_epoch,
                32,
                num_warps=8,
            )
        self.live_nk.fill_(count)

    def _carry(self, votes, max_vote):
        """Select HOT by exact vote cutoff, small-ID ties and ascending IDs."""
        count, stride = votes.numel(), self.carry_partial.stride(0)
        blocks = triton.cdiv(count, 8192)
        with torch.accelerator.device_index(self.device.index):
            ops._carry_histogram[(blocks,)](
                votes,
                self.carry_partial,
                count,
                0,
                max_vote,
                stride,
                8 if max_vote <= 4 else triton.next_power_of_2(stride),
                num_warps=8,
            )
            ops._carry_cutoff[(1,)](
                self.carry_partial,
                self.carry_state,
                min(SEED, count),
                max_vote,
                blocks,
                stride,
                triton.next_power_of_2(self.carry_partial.shape[0]),
                num_warps=8,
            )
            ops._carry_emit[(blocks,)](
                votes, self.hot, self.carry_state, count, 0, max_vote, num_warps=8
            )

    def _map(self, logical, count, votes, recent):
        with torch.accelerator.device_index(self.device.index):
            ops._map_votes[(min(triton.cdiv(logical * TOPK, 256), self.sms * 8),)](
                self.output[:logical],
                self.permutation[:count],
                self.zero[:logical],
                votes,
                logical * TOPK,
                logical,
                count,
                TOPK,
                votes.numel(),
                recent,
                num_warps=8,
            )

    def _stage(self, q, weights, starts, ends, key_offset: int) -> None:
        logical = len(q)
        for target, source in (
            (self.q_input, q),
            (self.w_input, weights),
            (self.start_input, starts),
            (self.end_input, ends),
        ):
            target[:logical].copy_(source)
            if logical < self.max_q:
                target[logical:].copy_(
                    target[logical - 1 : logical].expand(
                        self.max_q - logical, *target.shape[1:]
                    )
                )
        if key_offset:
            self.start_input.sub_(key_offset)
            self.end_input.sub_(key_offset)

    def run(
        self,
        q: torch.Tensor,
        keys: torch.Tensor,
        scales: torch.Tensor,
        weights: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        output: torch.Tensor,
        *,
        common_end: int,
        causal: bool = True,
        key_offset: int = 0,
    ) -> bool:
        """Write native MX/BF16 top2048 request-local K IDs.

        Q/K are contiguous E4M3 [Q,H,128]/[K,128]. Contiguous scales are exact
        FP32 UE8M0 decodes or I32 packed repeated bytes, copied to aligned scratch.
        Weights are contiguous BF16/FP32 [Q,H] with the native already-rounded
        BF16 values; promotion to owned FP32 storage is exact. Contiguous I32[Q]
        intervals are rebased by key_offset in owned staging, inside [0,K].
        common_end is visible to every query. Causal intervals are
        [0,common_end+row); noncausal calls retain the frozen bootstrap.
        Output is disjoint I32[Q,2048], with column stride1 and row stride>=2048.
        prepare() must precede calls. Calls serialize; borrowed storage must
        remain stable through stream completion. Unsupported metadata returns
        False before changing output or state.
        """
        if q.ndim != 3 or keys.ndim != 2 or torch.cuda.is_current_stream_capturing():
            return False
        rows, count = len(q), len(keys)
        if self.graph is None:
            raise RuntimeError(
                "DoSAWorkspace.prepare() must run before KV-cache sizing"
            )
        for value, shape, dtypes in (
            (q, (rows, self.heads, 128), (torch.float8_e4m3fn,)),
            (keys, (count, 128), (torch.float8_e4m3fn,)),
            (scales, (count,), (torch.float32, torch.int32)),
            (weights, (rows, self.heads), (torch.float32, torch.bfloat16)),
            (starts, (rows,), (torch.int32,)),
            (ends, (rows,), (torch.int32,)),
        ):
            if (
                value.shape != shape
                or value.dtype not in dtypes
                or value.device != self.device
                or not value.is_contiguous()
            ):
                return False
        if (
            output.dtype != torch.int32
            or output.shape != (rows, TOPK)
            or output.device != self.device
            or output.stride(1) != 1
            or output.stride(0) < TOPK
            or keys.data_ptr() % 16
            or key_offset < 0
            or not 0 < rows <= self.max_queries
            or not SEED <= common_end <= count <= self.max_k
            or (causal and common_end + rows - 1 > count)
        ):
            return False
        with self.lock, torch.accelerator.device_index(self.device.index):
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(self.done)
            for value in (q, keys, scales, weights, starts, ends, output):
                value.record_stream(stream)
            try:
                decoded = self.input_scale[:count]
                if scales.dtype == torch.int32:
                    ops.decode_ue8m0[(triton.cdiv(count, 256),)](
                        scales.view(torch.uint8), decoded, count, 256
                    )
                else:
                    decoded.copy_(scales)
                scales = decoded
                self._bootstrap(q, weights, keys, scales, common_end)
                if not causal:
                    self._gather(keys, scales, count, common_end)
                for begin in range(0, rows, self.max_q):
                    end = min(begin + self.max_q, rows)
                    logical = end - begin
                    if causal:
                        visible = common_end + end - 1
                        self._gather(keys, scales, visible, common_end + begin)
                    self._stage(
                        q[begin:end],
                        weights[begin:end],
                        starts[begin:end],
                        ends[begin:end],
                        key_offset,
                    )
                    self.graph.replay()
                    recent = min(logical, RECENT) if causal and end < rows else 0
                    votes = (
                        self.votes[: common_end + end - 1] if recent else self.votes[:0]
                    )
                    votes.zero_()
                    self._map(logical, visible if causal else count, votes, recent)
                    output[begin:end].copy_(self.output[:logical])
                    if recent:
                        self._carry(votes, recent)
                return True
            finally:
                self.done.record(stream)
