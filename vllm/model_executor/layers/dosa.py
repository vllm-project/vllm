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
        zeros, lengths, selected = self.zero[:n], self.lengths[:n], self.selected[:n]
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
            starts, ends = self.starts[:n], self.ends[:n]
            metadata = get_mqa_logits_metadata(starts, ends, owner.max_k, owner.heads)
            logits = fp8_fp4_mqa_logits(
                (owner.q[rows].view(torch.float8_e4m3fn), self.unit_q[:n]),
                (owner.pk, self.packed),
                self.weights[:n],
                starts,
                ends,
                owner.max_k,
                schedule_meta=metadata,
            )
            scores = as_fp32_logits(logits, lengths)
            stock_ops.top_k_per_row_prefill(
                scores,
                zeros,
                lengths,
                selected,
                n,
                scores.stride(0),
                scores.stride(1),
                TOPK,
            )
            origin = starts
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
            scores = owner.replay_scores[:n]
            owner._op(
                "sample_recover_score",
                self.packed_q,
                owner.pk,
                owner.ps,
                self.packed_w,
                scores,
                self.state,
                self.rows,
                owner.starts[rows],
                owner.ends[rows],
                owner.live_nk,
                owner.sms,
            )
            owner._topk(scores, zeros, lengths, selected)
            origin = zeros
        ops.recovery_publish[(n, TOPK // 256)](
            selected,
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
        self.tier_counts = self._empty((triton.cdiv(q, 128),), torch.int32)
        # All TMA-read padding is initialized, including unused physical K tiles.
        self.pk = self._empty((k, 128), torch.float8_e4m3fn).zero_()
        self.ps = self._empty((k,)).zero_()
        self.rho = self._empty((k,), torch.int32).zero_()
        self.live_nk = self._empty((1,), torch.int32)
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
        self.seed_logits = self._empty((q, SEED), torch.bfloat16)
        self.hot_epoch = torch.zeros(k, device=self.device, dtype=torch.int32)
        self.permutation = torch.arange(k, device=self.device, dtype=torch.int32)
        self.swap_a = self._empty((SEED,), torch.int32)
        self.swap_b = self._empty((SEED,), torch.int32)
        self.plan_counts = torch.zeros(3, device=self.device, dtype=torch.int32)
        self.hot = self._empty((SEED,), torch.int64)
        self.cold_q = self._empty((4, h, 128), torch.uint8)
        self.cold_w = self._empty((4, h))
        self.cold_scores = self._empty((4, k), torch.bfloat16)
        self.cold_ids = self._empty((4, TOPK), torch.int32)
        self.cold_zero = torch.zeros(4, device=self.device, dtype=torch.int32)
        self.cold_ends = self._empty((4,), torch.int32)
        self.votes = torch.zeros(k, device=self.device, dtype=torch.int32)
        self.carry_partial = self._empty((128, RECENT + 1), torch.int16)
        self.carry_state = torch.zeros(136, device=self.device, dtype=torch.int32)
        self.topk_state = self._empty((1024 * 1024,), torch.uint8).zero_()

    def _topk(self, scores, offsets, lengths, output, *, eager=False) -> None:
        self.topk.radix_topk_ragged_transform(
            scores,
            output,
            offsets,
            lengths,
            self.topk_state,
            TOPK,
            False,
            0,
            not eager or torch.cuda.is_current_stream_capturing(),
            row_starts=None,
        )

    def _record_state(self):
        return self.values, self.ids, self.counts

    def _row_domain(self):
        return self.starts, self.ends, self.error

    def _dense_score(self, q, k, scales, weights, output):
        self._op("sample_dense_score", q, k, scales, weights, output, self.sms)

    def _seed(self, q):
        self._dense_score(q, self.pk[:SEED], self.ps[:SEED], self.w, self.seed_logits)
        self._op(
            "select_prepare5",
            self.seed_logits,
            *self._record_state(),
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
            self.capture[:, 128],
            self.safe_tau,
            self.capture.stride(0),
            8,
            dot,
            g,
            bf16_factor,
            self.heads,
            self.high_paid,
            num_warps=1,
            enable_fp_fusion=False,
        )

    def _screen_args(self, paid):
        inputs = self.q, self.pk.view(torch.uint8), self.ps, self.rho, self.w
        # Native plan order: fine Q, scales, paid coefficients, raw/out bounds, hints.
        bounds = self.starts, self.ends, self.budget4, self.safe_tau
        tensors = [*inputs, *self.plans[paid], *bounds]
        return tensors, self.error, SEED, self.keep_words

    def _emit_low(self, paid):
        schedule = self.schedule[paid]
        groups = self.rows // 8
        grid = (2 if paid == 14 else 1) * max(groups, 8 * self.sms)
        ops.prepare_live[(1,)](
            self.budget4,
            self.live_nk,
            *schedule,
            groups,
            grid,
            self.sms,
            self.max_k,
            SEED,
            256 if paid == 14 else 128,
            paid,
            triton.next_power_of_2(groups),
            WHOLE_WAVES=True,
            num_warps=4,
        )
        self._kernel(f"low_emit_h{paid + 2}_indexed_trusted_live_k")(
            *self._screen_args(paid),
            *schedule,
            grid,
            self.live_nk,
            self.sms,
            validated_plan=True,
        )

    @contextmanager
    def _joined_side(self):
        """Order two owned branches before and after their independent work."""
        caller = torch.cuda.current_stream(self.device)
        side = self.high_graph.capture_stream if self.high_graph is not None else None
        if side == caller:
            side = None
        if side is not None:
            side.wait_stream(caller)
        try:
            yield side
        finally:
            if side is not None:
                caller.wait_stream(side)

    def _screen(self):
        x, error, seed, words = self._screen_args(self.high_paid)
        record, admitted = self.packed_record, self.high_budget
        self._kernel("high_prepare_fp16_h22_live_k")(
            x, error, record, admitted, words, seed, self.live_nk
        )
        packed_x = list(x)
        packed_x[13] = admitted
        self.high_args = (packed_x, error, seed, words, record, self.live_nk)
        graph = self.high_graph
        with self._joined_side() as side:
            # Each family owns disjoint query groups in the same bitmap.
            with torch.cuda.stream(side):
                if graph is not None and torch.cuda.is_current_stream_capturing():
                    self._kernel("select_attach_if_any")(
                        admitted, 14, graph.raw_cuda_graph()
                    )
                else:
                    self._kernel("high_emit_fp16_candidates_live_k")(*self.high_args)
            self._emit_low(14)
            self._emit_low(16)

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

    def _body(self):
        ops.begin_slab[(triton.cdiv(self.rows, 256),)](
            (self.error, self.pool_head, self.solo_state, self.common_state),
            self.capture[:, 128],
            self.live_nk,
            self.rows,
            self.max_k,
            SEED,
            self.capture.stride(0),
        )
        q = self.q.view(torch.float8_e4m3fn)
        with self._joined_side() as side:
            with torch.cuda.stream(side):
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
            for paid in (14, 16, self.high_paid):
                planner = "direct_abs" if paid in (14, 16) else "planner"
                self._kernel(f"{planner}_plan128")(
                    paid, q, self.w, self.capture, self.plans[paid], self.budget8
                )
        self._metadata()
        self._screen()
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
            *self._row_domain(),
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
                *self._record_state(),
                0,
                self.sms,
                *self._row_domain(),
                self.live_nk,
                SEED,
            )
        self._kernel("select_select_dense5")(
            *self._record_state(),
            *self._row_domain(),
            self.output,
            self.order,
            self.permutation,
            self.partition,
            self.max_k,
        )
        self._recover()
        ops.finish_slab[(triton.cdiv(self.rows, 128),)](
            (self.error, self.fast_error, self.slow_error),
            self.counts,
            self.order,
            (self.counts_original, self.error_original),
            self.rows,
        )

    def prepare(self) -> None:
        if self.graph is None:
            with torch.accelerator.device_index(self.device.index):
                self._prepare_on_device()

    def _prepare_on_device(self) -> None:
        from flashinfer.topk import get_topk_module

        self.topk = get_topk_module()
        self.recovery.prepare_native(self)
        q = self._empty((1, self.heads, 128), torch.float8_e4m3fn).zero_()
        w = self._empty((1, self.heads)).zero_()
        source_k = self.pk[:SEED].clone()
        self.live_nk.fill_(SEED)
        scale_bank = self._empty((SEED + 1,))
        for weights, raw in (
            (w, scale_bank),
            (w.to(torch.bfloat16), scale_bank.view(torch.int32)),
        ):
            raw.fill_(1 if raw.dtype == torch.float32 else 0x7F7F7F7F)
            for scales in (raw[:SEED], raw[1:]):
                self._bootstrap(q, weights, source_k, scales, SEED)
                self._gather(source_k, scales, SEED, SEED)
            self._stage(q, weights, self.cold_zero[:1], self.cold_ends[:1], 0)
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
        # The selector derives both its physical row pitch and vector width
        # from size(1); lengths exclude the short aligned tail of every row.
        pack = 16 // self.cold_scores.element_size()
        pitch = triton.cdiv(common_end, pack) * pack
        scores = self.cold_scores.view(-1)[: 4 * pitch].view(4, pitch)
        # Before the first permutation, PS is available as aligned anchor scratch.
        sf = scales[:common_end]
        if sf.dtype == torch.int32:
            ops.decode_ue8m0[(triton.cdiv(common_end, 256),)](
                sf.view(torch.uint8), self.ps, common_end, 256
            )
            sf = self.ps
        elif sf.data_ptr() % 16:
            self.ps[:common_end].copy_(sf)
            sf = self.ps
        self._dense_score(
            self.cold_q.view(torch.float8_e4m3fn),
            keys[:common_end],
            sf,
            self.cold_w,
            scores,
        )
        self.cold_ends.fill_(common_end)
        self._topk(scores, self.cold_zero, self.cold_ends, self.cold_ids, eager=True)
        votes = self.votes[:common_end]
        votes.zero_()
        ops._add_votes[(triton.cdiv(4 * TOPK, 256),)](
            self.cold_ids, votes, 4 * TOPK, common_end, 256
        )
        self._carry(votes, 4)

    def _gather(
        self, keys, scales, count: int, common_end: int, previous: int = 0
    ) -> None:
        """Publish changed K/scale/norm rows within this call's immutable inputs."""
        if self.plan_epoch + 1 >= 1 << 30:
            self.hot_epoch.zero_()
            self.plan_epoch = 0
        self.plan_epoch += 1
        state = (
            self.hot_epoch,
            self.pair_rank,
            self.permutation,
            self.swap_a,
            self.swap_b,
            self.plan_counts,
        )
        grid = (triton.cdiv(SEED, 256),)
        for build in (ops._mark_hot, ops._compact_pairs):
            build[grid](
                self.hot, state, SEED, common_end, self.plan_epoch, 256, num_warps=8
            )
        ops._publish_gather[(triton.cdiv(count, 32),)](
            keys.view(torch.uint8),
            scales,
            self.pk.view(torch.uint8),
            self.ps,
            self.rho,
            state,
            count,
            previous,
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
        ops._map_votes[(min(triton.cdiv(logical * TOPK, 256), self.sms * 8),)](
            self.output,
            self.permutation,
            self.zero,
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
        # Each 128-row tile records four exact byte counts and stable local ranks.
        tiles = len(self.tier_counts)
        rank = self.partition, self.tier_counts, self.order
        shape = self.rows, self.heads, self.high_paid
        ops.rank_query_tiles[(tiles,)](weights, rank[:2], len(q), *shape, num_warps=4)
        ops.stage_queries[(self.rows,)](
            (q.view(torch.uint8), weights, starts, ends),
            (self.q, self.w, self.starts, self.ends, self.budget8, self.budget4),
            rank,
            len(q),
            key_offset,
            *shape,
            triton.next_power_of_2(tiles),
        )

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
        FP32 scales or I32 packed low-byte UE8M0, held through the final gather.
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
                self._bootstrap(q, weights, keys, scales, common_end)
                if not causal:
                    self._gather(keys, scales, count, common_end)
                previous = 0
                for begin in range(0, rows, self.max_q):
                    end = min(begin + self.max_q, rows)
                    logical = end - begin
                    if causal:
                        visible = common_end + end - 1
                        self._gather(
                            keys, scales, visible, common_end + begin, previous
                        )
                        previous = visible
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
