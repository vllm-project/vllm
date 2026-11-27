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
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    get_mqa_logits_metadata,
    mqa_logits_mx_inputs,
    mqa_logits_reuse_shape,
    mqa_logits_storage,
)

SEED = 12288
TOPK = 2048
RECENT = 1536
RECORD_CAPACITY = 114688
CANDIDATE_SCRATCH_BYTES = 384 << 20
BF16_PACKED_CANDIDATE_SCRATCH_BYTES = 320 << 20
H64_BF16_PACKED_CANDIDATE_SCRATCH_BYTES = 448 << 20
PACKED_ID_PAGE_BYTES = 272
RECORD_PAGE = 512


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


class DoSAWorkspace:
    """Fixed-capacity scratch for one serialized WorkspaceManager execution slot.

    max_queries bounds logical calls; keys support H32/H64, D128 and K<=2**20.
    prepare() captures every branch before KV sizing. Physical
    slabs repeat the last active query in padding, without output or carry votes.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        max_queries: int,
        max_keys: int,
        heads: int,
        logits_dtype: torch.dtype = torch.bfloat16,
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
        self.logits_dtype = logits_dtype
        self.max_queries = max_queries
        self.max_k = triton.cdiv(max_keys, 128) * 128
        packed_bf16 = (
            heads == 32 and logits_dtype == torch.bfloat16 and self.max_k >= 524288
        )
        self.compact_records = packed_bf16
        packed_h64_bf16 = (
            heads == 64 and logits_dtype == torch.bfloat16 and self.max_k >= 524288
        )
        self.strict_seed_ties = packed_bf16 or packed_h64_bf16
        self._tight_candidate_ids = self.strict_seed_ties or (
            heads == 32 and logits_dtype == torch.float32
        )
        self.max_q = _query_slab_rows(max_queries, self.max_k)
        self.rows = self.max_q
        large_packed_pool = logits_dtype == torch.float32 and heads == 64
        if packed_h64_bf16:
            candidate_bytes = H64_BF16_PACKED_CANDIDATE_SCRATCH_BYTES
            candidate_page_bytes = PACKED_ID_PAGE_BYTES
        elif packed_bf16:
            candidate_bytes = BF16_PACKED_CANDIDATE_SCRATCH_BYTES
            candidate_page_bytes = PACKED_ID_PAGE_BYTES
        else:
            candidate_bytes = (
                (768 << 20) if large_packed_pool else CANDIDATE_SCRATCH_BYTES
            )
            candidate_page_bytes = 320 if large_packed_pool else 128 * 4
        self.max_pages = min(
            _candidate_page_budget(self.rows, self.max_k),
            candidate_bytes // candidate_page_bytes,
        )
        self.high_paid = 28 if heads == 64 else 20
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.graph: CUDAGraph | None = None
        self._pack_graph: CUDAGraph | None = None
        self.high_graph: CUDAGraph | None = None
        self.compact_retry_graphs: tuple[CUDAGraph, CUDAGraph] | None = None
        self.dense_probe_graphs: tuple[CUDAGraph, CUDAGraph] | None = None
        self.candidate_pressure = (
            torch.zeros(1, device=self.device, dtype=torch.int32)
            if packed_bf16
            else None
        )
        self.plan_epoch = 0
        self.lock = threading.RLock()
        self._use_hot = False
        self._hot_ops = None
        self._hot_graphs = None
        from vllm.model_executor.layers import dosa_hot

        self._bank_ops = dosa_hot
        self._allocate()
        hot_fp32 = self.fp32 is not None and self.max_k >= 524288
        if (packed_bf16 or hot_fp32) and dosa_hot.allocate_layout(self):
            self._hot_ops = dosa_hot
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
        self.seed_status = self._empty((q,), torch.int32)
        seed_entries = q * TOPK
        bitmap_shape = (q // 4, triton.cdiv(k - SEED, 32), 4)
        bitmap_end = seed_entries + math.prod(bitmap_shape)
        phase_floor = bitmap_end + seed_entries // 4
        # Selection records reuse the bitmap after compaction. For long H32,
        # make already allocated words available to the record page allocator.
        record_bytes = CANDIDATE_SCRATCH_BYTES
        if h == 32 and k >= 524288:
            record_bytes = max(record_bytes, phase_floor * 4)
        page_budget = min(
            q * RECORD_CAPACITY // RECORD_PAGE,
            record_bytes // (5 * RECORD_PAGE),
        )
        record_words = page_budget * RECORD_PAGE
        low_words = 0 if self.compact_records else record_words // 4
        if self.compact_records:
            phase_floor = bitmap_end
        phase_end = max(record_words + low_words, phase_floor)
        direct_boundary = self.logits_dtype == torch.float32
        recovery_size = 0
        if not direct_boundary:
            batch = min(q, 128 if self.strict_seed_ties else 64)
            shape = mqa_logits_reuse_shape(batch, k, h, self.logits_dtype)
            if shape is None:
                batch = min(q, 64)
            else:
                recovery_size = math.prod(shape) // 2
        pool_size = self.max_pages * (PACKED_ID_PAGE_BYTES // 4)
        boundary_entries = q * 4096
        boundary_scratch = boundary_entries * (2 if direct_boundary else 1)
        output_offset = max(recovery_size, boundary_scratch)
        scratch_size = max(pool_size, output_offset + seed_entries)
        joint_scratch = (h == 32 and k >= 524288) or (
            h == 64 and self.logits_dtype == torch.bfloat16 and k < 524288
        )
        if joint_scratch:
            # Long calls use disjoint slices. The short scorer can borrow both
            # after completion, with the internal TopK output kept at the end.
            scratch_start = triton.cdiv(phase_end, 64) * 64
            arena = self._empty((scratch_start + scratch_size,), torch.int32)
            phase_words, scratch = arena[:phase_end], arena[scratch_start:]
            output_offset = scratch_size - seed_entries
            self._short_score_storage = arena[:-seed_entries]
        else:
            phase_words = self._empty((phase_end,), torch.int32)
            scratch = self._empty((scratch_size,), torch.int32)
            self._short_score_storage = phase_words
        self._phase_words = phase_words
        self.ids = phase_words[:record_words].view(page_budget, RECORD_PAGE)
        self.keep_words = phase_words[seed_entries:bitmap_end].view(bitmap_shape)
        low = phase_words[phase_end - low_words :].view(torch.uint8)
        self.values = low.view(-1, RECORD_PAGE)
        self.seed_values = low[-seed_entries:].view(-1, TOPK)
        self.seed_ids = phase_words[:seed_entries].view(q, TOPK)
        self.record_pages = self._empty(
            (q, RECORD_CAPACITY // RECORD_PAGE), torch.int32
        )
        self.record_head = self._empty((1,), torch.int32)
        self.counts = self._empty((q,), torch.int32)
        self.order = self._empty((q,), torch.int64)
        self.partition = self._empty((q,), torch.int32)
        self.error_original = self._empty((q,), torch.int32)
        self.counts_original = self._empty((q,), torch.int32)
        self.zero = torch.zeros(q, device=self.device, dtype=torch.int32)
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
        # The budget assigns each query row to one plan. Narrowing the H18
        # plan moves retained heads forward within that same row.
        fine = (
            self._empty((q, self.high_paid + 2, 128), torch.uint8)
            if h == 32 and k >= 524288
            else None
        )
        for paid in (14, 16, self.high_paid):
            self.plans[paid] = [
                fine[:, : paid + 2]
                if fine is not None
                else self._empty((q, paid + 2, 128), torch.uint8),
                self._empty((q, paid + 2)),
                self._empty((q, 64 if paid <= 16 else 96)),
                self._empty((q,)),
                self._empty((q,)),
                self._empty((q,), torch.int32),
            ]
        # Refine finishes reading candidate IDs before selection writes output.
        self.output = scratch[output_offset : output_offset + seed_entries].view(
            q, TOPK
        )
        # The seed branch joins before compact begins writing candidate IDs.
        if not direct_boundary:
            self.seed_ends = torch.full_like(self.zero, SEED)
            self.seed_logits = mqa_logits_storage(scratch, q, SEED, h, torch.bfloat16)
        self.id_pool = (
            scratch[:pool_size]
            .view(torch.uint8)
            .view(self.max_pages, PACKED_ID_PAGE_BYTES)
        )
        # The candidate IDs are dead by the time the boundary is produced.
        self.select_boundary = scratch[:boundary_entries].view(q, 4096)
        # Both planes reuse the dead candidate arena after direct refinement.
        # Scores must not alias E IDs: BQ1 reads arbitrary IDs while writing them.
        self.select_boundary_scores = (
            scratch[boundary_entries : 2 * boundary_entries]
            .view(torch.float32)
            .view(q, 4096)
            if direct_boundary
            else None
        )
        chunk_capacity = max(1, self.max_pages // 32)
        tail_groups = q // 4 if self._tight_candidate_ids else 0
        self.solo_tasks = self._empty(
            (chunk_capacity + 4 * tail_groups, 3), torch.int32
        )
        self.common_tasks = self._empty((chunk_capacity + tail_groups, 3), torch.int32)
        self.pool_head = torch.zeros(1, device=self.device, dtype=torch.int64)
        self.solo_state = torch.zeros(2, device=self.device, dtype=torch.int32)
        self.common_state = torch.zeros(2, device=self.device, dtype=torch.int32)
        self.schedule = {
            paid: (
                self._empty((q // 8,), torch.int32),
                self._empty((q // 8,), torch.int32),
                self._empty((3,), torch.int32).zero_(),
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
        self.hot_epoch = torch.zeros(k, device=self.device, dtype=torch.int32)
        self.permutation = torch.arange(k, device=self.device, dtype=torch.int32)
        seed_tiles = triton.cdiv(SEED, 256)
        self._seed_scan = (
            self._empty((seed_tiles,), torch.int64),
            self._empty((seed_tiles,), torch.int64),
        )
        self._seed_swaps = (
            self._empty((SEED,), torch.int32),
            self._empty((SEED,), torch.int32),
            self._empty((1,), torch.int32),
        )
        self.hot = self._empty((SEED,), torch.int64)
        anchors = 8 if h == 32 and not direct_boundary else 4
        self.cold_q = self._empty((anchors, h, 128), torch.uint8)
        self.cold_w = self._empty((anchors, h), self.logits_dtype)
        if not direct_boundary:
            # Bootstrap finishes before the slab uses the arena.
            self.cold_logits_storage = scratch
        self.cold_ids = self._empty((anchors, TOPK), torch.int32)
        self.cold_zero = torch.zeros(anchors, device=self.device, dtype=torch.int32)
        self.cold_ends = self._empty((anchors,), torch.int32)
        self.votes = torch.zeros(k, device=self.device, dtype=torch.int32)
        self.carry_partial = self._empty((128, RECENT + 1), torch.int16)
        self.carry_state = self._empty((2 * len(self.carry_partial) + 1,), torch.int32)
        self.topk_state = self._empty((1024 * 1024,), torch.uint8).zero_()
        from vllm.model_executor.layers.dosa_fp32 import (
            NativeFP32Stages,
            NativeRecovery,
        )

        self.fp32 = None
        if direct_boundary:
            self.fp32 = NativeFP32Stages(self, SEED, scratch)
            packed = self.fp32.packed_scales
            recovery_storage = phase_words
            batch = 32
            if h == 32 and k >= 524288:
                shape = mqa_logits_reuse_shape(128, k, h, self.logits_dtype)
                if shape is not None:
                    available = phase_words.numel() // shape[1]
                    if available >= 128:
                        batch = 128
                    elif available >= 64:
                        batch = 64
        else:
            packed = self._empty((k,), torch.int32)
            recovery_storage = scratch[:recovery_size]
        self.recovery = NativeRecovery(self, packed, recovery_storage, batch)

    @contextmanager
    def short_logits(self, rows: int, keys: int):
        # Short scoring borrows storage between complete calls.
        logits = mqa_logits_storage(
            self._short_score_storage, rows, keys, self.heads, self.logits_dtype
        )
        if logits is None:
            yield None
            return
        with self.lock, torch.accelerator.device_index(self.device.index):
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(self.done)
            try:
                yield logits
            finally:
                self.done.record(stream)

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

    def _row_domain(self):
        return self.starts, self.ends, self.error

    def _native_bf16_scores(
        self, q, keys, scales, weights, starts, ends, output, *, schedule=None
    ):
        packed = self.recovery.packed[: len(keys)]
        ops.recovery_pack_scales[(triton.cdiv(len(keys), 256),)](
            scales, packed, len(keys), 256
        )
        unit, weights = mqa_logits_mx_inputs(q, None, weights, torch.bfloat16)
        return fp8_fp4_mqa_logits(
            (q, unit),
            (keys, packed),
            weights,
            starts,
            ends,
            len(keys),
            logits_dtype=torch.bfloat16,
            schedule_meta=schedule,
            output=output,
        )

    def _seed(self, q):
        if self.fp32 is not None:
            logits = self.fp32.score_seed(q)
        else:
            logits = self._native_bf16_scores(
                q,
                self.pk[:SEED],
                self.ps[:SEED],
                self.w,
                self.zero,
                self.seed_ends,
                self.seed_logits,
            )
        self._op(
            "select_prepare5",
            logits,
            self.seed_values,
            self.seed_ids,
            self.counts,
            self.seed_tau,
            self.error,
            self.seed_status,
            self.compact_records,
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
            self.fp32 is not None,
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
        native_factor = 1.0
        unit = 2.0**-24 if self.fp32 is not None else 2.0**-8
        for _ in range(steps):
            native_factor = math.nextafter(native_factor * (1 + unit), math.inf)
        if self.fp32 is not None:
            # Native F32 sums doubled ReLU terms before its final half.
            native_factor *= 2.0
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
            native_factor,
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

    def _attach_two_way(self, choice, graphs, *args):
        self._op(
            "select_attach_two_way",
            choice,
            *(g.raw_cuda_graph() for g in graphs),
            *args,
        )

    def _emit_low(self):
        if not (
            self._use_hot
            and self._hot_low_graphs
            and torch.cuda.is_current_stream_capturing()
        ):
            self._emit_low_stock(14)
            self._emit_low_stock(16)
            return
        self._attach_two_way(self._layout_active, self._hot_low_graphs)
        if self.fp32 is None:
            self._emit_low_stock(16)
            return
        groups = self.rows // 8
        self._hot_ops.choose_h18[(1,)](
            self.budget4,
            self._layout_active,
            self._h18_choice,
            groups,
            self.sms,
            triton.next_power_of_2(groups),
            num_warps=4,
        )
        self._attach_two_way(self._h18_choice, self._hot_h18_graphs)

    def _prepare_low(self, paid, length, config=None, offset=None):
        ids, inv, primary = self.schedule[paid]
        config = primary if config is None else config
        groups = self.rows // 8
        grid = (2 if paid == 14 else 1) * max(groups, 8 * self.sms)
        ops.prepare_live[(1,)](
            self.budget4,
            length,
            ids,
            inv,
            config,
            groups,
            grid,
            self.sms,
            self.max_k,
            SEED,
            256 if paid == 14 else 128,
            paid,
            triton.next_power_of_2(groups),
            WHOLE_WAVES=True,
            OFFSET=offset,
            num_warps=4,
        )
        return ids, inv, config, grid

    def _emit_low_stock(self, paid):
        self._op(
            "low_emit_indexed",
            *self._screen_args(paid),
            *self._prepare_low(paid, self.live_nk),
            self.live_nk,
            self.sms,
            paid,
            False,
            False,
        )

    def _emit_hot_low(self, paid):
        x, error, seed, words = self._screen_args(paid)
        self._hot_ops.prepare_cutoffs(self, paid, self._cutoff, self._second_cutoff)
        hot = x.copy()
        second_offset = cold_offset = None
        second_cfg, cold_cfg = self._second_config, self._cold_config
        if paid == 16:
            second_offset, cold_offset = second_cfg[2:], cold_cfg[2:]
            second_cfg, cold_cfg = self._second_h18_config, self._cold_h18_config

        for cutoff, end, length, config, offset in (
            (self._cutoff, self._hot_end, self._hot_end, None, None),
            (
                self._second_cutoff,
                self._second_end,
                self._second_live,
                second_cfg,
                second_offset,
            ),
        ):
            hot[14] = cutoff
            self._op(
                "low_emit_indexed",
                hot,
                error,
                seed,
                words,
                *self._prepare_low(paid, length, config, offset),
                end,
                self.sms,
                paid,
                True,
                True,
            )
        self._op(
            "low_emit_indexed",
            x,
            error,
            seed,
            words,
            *self._prepare_low(paid, self._cold_live, cold_cfg, cold_offset),
            self.live_nk,
            self.sms,
            paid,
            True,
            False,
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

    def _probe_hot_layout(self):
        if self.heads == 64:
            groups = self.rows // 4
            self._hot_ops.choose_high_layout[(1,)](
                self.budget4,
                self._common_end,
                self._layout_active,
                groups,
                self.sms,
                triton.next_power_of_2(groups),
                num_warps=4,
            )
            return
        x, error, seed, words = self._screen_args(14)
        self._sampled_ids.zero_()
        self._op(
            "low_estimate_f1_pressure",
            x,
            error,
            seed,
            words,
            self.live_nk,
            self._sampled_ids,
        )
        self._hot_ops.decide_layout[(1,)](
            self._sampled_ids,
            self.live_nk,
            self._common_end,
            self._layout_active,
            self.max_pages,
            SEED,
            num_warps=1,
        )

    def _choose_hot_layout(self):
        if self._hot_probe_graph is None:
            self._probe_hot_layout()
            self._hot_ops.partition(self)
        elif torch.cuda.is_current_stream_capturing():
            self._attach_if_any(self._first_slab, self._hot_probe_graph)
            self._hot_ops.first_reorder[(1,)](
                self._layout_active,
                self._first_slab,
                self._first_reorder,
                num_warps=1,
            )
            self._attach_if_any(self._first_reorder, self._hot_partition_graph)

    def _emit_hot_high(self, x, error, seed, words, record, live):
        self._hot_ops.prepare_cutoffs(
            self,
            self.high_paid,
            self._high_cutoff,
            self._high_cutoff_second,
            active_budget=self.high_budget,
        )
        args = x, error, seed, words, record
        for end, offset, cutoff in (
            (self._hot_end, self.zero[:1], self._high_cutoff),
            (self._second_end, self._second_config[2:], self._high_cutoff_second),
        ):
            self._op(
                "high_emit_fp16_uniform_candidates_live_k", *args, end, offset, cutoff
            )
        self._op(
            "high_emit_fp16_cold_candidates_live_k", *args, live, self._cold_config[2:]
        )

    def _screen(self):
        if self._use_hot:
            self._choose_hot_layout()
        probe = self.dense_probe_graphs
        if probe is not None:
            if torch.cuda.is_current_stream_capturing():
                self._attach_if_any(self.candidate_pressure, probe[0])
            else:
                probe[1].replay()
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
                    if self._use_hot:
                        self._attach_two_way(
                            self._layout_active,
                            (self._hot_stock_high, graph),
                            admitted,
                            14,
                        )
                    else:
                        self._attach_if_any(admitted, graph, 14)
                elif self._use_hot:
                    self._emit_hot_high(*self.high_args)
                else:
                    self._kernel("high_emit_fp16_candidates_live_k")(*self.high_args)
            self._emit_low()

    def _pack_recovery_scales(self) -> None:
        ops.recovery_pack_scales[(triton.cdiv(self.max_k, 256),)](
            self.ps, self.recovery.packed, self.max_k, 256
        )

    def _recover(self):
        if self.fp32 is None:
            if self._pack_graph is None:
                self._pack_recovery_scales()
            else:
                self._attach_if_any(self.error, self._pack_graph)
        self.recovery.run()

    def _body(self):
        self.record_pages.fill_(-1)
        self.record_head.fill_(self.rows * (TOPK // RECORD_PAGE + 4))
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
                    paid,
                    q,
                    self.w,
                    self.capture,
                    self.plans[paid],
                    self.budget8,
                    self.fp32 is not None,
                )
        self._metadata()
        if self.heads == 32 and self.logits_dtype == torch.bfloat16:
            self._kernel("narrow_prepack_h18_to_h16")(
                self.plans[16], self.plans[14], self.budget4, self.error
            )
        self._screen()
        retry = self.compact_retry_graphs
        retry_address = 0
        if retry is not None and torch.cuda.is_current_stream_capturing():
            retry_address = retry[0].raw_cuda_graph()
        self._compact_candidates(False, retry is not None, retry_address)
        if retry is not None and retry_address == 0:
            retry[1].replay()
        self._kernel("refine_validate_seed5_live_nk")(
            self.seed_ids,
            self.counts,
            self.seed_tau,
            self.safe_tau,
            *self._row_domain(),
            self.max_k,
            self.live_nk,
            SEED,
            self.compact_records,
        )
        if self.fp32 is not None:
            self.fp32.refine()
        else:
            self._refine_bf16(q)
        self._select_records()
        if self.fp32 is not None:
            self.fp32.boundary()
        self._recover()
        ops.finish_slab[(triton.cdiv(self.rows, 128),)](
            self.error,
            self.counts,
            self.order,
            (self.counts_original, self.error_original),
            self.rows,
        )

    def _compact_candidates(self, tight, *options) -> None:
        self._op(
            "compact_packed20_spans",
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
            tight,
            *options,
        )

    def _refine_bf16(self, q):
        for common, tasks, state in (
            (False, self.solo_tasks, self.solo_state),
            (True, self.common_tasks, self.common_state),
        ):
            self._op(
                "refine_exact_span_packed20_live_nk",
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
                self.sms,
                *self._row_domain(),
                self.live_nk,
                SEED,
                self.record_pages,
                self.record_head,
                self.compact_records,
                self.strict_seed_ties,
                common,
            )

    def _select_records(self):
        self._kernel("select_select_dense5")(
            self.values,
            self.ids,
            self.counts,
            *self._row_domain(),
            self.output,
            self.order,
            self.permutation,
            self.partition,
            self.max_k,
            self.fp32 is None,
            self.record_pages,
            self.select_boundary,
            self.compact_records,
        )

    def _capture_operation(self, op, *args, keep_graph=True):
        with capture(self.device, keep_graph=keep_graph) as graph:
            op(*args)
        return graph

    def _attach_if_any(self, status, child, threshold=0):
        self._op("select_attach_if_any", status, threshold, child.raw_cuda_graph())

    def _capture_conditional(self, status, op, *args):
        child = self._capture_operation(op, *args)
        dispatch = self._capture_operation(
            self._attach_if_any, status, child, keep_graph=False
        )
        return child, dispatch

    def prepare(self) -> None:
        if self.graph is None:
            with torch.accelerator.device_index(self.device.index):
                self._prepare_on_device()

    def _prepare_on_device(self) -> None:
        from flashinfer.topk import get_topk_module

        self.topk = get_topk_module()
        q = self._empty((1, self.heads, 128), torch.float8_e4m3fn).zero_()
        w = self._empty((1, self.heads)).zero_()
        source_k = self.pk[:SEED].clone()
        self.live_nk.fill_(SEED)
        scale_bank = self._empty((SEED + 1,))
        packed = scale_bank.view(torch.int32)
        fixtures = (
            ((w, packed),)
            if self.fp32 is not None
            else ((w, scale_bank), (w.to(torch.bfloat16), packed))
        )
        for weights, raw in fixtures:
            raw.fill_(1 if raw.dtype == torch.float32 else 0x7F7F7F7F)
            for scales in (raw[:SEED], raw[1:]):
                self._bootstrap(q, weights, source_k, scales, SEED)
                self._gather(source_k, scales, SEED, SEED)
            self._stage(q, weights, self.cold_zero[:1], self.cold_ends[:1], 0)
        self._body()
        self.recovery.leaf = self._capture_operation(self.recovery.body)
        if self.fp32 is None:
            self._pack_graph = self._capture_operation(self._pack_recovery_scales)
        if self._tight_candidate_ids:
            self._capture_compaction_retry()
        self.high_graph = self._capture_operation(
            self._op, "high_emit_fp16_candidates_live_k", *self.high_args
        )
        self.graph = self._capture_operation(self._body, keep_graph=False)
        self.graph.replay()
        recent = min(self.max_q, RECENT)
        self._map(self.rows, SEED, self.votes[:SEED], recent)
        self._carry(self.votes[:SEED], recent)
        if len(self.cold_q) == 8:
            self._carry(self.votes[:SEED], 8)
        if self._hot_ops is not None:
            self._prepare_hot_graphs()
        self.done.record(torch.cuda.current_stream(self.device))

    def _prepare_hot_graphs(self) -> None:
        hot = self._hot_ops
        stock = self.graph, self.high_graph
        self._use_hot = True
        self._hot_probe_graph = self._hot_low_graphs = None
        self.high_graph = None
        self._emit_hot_high(*self.high_args)
        low_roles = (14, 16) if self.fp32 is not None else (14,)
        if self.heads == 32:
            for paid in low_roles:
                self._emit_hot_low(paid)
        hot.repair_bands(self)
        self._hot_probe_graph = self._capture_operation(self._probe_hot_layout)
        self._hot_partition_graph = self._capture_operation(hot.partition, self)
        if self.heads == 32:
            for paid, attr in zip(low_roles, ("_hot_low_graphs", "_hot_h18_graphs")):
                setattr(
                    self,
                    attr,
                    tuple(
                        self._capture_operation(emit, paid)
                        for emit in (self._emit_low_stock, self._emit_hot_low)
                    ),
                )
        high = self._capture_operation(self._emit_hot_high, *self.high_args)
        self._hot_stock_high = stock[1]
        self.high_graph = high
        self._hot_graphs = self._capture_operation(self._body, keep_graph=False), high
        self._use_hot = False
        self.graph, self.high_graph = stock
        self._hot_repair_graph, self._hot_repair_dispatch = self._capture_conditional(
            self._layout_active, hot.repair_bands, self
        )

    def _capture_compaction_retry(self) -> None:
        pressure = self.candidate_pressure

        def retry():
            if pressure is not None:
                pressure.fill_(1)
            elif self._hot_ops is not None:
                self._layout_active.zero_()
            self.pool_head.zero_()
            self.solo_state.zero_()
            self.common_state.zero_()
            self._compact_candidates(True)

        self.compact_retry_graphs = self._capture_conditional(
            self.solo_state[1:], retry
        )
        if pressure is not None:
            self.dense_probe_graphs = self._capture_conditional(
                pressure,
                self._op,
                "low_probe_native_dense",
                *self._screen_args(14),
                self.live_nk,
                self.budget4,
            )

    def _bootstrap(
        self, q, weights, keys, scales, common_end: int, causal: bool = False
    ) -> None:
        rows = len(q)
        if causal and self.heads == 32:
            # Later causal slabs use carry votes.
            rows = min(rows, self.max_q)
        anchors = len(self.cold_q) if causal and len(keys) >= 524288 else 4
        ops._spread_queries[(anchors,)](
            q.view(torch.uint8),
            weights,
            self.cold_q,
            self.cold_w,
            rows,
            self.heads,
            num_warps=4,
        )
        self.cold_ends.fill_(common_end)
        if self.fp32 is not None:
            packed = self.fp32.packed_scales[:common_end]
            packed.copy_(scales[:common_end])
            scores = fp8_fp4_mqa_logits(
                (self.cold_q.view(torch.float8_e4m3fn), self.fp32.unit_q[:4]),
                (keys[:common_end], packed),
                self.cold_w,
                self.cold_zero,
                self.cold_ends,
                common_end,
                logits_dtype=torch.float32,
            )
            stock_ops.top_k_per_row_prefill(
                scores,
                self.cold_zero,
                self.cold_ends,
                self.cold_ids,
                4,
                scores.stride(0),
                scores.stride(1),
                TOPK,
            )
        else:
            self._bootstrap_bf16(keys, scales, common_end, anchors)
        votes = self.votes[:common_end]
        votes.zero_()
        ops._add_votes[(triton.cdiv(anchors * TOPK, 256),)](
            self.cold_ids, votes, anchors * TOPK, common_end, 256
        )
        self._carry(votes, anchors)

    def _bootstrap_bf16(self, keys, scales, common_end, anchors):
        storage = mqa_logits_storage(
            self.cold_logits_storage, anchors, common_end, self.heads, torch.bfloat16
        )
        starts, ends = self.cold_zero[:anchors], self.cold_ends[:anchors]
        schedule = get_mqa_logits_metadata(starts, ends, common_end, self.heads)
        scores = self._native_bf16_scores(
            self.cold_q[:anchors].view(torch.float8_e4m3fn),
            keys[:common_end],
            scales[:common_end],
            self.cold_w[:anchors],
            starts,
            ends,
            storage,
            schedule=schedule,
        )
        # FlashInfer derives its pitch from the physical column extent.
        pitch = scores.stride(0)
        scores = scores.as_strided((anchors, pitch), (pitch, 1))
        self._topk(
            scores,
            self.cold_zero[:anchors],
            self.cold_ends[:anchors],
            self.cold_ids[:anchors],
            eager=True,
        )

    def _gather(
        self, keys, scales, count: int, common_end: int, previous: int = 0
    ) -> None:
        """Publish changed K/scale/norm rows within this call's immutable inputs."""
        hot = self._hot_ops if self._use_hot else None
        layout = self._layout_active if hot is not None and previous else None
        if layout is not None:
            if self.candidate_pressure is not None:
                hot.disable_on_pressure[(1,)](
                    layout, self.candidate_pressure, num_warps=1
                )
            elif self.heads == 64:
                hot.disable_on_pool_overflow[(1,)](
                    layout, self.pool_head, self.max_pages, num_warps=1
                )
        if self.plan_epoch + 1 >= 1 << 30:
            self.hot_epoch.zero_()
            self.plan_epoch = 0
        self.plan_epoch += 1
        self._bank_ops.append_and_seed(self, keys, scales, count, previous)
        self.live_nk.fill_(count)
        if hot is not None:
            hot.reset_ranges[(1,)](
                self._common_end,
                self._first_slab,
                self._layout_active,
                self._hot_end,
                self._second_end,
                self._second_live,
                self._cold_live,
                self._second_config,
                self._cold_config,
                common_end,
                count,
                previous == 0,
                SEED,
                num_warps=1,
            )
            if layout is not None:
                self._hot_repair_dispatch.replay()

    def _carry(self, votes, max_vote):
        """Select HOT by exact vote cutoff, small-ID ties and ascending IDs."""
        count, stride = votes.numel(), self.carry_partial.stride(0)
        block_pad = len(self.carry_partial)
        blocks = triton.cdiv(count, 8192)
        ops._carry_histogram[(blocks,)](
            votes,
            self.carry_partial,
            count,
            max_vote,
            stride,
            triton.next_power_of_2(max(8, max_vote + 1)),
            num_warps=8,
        )
        ops._carry_cutoff[(1,)](
            self.carry_partial,
            self.carry_state,
            min(SEED, count),
            max_vote,
            blocks,
            stride,
            block_pad,
            num_warps=8,
        )
        ops._carry_emit[(blocks,)](
            votes, self.hot, self.carry_state, count, max_vote, block_pad, num_warps=8
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
        """Write native MX top2048 request-local K IDs.

        Q/K are contiguous E4M3 [Q,H,128]/[K,128]. Scales are contiguous
        UE8M0 powers of two represented as F32 or packed I32.
        The BF16 mode accepts native already-rounded weights. FP32 takes F32
        weights and the original packed MX scales. Contiguous I32[Q]
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
        scale_types = (
            (torch.int32,) if self.fp32 is not None else (torch.float32, torch.int32)
        )
        weight_types = (
            (torch.float32,)
            if self.fp32 is not None
            else (torch.float32, torch.bfloat16)
        )
        for value, shape, dtypes in (
            (q, (rows, self.heads, 128), (torch.float8_e4m3fn,)),
            (keys, (count, 128), (torch.float8_e4m3fn,)),
            (scales, (count,), scale_types),
            (weights, (rows, self.heads), weight_types),
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
                self._use_hot = (
                    self._hot_graphs is not None
                    and causal
                    and key_offset == 0
                    and common_end >= 524288
                )
                graph = self._hot_graphs[0] if self._use_hot else self.graph
                if self.candidate_pressure is not None:
                    self.candidate_pressure.zero_()
                self._bootstrap(q, weights, keys, scales, common_end, causal=causal)
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
                    graph.replay()
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
                self._use_hot = False
                self.done.record(stream)
