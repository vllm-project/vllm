# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact DoSA selection over one request's contiguous FP8 index keys.

The owner retains scratch and private CUDA graphs. Calls borrow Q/K, weights,
scales, intervals and output through completion on the calling stream. Bootstrap
and carry are local to one call; sharing this owner never shares request hints.
"""

import math
import threading

import torch
from torch.cuda import CUDAGraph

from vllm import _custom_ops as stock_ops
from vllm.model_executor.layers import dosa_caller as caller
from vllm.model_executor.layers import dosa_ops as ops
from vllm.model_executor.layers.dosa_ops import as_fp32_logits
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, get_mqa_logits_metadata

SEED = 12288
TOPK = 2048
RECENT = 1536
RECORD_CAPACITY = 114688
MIN_KEYS = 131072


def _id_pool_pages(rows: int, keys: int) -> int:
    # Common plus three solo streams can add four partial pages per Q4.
    causal = (rows // 4) * ((keys - SEED + 2 * (rows - 1) + 127) // 128 + 4)
    return max(256 * rows, causal)


def _slab_storage(rows: int, keys: int) -> tuple[int, int]:
    rows = min(triton.cdiv(rows, 8) * 8, 16384)
    page_bytes = (128 + 2 + 2) * 4
    budget = rows * (65536 * 5 + 256 * page_bytes)
    while (
        rows > 8
        and rows * RECORD_CAPACITY * 5 + _id_pool_pages(rows, keys) * page_bytes
        > budget
    ):
        rows = max(8, rows // 16 * 8)
    return rows, _id_pool_pages(rows, keys)


class FastRecovery:
    """Native-BF16 raw-dot recovery, with its domain certified by safe_tau."""

    def __init__(self, owner: "DoSAWorkspace") -> None:
        rows, keys, heads = min(owner.max_q, 512), owner.max_k, owner.heads
        self.chunks = triton.cdiv(keys, 32768)
        self.rows = owner._empty((rows,), torch.int32)
        self.state = owner._empty((2,), torch.int32)
        self.packed_q = owner._empty((rows, heads, 128), torch.float8_e4m3fn)
        self.packed_w = owner._empty((rows, heads), torch.float32)
        self.finite = owner._empty((8,), torch.int32)
        self.lengths = owner._empty((8 * self.chunks,), torch.int32)
        self.merge_lengths = owner._empty((8,), torch.int32)
        self.fallback_lengths = owner._empty((rows,), torch.int32)
        self.leaf = owner._empty((8 * self.chunks, 32768), torch.float32)
        self.merge = owner._empty((8, self.chunks * TOPK), torch.float32)
        self.ids = owner._empty((8, self.chunks * TOPK), torch.int32)
        self.leaf_selected = owner._empty((8 * self.chunks, TOPK), torch.int32)
        self.merge_selected = owner._empty((8, TOPK), torch.int32)
        self.full_selected = owner._empty((rows, TOPK), torch.int32)
        self.tasks = owner._empty((rows * triton.cdiv(keys, 4096), 2), torch.int32)
        self.replay_state = owner._empty((20,), torch.int32)
        self.row_slot = owner._empty((rows,), torch.int32)
        self.leaf_zero = torch.zeros_like(self.lengths)
        self.merge_zero = torch.zeros_like(self.merge_lengths)
        self.full_zero = torch.zeros(rows, device=owner.device, dtype=torch.int32)

    def __call__(self, owner: "DoSAWorkspace", begin: int, end: int) -> None:
        n = end - begin
        dense, counts = owner.replay_scores[:n], owner.replay_counts[:n]
        status = owner.fast_error[begin:end]
        owner._op(
            "recovery_replay_and_leaf_capacity",
            owner.q[begin:end].view(torch.float8_e4m3fn),
            owner.pk,
            owner.ps,
            owner.w[begin:end],
            owner.starts[begin:end],
            owner.ends[begin:end],
            status,
            dense,
            counts,
            self.leaf,
            self.rows,
            self.state,
            self.finite,
            self.lengths,
            owner.live_nk,
            self.packed_q,
            self.packed_w,
            self.tasks,
            self.replay_state,
            self.row_slot,
        )
        owner._op(
            "sample_recover_score",
            self.packed_q,
            owner.pk,
            owner.ps,
            self.packed_w,
            dense,
            self.state,
            self.rows,
            owner.starts[begin:end],
            owner.ends[begin:end],
            owner.live_nk,
            owner.sms,
        )
        selected = owner._topk(
            self.leaf, self.leaf_zero, self.lengths, self.leaf_selected
        )
        owner._op(
            "recovery_gather",
            self.leaf,
            selected,
            self.lengths,
            self.rows,
            self.state,
            self.finite,
            self.merge,
            self.ids,
        )
        owner._op(
            "recovery_final_meta",
            counts,
            self.rows,
            self.state,
            self.finite,
            self.merge_lengths,
            self.fallback_lengths[:n],
            self.chunks,
        )
        selected = owner._topk(
            self.merge, self.merge_zero, self.merge_lengths, self.merge_selected
        )
        output = owner._topk(
            dense, self.full_zero[:n], self.fallback_lengths[:n], self.full_selected[:n]
        )
        owner._op(
            "recovery_publish",
            self.ids,
            selected,
            self.rows,
            self.state,
            self.finite,
            output,
        )
        owner._op(
            "recovery_recovered", output, owner.order[begin:end], status, owner.output
        )


class NativeRecovery:
    """One serialized batch capacity, allocated before KV-cache sizing.

    Q/K are contiguous E4M3 [rows,H,128]/[Kcap,128]. Scales are exact F32 decodes
    of repeated UE8M0, including exceptional exponents/NaN. K/scale storage is
    initialized; retained tail values are excluded by live intervals. W contains
    native BF16-rounded values in F32/BF16 [rows,H]. I32 intervals refer to the
    supplied K; every valid failed row has at least2048 keys. Nonzero error entries
    choose recovery, and order maps these sorted rows into the output slab.
    The required I32[1] logical_nk bounds intervals within the physical Kcap.
    Each call takes1..capacity rows; only shape metadata determines n.

    Returned IDs remain in the supplied (permuted) K domain; the owner's normal
    later permutation publication maps them to logical IDs. Healthy output rows
    stay unchanged. Invalid intervals become empty and retain/set error bit2,
    matching the original recovery guard. The parent owns stream serialization,
    input lifetime and graph lifetime; this helper creates no graph or registry.
    Native metadata partitions independent K-token tiles across SMs; each key's
    D128 dot and BF16 epilogue retain the native scorer's arithmetic.
    """

    def __init__(self, device, rows, heads, max_keys):
        self.capacity, self.max_keys = rows, max_keys
        self.unit_q = torch.full(
            (rows, heads), 0x7F7F7F7F, device=device, dtype=torch.int32
        )
        self.weights = torch.empty((rows, heads), device=device, dtype=torch.bfloat16)
        self.starts = torch.empty(rows, device=device, dtype=torch.int32)
        self.ends = torch.empty_like(self.starts)
        self.lengths = torch.empty_like(self.starts)
        self.zero = torch.zeros_like(self.starts)
        self.selected = torch.empty((rows, TOPK), device=device, dtype=torch.int32)
        self.packed = torch.empty(max_keys, device=device, dtype=torch.int32)

    def pack_scales(self, scales):
        """Publish K scale bytes once before the serialized512-row traversal."""
        ops.recovery_pack_scales[(triton.cdiv(self.max_keys, 256),)](
            scales, self.packed, self.max_keys, 256
        )

    def __call__(
        self, q, keys, weights, starts, ends, error, order, output, logical_nk
    ):
        n = len(q)
        self.weights[:n].copy_(weights)
        ops.recovery_intervals[(triton.cdiv(n, 128),)](
            starts,
            ends,
            error,
            self.starts,
            self.ends,
            self.lengths,
            logical_nk,
            n,
            self.max_keys,
        )
        metadata = get_mqa_logits_metadata(
            self.starts[:n], self.ends[:n], len(keys), q.shape[1]
        )
        logits = fp8_fp4_mqa_logits(
            (q, self.unit_q[:n]),
            (keys, self.packed),
            self.weights[:n],
            self.starts[:n],
            self.ends[:n],
            self.max_keys,
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
        ops.recovery_publish[(n, TOPK // 256)](
            self.selected[:n],
            error,
            order,
            self.starts[:n],
            output,
            TOPK,
            output.stride(0),
        )


class DoSAWorkspace:
    """Serial, fixed-capacity DoSA scratch for a WorkspaceManager execution slot.

    Args:
        device: SM100 CUDA device.
        max_queries: Logical call capacity; larger calls use independent slabs.
        max_keys: Maximum logical K length, at most 2**20.
        heads: 32 or 64 score heads; each head has 128 FP8 values.

    ``prepare()`` allocates and captures all execution branches before KV-cache
    sizing. Each slab uses the same physical Q capacity; inactive Q lanes repeat
    the last active query and never publish output or carry votes.

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
        self.max_q, self.max_pages = _slab_storage(max_queries, self.max_k)
        self.rows = self.max_q
        self.high_paid = 28 if heads == 64 else 20
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.graph: CUDAGraph | None = None
        self.recovery_graphs: tuple[CUDAGraph, CUDAGraph] | None = None
        self.native_recovery_graphs: tuple[CUDAGraph, CUDAGraph] | None = None
        self.high_graphs: tuple[CUDAGraph, ...] | None = None
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
        self.sample_scores = self._empty((q, 128), torch.float32)
        self.sample_masks = self._empty(
            (q, 128), torch.int64 if h == 64 else torch.int32
        )
        self.safe_tau = self._empty((q,), torch.float32)
        self.seed_tau = self._empty((q,), torch.float32)
        self.error = self._empty((q,), torch.int32)
        self.fast_error = self._empty((q,), torch.int32)
        self.slow_error = self._empty((q,), torch.int32)
        self.lengths = self._empty((q,), torch.int32)
        self.seed_diag = self._empty((q, 4), torch.int32)
        self.values = self._empty((q, RECORD_CAPACITY), torch.uint8)
        self.ids = self._empty((q, RECORD_CAPACITY), torch.int32)
        self.counts = self._empty((q,), torch.int32)
        self.solo_counts = self._empty((q,), torch.int32)
        self.common_counts = self._empty((q // 4,), torch.int32)
        self.order = self._empty((q,), torch.int64)
        self.ordered_output = self._empty((q, TOPK), torch.int32)
        self.output = self._empty((q, TOPK), torch.int32)
        self.error_original = self._empty((q,), torch.int32)
        self.counts_original = self._empty((q,), torch.int32)
        self.select_diag = self._empty((q, 2), torch.int32)
        self.boundary_ids = self._empty((q, 4096), torch.int32)
        self.boundary_counts = self._empty((q,), torch.int32)
        self.zero = torch.zeros(q, device=self.device, dtype=torch.int32)
        self.replay_scores = self._empty((min(q, 512), k))
        self.replay_counts = self._empty((min(q, 512),), torch.int32)
        self.fast_recovery = FastRecovery(self)
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
        self.sample_scale = self._empty((128,))
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
        self.page_table = self._empty((q + q // 4, triton.cdiv(k, 128)), torch.int32)
        self.id_pool = self._empty((self.max_pages, 128), torch.int32)
        self.solo_tasks = self._empty((self.max_pages, 2), torch.int32)
        self.common_tasks = self._empty((self.max_pages, 2), torch.int32)
        self.pool_head = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.pool_status = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.solo_n = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.common_n = torch.zeros(1, device=self.device, dtype=torch.int32)
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
        self.fast_budget = self._empty((q // 4,), torch.int32)
        self.slow_budget = self._empty((q // 4,), torch.int32)
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
        self.plan_counts = torch.zeros(4, device=self.device, dtype=torch.int32)
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
        self.refine_tasks = self._empty((self.max_pages, 2), torch.int32)
        self.refine_state = self._empty((3,), torch.int32)
        self.topk_state = torch.zeros(
            1024 * 1024, device=self.device, dtype=torch.uint8
        )

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
            "seed_prepare5",
            self.seed_logits,
            self.values,
            self.ids,
            self.counts,
            self.seed_tau,
            self.error,
            self.lengths,
            self.seed_diag,
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
            self.seed_diag,
            self.seed_tau,
            self.error,
            self.safe_tau,
            8,
            dot,
            g,
            bf16_factor,
            True,
            self.heads,
            self.high_paid,
            num_warps=1,
            enable_fp_fusion=False,
        )

    def _emit(self, paid, x, streams):
        args = (
            paid,
            x,
            streams,
            self.id_pool,
            self.page_table,
            self.pool_head,
            self.pool_status,
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
        record, fast, slow = (
            self.packed_record[: self.rows // 4],
            self.fast_budget[: self.rows // 4],
            self.slow_budget[: self.rows // 4],
        )
        self._kernel("high_prepare_fp16_h22_live_k")(
            x, self.error, record, fast, slow, SEED, self.live_nk
        )
        slow_x, fast_x = list(x), list(x)
        slow_x[13], fast_x[13] = slow, fast
        common = (
            streams,
            self.id_pool,
            self.page_table,
            self.pool_head,
            self.pool_status,
            self.error,
            SEED,
            self.keep_words,
        )
        self.high_args = (
            (paid, slow_x, *common, self.live_nk),
            (paid, fast_x, *common, record, self.live_nk),
        )
        if self.high_graphs is not None and torch.cuda.is_current_stream_capturing():
            self._kernel("post_attach_if_any_not_in")(
                slow, 14, 16, self.high_graphs[0].raw_cuda_graph()
            )
            self._kernel("post_attach_if_any_not_in")(
                fast, 14, 14, self.high_graphs[1].raw_cuda_graph()
            )
        else:
            self._kernel("high_emit_trusted_candidates_live_k")(*self.high_args[0])
            self._kernel("high_emit_fp16_candidates_live_k")(*self.high_args[1])

    def _recover_native(self):
        self.recovery.pack_scales(self.ps)
        for begin in range(0, self.rows, 512):
            end = min(self.rows, begin + 512)
            self.recovery(
                self.q[begin:end].view(torch.float8_e4m3fn),
                self.pk,
                self.w[begin:end],
                self.starts[begin:end],
                self.ends[begin:end],
                self.slow_error[begin:end],
                self.order[begin:end],
                self.output,
                self.live_nk,
            )

    def _recover_fast(self):
        for begin in range(0, self.rows, 512):
            self.fast_recovery(self, begin, min(self.rows, begin + 512))

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
        fast_graphs = self.recovery_graphs
        if fast_graphs is None:
            self._recover_fast()
        elif torch.cuda.is_current_stream_capturing():
            self._kernel("post_attach_if_any")(
                self.fast_error, False, fast_graphs[0].raw_cuda_graph()
            )
        else:
            fast_graphs[1].replay()
        graphs = self.native_recovery_graphs
        if graphs is None:
            self._recover_native()
        elif torch.cuda.is_current_stream_capturing():
            self._kernel("post_attach_if_any")(
                self.slow_error, False, graphs[0].raw_cuda_graph()
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
        for value in (
            self.pool_head,
            self.pool_status,
            self.solo_counts,
            self.common_counts,
            self.solo_n,
            self.common_n,
            self.error,
        ):
            value.zero_()
        ops.clear_live_table[(self.rows + self.rows // 4,)](
            self.page_table,
            self.error,
            self.pool_status,
            self.live_nk,
            self.rows,
            self.page_table.stride(0),
            self.max_k,
            SEED,
            256,
            num_warps=4,
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
        torch.index_select(self.ps, 0, self.sample_ids, out=self.sample_scale)
        self._kernel("sample_seed_score")(
            q,
            self.sample_k.view(torch.float8_e4m3fn),
            self.sample_scale,
            self.w,
            self.sample_scores,
            self.sample_masks,
            self.sms,
        )
        self.capture[:, :128].copy_(self.sample_masks)
        self.capture[:, 128].copy_(self.error)
        for paid in (14, 16, self.high_paid):
            planner = "direct_abs" if paid in (14, 16) else "planner"
            self._kernel(f"{planner}_plan128")(
                paid, q, self.w, self.capture, self.plans[paid], self.budget8
            )
        self._metadata()
        streams = [
            self.solo_counts,
            self.solo_tasks,
            self.solo_n,
            self.common_counts,
            self.common_tasks,
            self.common_n,
        ]
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
            self._emit(paid, x, streams)
        self._kernel("compact_compact_candidates_live_nk")(
            streams,
            self.id_pool,
            self.page_table,
            self.pool_head,
            self.pool_status,
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
        for method, counts, tasks, task_count, table in (
            (
                self._kernel("refine_exact_paged_solo_live_nk"),
                self.solo_counts,
                self.solo_tasks,
                self.solo_n,
                self.page_table[: self.rows],
            ),
            (
                self._kernel("refine_exact_paged_common_live_nk"),
                self.common_counts,
                self.common_tasks,
                self.common_n,
                self.page_table[self.rows :],
            ),
        ):
            method(
                q,
                self.pk,
                self.ps,
                self.w,
                self.safe_tau,
                self.id_pool,
                counts,
                tasks,
                task_count,
                self.values,
                self.ids,
                self.counts,
                0,
                self.sms,
                table,
                self.max_pages,
                self.starts,
                self.ends,
                self.error,
                self.live_nk,
                SEED,
                self.refine_tasks,
                self.refine_state,
            )
        self._kernel("select_select_dense5")(
            self.values,
            self.ids,
            self.counts,
            self.starts,
            self.ends,
            self.pool_status,
            self.error,
            self.ordered_output,
            self.select_diag,
            self.boundary_ids,
            self.boundary_counts,
            self.max_k,
        )
        self._kernel("post_complete_ties")(
            self.boundary_ids,
            self.boundary_counts,
            self.select_diag,
            self.starts,
            self.ends,
            self.error,
            self.ordered_output,
            self.permutation,
        )
        self.output.index_copy_(0, self.order, self.ordered_output)
        self._recover()
        self.counts_original.index_copy_(0, self.order, self.counts)
        self.error_original.index_copy_(0, self.order, self.error)

    def prepare(self) -> None:
        """Capture every branch with valid initialized data during memory profiling."""
        from flashinfer.topk import get_topk_module

        from vllm.model_executor.layers.dosa_graph import capture

        if self.graph is not None:
            return
        self.topk = get_topk_module()
        self.recovery = NativeRecovery(
            self.device,
            min(self.rows, 512),
            self.heads,
            self.max_k,
        )
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
        with capture(self.device) as native_recovery_graph:
            self._recover_native()
        with capture(self.device, keep_graph=False) as native_recovery_dispatch:
            self._op(
                "post_attach_if_any",
                self.slow_error,
                False,
                native_recovery_graph.raw_cuda_graph(),
            )
        self.native_recovery_graphs = native_recovery_graph, native_recovery_dispatch
        with capture(self.device) as recovery_graph:
            self._recover_fast()
        with capture(self.device, keep_graph=False) as recovery_dispatch:
            self._op(
                "post_attach_if_any",
                self.fast_error,
                False,
                recovery_graph.raw_cuda_graph(),
            )
        self.recovery_graphs = recovery_graph, recovery_dispatch
        children = []
        for name, args in zip(
            ("high_emit_trusted_candidates_live_k", "high_emit_fp16_candidates_live_k"),
            self.high_args,
        ):
            with capture(self.device) as child:
                self._op(name, *args)
            children.append(child)
        self.high_graphs = tuple(children)
        with capture(self.device, keep_graph=False) as graph:
            self.graph = graph
            self._body()
        graph.replay()
        recent = min(self.max_q, RECENT)
        caller.map_topk_vote_stats_(
            self.output,
            self.permutation[:SEED],
            self.zero,
            self.votes[:SEED],
            recent,
        )
        caller.carry_votes_topk_reset_(
            self.votes[:SEED],
            self.hot,
            self.carry_partial,
            self.carry_state,
            SEED,
            recent,
            0,
        )
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
        caller.carry_votes_topk_reset_(
            votes,
            self.hot,
            self.carry_partial,
            self.carry_state,
            SEED,
            4,
            0,
        )

    def _gather(self, keys, scales, count: int, common_end: int) -> None:
        if self.plan_epoch + 1 >= 1 << 30:
            self.hot_epoch.zero_()
            self.plan_epoch = 0
        self.plan_epoch += 1
        # Dedicated permutation ranks stay live through the serialized gather.
        caller.plan_and_permuted_gather_rho_out(
            self.hot,
            self.hot_epoch[:count],
            self.permutation[:count],
            self.pair_rank[:count],
            self.swap_a,
            self.swap_b,
            self.plan_counts,
            0,
            common_end,
            self.plan_epoch,
            keys[:count],
            scales[:count],
            self.pk[:count],
            self.ps[:count],
            self.rho[:count],
        )
        self.live_nk.fill_(count)

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
        """Write native MX/BF16-score top2048 logical K IDs for one request.

        Q is contiguous E4M3 [Q,H,128]; K is contiguous E4M3 [K,128]; scales
        are contiguous exact FP32 UE8M0 decodes or I32 packed repeated bytes. Weights
        are contiguous BF16/FP32 [Q,H]; BF16 values promote exactly into owned FP32
        storage. FP32 inputs must contain the already-rounded weight values used
        by the calling model. Scale decode/copy uses explicitly owned aligned
        storage. Intervals are contiguous I32 [Q]; key_offset is subtracted in
        owned staging storage to obtain request-local intervals within [0,K].
        ``common_end`` is visible to every query. With ``causal=True``, intervals
        are [0, common_end + row); otherwise they share the frozen bootstrap.
        Output is I32 [Q,2048] with contiguous columns and no input aliasing.
        The caller keeps all borrowed storage stable through stream completion.
        This owner's uses must not overlap, and prepare() must precede the call.
        Returns False on unsupported metadata without mutating output or state.
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
                    caller.map_topk_vote_stats_(
                        self.output[:logical],
                        self.permutation[: visible if causal else count],
                        self.zero[:logical],
                        votes,
                        recent,
                    )
                    output[begin:end].copy_(self.output[:logical])
                    if recent:
                        caller.carry_votes_topk_reset_(
                            votes,
                            self.hot,
                            self.carry_partial,
                            self.carry_state,
                            SEED,
                            recent,
                            0,
                        )
                return True
            finally:
                self.done.record(stream)
