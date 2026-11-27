# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MX recovery and FP32 sparse scoring for DoSA."""

import weakref

import torch

from vllm import _custom_ops as stock_ops
from vllm.model_executor.layers.indexer_topk import _HAS_DEEP_SELECT_C, deep_select_topk
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    get_mqa_logits_metadata,
    mqa_logits_storage,
)


@triton.jit
def _collect_failed(ERROR, ROW_MAP, COUNT, ROWS: tl.constexpr):
    row = tl.program_id(0) * 256 + tl.arange(0, 256)
    failed = tl.load(ERROR + row, row < ROWS, 0) != 0
    prefix = tl.cumsum(failed.to(tl.int32))
    reserved = tl.sum(failed.to(tl.int32), 0)
    if reserved > 0:
        first = tl.atomic_add(COUNT, reserved, sem="relaxed")
        tl.store(ROW_MAP + first + prefix - 1, row, failed)


@triton.jit
def _gather(
    Q,
    W,
    START,
    END,
    ERROR,
    LIVE,
    ROW_MAP,
    CURSOR,
    COUNT,
    OUT_Q,
    OUT_W,
    OUT_START,
    OUT_END,
    LENGTH,
    HEADS: tl.constexpr,
    KEYS: tl.constexpr,
):
    slot = tl.program_id(0)
    index = tl.load(CURSOR) + slot
    active = index < tl.load(COUNT)
    row = tl.load(ROW_MAP + index, active, 0)
    start = tl.load(START + row, active, 0)
    end = tl.load(END + row, active, 0)
    live = tl.load(LIVE)
    valid = (live >= 0) & (live <= KEYS) & (start >= 0) & (start <= end)
    valid &= end <= live
    tl.atomic_or(ERROR + row, 2, active & ~valid, sem="relaxed")
    active &= valid
    tl.store(OUT_START + slot, tl.where(active, start, 0))
    tl.store(OUT_END + slot, tl.where(active, end, 0))
    tl.store(LENGTH + slot, tl.where(active, end - start, 0))
    head = tl.arange(0, HEADS)
    element = tl.arange(0, HEADS * 128)
    tl.store(
        OUT_Q + slot * HEADS * 128 + element,
        tl.load(Q + row * HEADS * 128 + element, active, 0),
    )
    tl.store(OUT_W + slot * HEADS + head, tl.load(W + row * HEADS + head, active, 0))


@triton.jit
def _publish(
    SELECTED,
    START,
    ORDER,
    ROW_MAP,
    CURSOR,
    COUNT,
    OUTPUT,
    TOPK: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
):
    slot = tl.program_id(0)
    local_id = tl.program_id(1) * 256 + tl.arange(0, 256)
    index = tl.load(CURSOR) + slot
    active = index < tl.load(COUNT)
    row = tl.load(ROW_MAP + index, active, 0)
    dest = tl.load(ORDER + row, active, 0)
    begin = tl.load(START + slot, active, 0)
    selected = tl.load(SELECTED + slot * TOPK + local_id, active, -1)
    tl.store(
        OUTPUT + dest * OUTPUT_STRIDE + local_id,
        tl.where(selected >= 0, selected + begin, -1),
        active,
    )


def native_mqa_topk(q, keys, weights, bounds, output, logits_layout):
    starts, ends, lengths, row_starts = bounds
    max_k, logits_dtype, storage = logits_layout
    schedule = get_mqa_logits_metadata(starts, ends, len(keys[0]), q[0].shape[1])
    scores = fp8_fp4_mqa_logits(
        q,
        keys,
        weights,
        starts,
        ends,
        max_k,
        logits_dtype=logits_dtype,
        schedule_meta=schedule,
        output=storage,
    )
    topk = output.shape[1]
    if row_starts is None:
        deep_select_topk(scores, topk, end=lengths, output_idx=output)
    else:
        stock_ops.top_k_per_row_prefill(
            scores, row_starts, lengths, output, len(scores), *scores.stride(), topk
        )


class NativeRecovery:
    """Compact failed query rows into windows for the native scorer."""

    def __init__(self, owner, packed_scales, storage, batch):
        self.owner = weakref.proxy(owner)
        self.batch = batch
        self.dtype = owner.logits_dtype
        self.use_deep_select = self.dtype == torch.bfloat16 or (
            batch == 128 and _HAS_DEEP_SELECT_C
        )
        self.attach_while = owner._kernel("native_fp32_attach_while")
        self.logits = mqa_logits_storage(
            storage, batch, owner.max_k, owner.heads, self.dtype
        )
        self.row_map = owner._empty((owner.rows,), torch.int32)
        self.state = torch.zeros(2, dtype=torch.int32, device=owner.device)
        self.cursor, self.count = self.state[:1], self.state[1:]
        self.q = owner._empty((batch, owner.heads, 128), torch.float8_e4m3fn)
        self.w = owner._empty((batch, owner.heads), self.dtype)
        self.unit = torch.full_like(self.w, 0x7F7F7F7F, dtype=torch.int32)
        self.packed = packed_scales
        self.starts = self.row_map.new_empty(batch)
        self.ends = torch.empty_like(self.starts)
        self.lengths = torch.empty_like(self.starts)
        self.zeros = None if self.use_deep_select else torch.zeros_like(self.starts)
        self.selected = self.row_map.new_empty((batch, 2048))
        self.leaf = None

    def body(self):
        owner = self.owner
        _gather[(self.batch,)](
            owner.q.view(torch.uint8),
            owner.w,
            owner.starts,
            owner.ends,
            owner.error,
            owner.live_nk,
            self.row_map,
            self.cursor,
            self.count,
            self.q.view(torch.uint8),
            self.w,
            self.starts,
            self.ends,
            self.lengths,
            owner.heads,
            owner.max_k,
            num_warps=8,
        )
        native_mqa_topk(
            (self.q, self.unit),
            (owner.pk, self.packed),
            self.w,
            (self.starts, self.ends, self.lengths, self.zeros),
            self.selected,
            (owner.max_k, self.dtype, self.logits),
        )
        topk = self.selected.shape[1]
        _publish[(self.batch, topk // 256)](
            self.selected,
            self.starts,
            owner.order,
            self.row_map,
            self.cursor,
            self.count,
            owner.output,
            topk,
            owner.output.stride(0),
        )

    def run(self):
        error = self.owner.error
        self.count.zero_()
        _collect_failed[(triton.cdiv(error.numel(), 256),)](
            error, self.row_map, self.count, error.numel()
        )
        if self.leaf is None:
            self.body()
        else:
            self.attach_while(
                self.cursor, self.count, self.batch, self.leaf.raw_cuda_graph()
            )


class NativeFP32Stages:
    """Direct MX refinement/E plus exact native recovery for FP32 workspaces."""

    def __init__(self, owner, seed, scratch):
        self.owner = weakref.proxy(owner)
        self.seed = seed
        self.mx_refine = owner._kernel("refine_mx_exact_mx_packed20")
        self.mx_boundary = owner._kernel("refine_mx_score_boundary")
        self.mx_q = owner.q.view(torch.float8_e4m3fn)
        self.packed_scales = owner._empty((owner.max_k,), torch.int32).fill_(0x7F7F7F7F)
        self.unit_q = owner._empty(owner.w.shape, torch.int32).fill_(0x7F7F7F7F)
        self.seed_ends = torch.full_like(owner.zero, seed)
        # Seed preparation joins before candidate compaction writes scratch.
        self.seed_logits = mqa_logits_storage(
            scratch, owner.rows, seed, owner.heads, torch.float32
        )
        self.boundary_tasks = owner._empty((owner.rows, 3), torch.int32)
        self.boundary_state = torch.zeros(2, device=owner.device, dtype=torch.int32)

    def score_seed(self, q):
        owner = self.owner
        return fp8_fp4_mqa_logits(
            (q, self.unit_q),
            (owner.pk[: self.seed], self.packed_scales[: self.seed]),
            owner.w,
            owner.zero,
            self.seed_ends,
            self.seed,
            logits_dtype=torch.float32,
            output=self.seed_logits,
        )

    def refine(self):
        owner = self.owner
        for common, tasks, state in (
            (False, owner.solo_tasks, owner.solo_state),
            (True, owner.common_tasks, owner.common_state),
        ):
            self.mx_refine(
                self.mx_q,
                owner.pk,
                self.packed_scales,
                owner.w,
                owner.safe_tau,
                owner.id_pool,
                tasks,
                state,
                owner.values,
                owner.ids,
                owner.record_pages,
                owner.counts,
                owner.error,
                owner.record_head,
                owner.starts,
                owner.ends,
                owner.live_nk,
                owner.sms,
                self.seed,
                common,
            )

    def boundary(self):
        owner = self.owner
        owner._op(
            "native_fp32_boundary_tasks",
            owner.partition,
            owner.error,
            self.boundary_tasks,
            self.boundary_state,
            owner.select_boundary.stride(0),
        )
        self.mx_boundary(
            self.mx_q,
            owner.pk,
            self.packed_scales,
            owner.w,
            owner.select_boundary,
            self.boundary_tasks,
            self.boundary_state,
            owner.select_boundary_scores,
            owner.starts,
            owner.ends,
            owner.error,
            owner.live_nk,
            owner.sms,
        )
        owner._op(
            "select_finish_boundary_native_fp32",
            owner.select_boundary_scores,
            owner.select_boundary,
            owner.partition,
            owner.permutation,
            owner.order,
            owner.output,
            owner.error,
        )
