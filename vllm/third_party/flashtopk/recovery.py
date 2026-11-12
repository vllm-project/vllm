# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Exact recovery of the rows the selection could not finish, enqueued on every call and decided on the device.

A row is bad when its candidates overflowed the buffer or it selected fewer than min(topk, window) keys. The
selection counts bad rows and records the first RECOVERY_ROWS of them; triage (kernels/triage.py) then picks:

  * per-row recovery, at most RECOVERY_ROWS bad rows: their q rows are gathered into slots, scored densely over their
    windows, and each slot's exact top-k replaces its row. Windows of at least SPLIT_MIN_KEYS keys are cut into
    slices whose top-ks are selected first, so a long row keeps the whole GPU busy;
  * whole-call recovery, more bad rows: every row is rescored densely, `chunk` rows at a time, and reselected;
  * nothing: every stage sees empty windows and unmapped slots, and returns at once.

A recovered row is the exact top-k of the dense scan's scores, which are the scan's own bits, in canonical order:
the output does not depend on which path produced it.
"""

from __future__ import annotations

import math

import torch

from .kernels import launch
from .kernels.common import HEAD_DIM, NEXT_N, NUM_HEADS
from .kernels.selection import RECOVERY_ROWS

SPLIT_MIN_KEYS = 32768
SPLIT_KEYS = 16384
MAX_SPLITS = 64


def recovery_splits(seq_len_kv: int) -> int:
    """Slices a per-row recovery slot is cut into: one per SPLIT_KEYS keys from SPLIT_MIN_KEYS keys."""
    if seq_len_kv < SPLIT_MIN_KEYS:
        return 1
    return min(MAX_SPLITS, math.ceil(seq_len_kv / SPLIT_KEYS))


def dense_rows(arena_elements: int, seq_len_kv: int, num_q: int) -> int:
    """Rows whole-call recovery scores at a time in an arena of `arena_elements` floats: whole 8-row tiles."""
    rows = arena_elements // max(seq_len_kv, 1) // NEXT_N * NEXT_N
    return max(NEXT_N, min(rows, math.ceil(num_q / NEXT_N) * NEXT_N))


class Recovery:
    """One plan's recovery buffers, for up to `max_rows` rows and `max_kv` keys. `arena` is the call's float
    workspace (free once the selection has run), at least RECOVERY_ROWS * max_kv floats."""

    def __init__(self, max_rows: int, max_kv: int, topk: int, dtype: torch.dtype, arena: torch.Tensor):
        dev = arena.device
        if arena.numel() < RECOVERY_ROWS * max_kv:
            raise ValueError("the recovery arena must hold RECOVERY_ROWS rows of dense scores")
        i32 = dict(dtype=torch.int32, device=dev)
        words_per_head = HEAD_DIM * torch.empty((), dtype=dtype).element_size() // 4
        self.topk = topk
        self.arena = arena
        self.row_map = torch.empty(RECOVERY_ROWS, **i32)
        self.slot_ks = torch.empty(RECOVERY_ROWS, **i32)
        self.slot_ke = torch.empty(RECOVERY_ROWS, **i32)
        self.q_gather = torch.empty(RECOVERY_ROWS * NUM_HEADS * words_per_head, **i32)
        self.w_gather = torch.empty(RECOVERY_ROWS, NUM_HEADS, dtype=torch.float32, device=dev)
        self.whole_map = torch.empty(max_rows, **i32)
        self.whole_ks = torch.empty(max_rows, **i32)
        self.whole_ke = torch.empty(max_rows, **i32)
        self.scratch = torch.empty(max(max_rows, RECOVERY_ROWS), **i32)
        self.max_splits = recovery_splits(max_kv)
        stage_rows = RECOVERY_ROWS * self.max_splits if self.max_splits > 1 else 0
        self.stage_scores = torch.empty(stage_rows, topk, dtype=torch.float32, device=dev)
        self.stage_indices = torch.empty(stage_rows, topk, **i32)

    def run(self, q, kv, kv_scales, weights, ks, ke, health, bad_rows, scores, indices) -> None:
        """Enqueue the recovery of this call's selection (scores / indices, its health and bad-row table)."""
        num_q, seq_len_kv = q.shape[0], kv.shape[0]
        sm90 = launch.arch(q) == 9
        launch.triage(health, bad_rows, ks, ke, q, weights, self.row_map, self.slot_ks, self.slot_ke, self.q_gather,
                      self.w_gather, self.whole_map[:num_q], self.whole_ks[:num_q], self.whole_ke[:num_q])

        # per-row: the gathered slots
        dense = self.arena[: RECOVERY_ROWS * seq_len_kv].view(RECOVERY_ROWS, seq_len_kv)
        q_slots = self.q_gather.view(q.dtype).view(RECOVERY_ROWS, NUM_HEADS, HEAD_DIM)
        operand = launch.as_operand(q_slots).view(-1, HEAD_DIM, 1) if sm90 else None
        launch.scan_dense(q_slots, kv, kv_scales, self.w_gather, self.slot_ks, self.slot_ke, dense,
                          self.scratch[:RECOVERY_ROWS], q_operand=operand)
        splits = min(self.max_splits, recovery_splits(seq_len_kv))
        if splits > 1:
            stage = RECOVERY_ROWS * splits
            launch.select_dense(dense, self.slot_ks, self.slot_ke, self.row_map, self.stage_scores[:stage],
                                self.stage_indices[:stage], splits)
            launch.select_staged(self.stage_scores[:stage], self.stage_indices[:stage], self.row_map, scores, indices)
        else:
            launch.select_dense(dense, self.slot_ks, self.slot_ke, self.row_map, scores, indices)

        # whole call, `chunk` rows at a time
        chunk = dense_rows(self.arena.numel(), seq_len_kv, num_q)
        full_operand = launch.q_operand_sm90(q) if sm90 else None
        for off in range(0, num_q, chunk):
            rows = min(chunk, num_q - off)
            dense = self.arena[: rows * seq_len_kv].view(rows, seq_len_kv)
            part = slice(off, off + rows)
            operand = None
            if sm90:
                operand = full_operand[off * NUM_HEADS: (off + math.ceil(rows / NEXT_N) * NEXT_N) * NUM_HEADS]
            launch.scan_dense(q[part], kv, kv_scales, weights[part], self.whole_ks[part], self.whole_ke[part], dense,
                              self.scratch[:rows], q_operand=operand)
            launch.select_dense(dense, self.whole_ks[part], self.whole_ke[part], self.whole_map[part], scores,
                                indices)
