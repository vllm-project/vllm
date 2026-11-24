# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 the FlashTopK authors.
# Licensed under the Apache License, Version 2.0.
"""The selection: every query row's exact top-k keys under the indexer score, in
canonical order.

For row r with window [ks[r], ke[r]) of the keys, the score of key t is

    s(r, t) = kv_scale[t] * sum_h weights[r, h] * relu(q[r, h] . kv[t])

(q and kv FP8 E4M3), and the output is the topk keys of the window with the largest
scores, sorted by score descending, then by key index ascending; -0.0 ties +0.0.
A window shorter than topk is padded with (-inf, -1).

One call runs, all on one stream and without reading anything back:

  1. the shared sample: the scores of a few thousand keys spread over the KV (the dense
     scan, the scan's bits);
  2. calibration (seed_prep): each row's bucket map and a gate aimed at a little more
     than topk candidates;
  3. the scan, or key pruning (an upper bound's mask, then the full score of the keys it
     keeps): the keys whose score passes the gate become the row's candidates. From
     plan.ADAPTIVE_MIN_KEYS keys the GPU chooses between them from the planner's
     estimate of the share of keys the bound keeps; shorter calls scan;
  4. the selection: the exact top-k of each row's candidates, in canonical order, with
     the call's health;
  5. recovery: rows that overflowed their candidate buffer or came up short are rescored
     densely and reselected.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from . import plan as planner
from .kernels import launch
from .kernels.selection import HEALTH_WORDS, RECOVERY_ROWS
from .recovery import Recovery, stage_rows

NUM_BUCKETS = 256
SAMPLE_LEN = 8192
SHARED_MIN_SAMPLES = 2048
KEYS_PER_SAMPLE = 128
# A sampled row's gate starts at about GATE_SLACK * topk candidates (recovery covers a
# miss), and no tighter than the K_MIN-th best sample.
GATE_SLACK = 2.0
K_MIN = 32
# A row read whole aims its gate at its (topk + SEED_MARGIN)-th best key, so a gate
# bucket of exactly topk keys cannot drop the topk-th.
SEED_MARGIN = 1
MAX_TOPK = 4096


def sample_count(seq_len_kv: int) -> int:
    """Keys in the shared sample: one per KEYS_PER_SAMPLE, within [SHARED_MIN_SAMPLES,
    SAMPLE_LEN] and the KV."""
    return min(
        max(seq_len_kv // KEYS_PER_SAMPLE, SHARED_MIN_SAMPLES), SAMPLE_LEN, seq_len_kv
    )


def candidate_cap(topk: int, num_q: int, seq_len_kv: int) -> int:
    """Candidate slots per row. The scan tightens each row's gate as it goes, which
    takes a wave of row tiles: calls of fewer rows keep their sampled gates longer and
    get more slots, as do calls past 1M keys, whose steady state alone nears 16K
    candidates."""
    cap = max(4 * topk, 16384)
    if seq_len_kv > 1024 * 1024:
        cap = max(cap, 32768)
    if num_q < 128:
        extrapolated = (seq_len_kv * topk + 2 * SAMPLE_LEN - 1) // (2 * SAMPLE_LEN)
        cap = max(cap, (cap + extrapolated + 255) // 256 * 256)
    elif num_q < 1024:
        cap = max(cap, 32768)
    return cap


def _arena_elements(rows: int, seq_len_kv: int, topk: int) -> int:
    width = math.ceil(sample_count(seq_len_kv) / 4) * 4
    return max(
        2 * rows * candidate_cap(topk, rows, seq_len_kv),
        rows * width,
        RECOVERY_ROWS * seq_len_kv,
    )


class _Take:
    """Consecutive 256-byte aligned views of a uint8 buffer, take(shape, dtype) at a
    time; without a buffer it only counts the bytes they take (used)."""

    def __init__(self, buf: torch.Tensor | None = None):
        self.buf, self.used = buf, 0

    def __call__(self, shape, dtype: torch.dtype) -> torch.Tensor | None:
        start, size = self.used, math.prod(shape) * dtype.itemsize
        self.used = (start + size + 255) // 256 * 256
        return (
            None
            if self.buf is None
            else self.buf[start : start + size].view(dtype).view(shape)
        )


class Selector:
    """Top-k selection of FP8 E4M3 operands, for one device and top-k.

    The selector holds only per-row state (plan); each call's scratch (scratch_bytes:
    the candidates, a slice's scores, recovery's staged slices, the pruning plan and its
    mask) is the caller's, passed to run(). run() never synchronizes, so it can be
    captured in a CUDA graph for one shape. A call of more than chunk_size rows runs in
    slices of chunk_size rows (rows are independent: the slices select the same bits),
    which bounds the scratch. compile() compiles every CuTe kernel variant the calls can
    take (the Triton planner compiles at its first use).
    """

    def __init__(self, topk: int, device: torch.device, chunk_size: int):
        if not 1 <= topk <= MAX_TOPK:
            raise ValueError(f"topk must be in [1, {MAX_TOPK}], got {topk}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.topk = topk
        self.device = torch.device(device)
        self.chunk_size = chunk_size
        self.max_rows = 0
        self.max_kv = 0
        self._recovery: Recovery | None = None
        self._scratch_bytes: dict[tuple[int, int], int] = {}

    def compile(self) -> None:
        """Compile every CuTe kernel variant of this architecture and top-k."""
        launch.warmup(torch.cuda.get_device_capability(self.device)[0], self.topk)

    def plan(self, max_rows: int, max_kv: int) -> Selector:
        """Allocate the per-row state of calls of up to max_rows rows (at most
        chunk_size at once) and max_kv keys, growing an earlier plan."""
        max_rows = max(min(max_rows, self.chunk_size), self.max_rows)
        max_kv = max(max_kv, self.max_kv)
        if (max_rows, max_kv) == (self.max_rows, self.max_kv) and self._recovery:
            return self
        dev = self.device
        i32 = dict(dtype=torch.int32, device=dev)
        f32 = dict(dtype=torch.float32, device=dev)
        self._origin = torch.empty(max_rows, **f32)
        self._inv_delta = torch.empty(max_rows, **f32)
        self._th_bucket = torch.empty(max_rows, **i32)
        self._cand_cnt = torch.empty(max_rows, **i32)
        self._lo = torch.empty(max_rows, **i32)
        self._hi = torch.empty(max_rows, **i32)
        self._scan_ks = torch.empty(max_rows, **i32)
        self._prune_ks = torch.empty(max_rows, **i32)
        self._flag = torch.zeros(1, **i32)
        self._health = torch.zeros(HEALTH_WORDS, dtype=torch.int64, device=dev)
        self._bad_rows = torch.zeros(RECOVERY_ROWS, **i32)
        self._recovery = Recovery(max_rows, self.topk, dev)
        self.max_rows, self.max_kv = max_rows, max_kv
        return self

    def scratch_bytes(self, num_q: int, seq_len_kv: int) -> int:
        """Bytes of scratch a run over num_q rows (chunk_size of them at once) and
        seq_len_kv keys takes."""
        rows = min(num_q, self.chunk_size)
        key = (rows, seq_len_kv)
        if key not in self._scratch_bytes:
            used = 0
            # Calls of fewer rows get more candidate slots each (candidate_cap).
            for r in (rows, min(rows, 1023), min(rows, 127)):
                take = _Take()
                self._carve(take, r, seq_len_kv)
                used = max(used, take.used)
            self._scratch_bytes[key] = used
        return self._scratch_bytes[key]

    def _carve(self, take: _Take, num_q: int, seq_len_kv: int) -> SimpleNamespace:
        """The scratch of one pass over num_q rows and seq_len_kv keys, from take."""
        topk, stage = self.topk, stage_rows(seq_len_kv)
        index = (
            self.device.index
            if self.device.index is not None
            else torch.accelerator.current_device_index()
        )
        can_prune = planner.adaptive_threshold(index, num_q, seq_len_kv) > 0.0
        return SimpleNamespace(
            arena=take((_arena_elements(num_q, seq_len_kv, topk),), torch.float32),
            stage_scores=take((stage, topk), torch.float32),
            stage_indices=take((stage, topk), torch.int32),
            plan=planner.plan_scratch(take, num_q, seq_len_kv) if can_prune else None,
            mask=take(launch.mask_shape(num_q, seq_len_kv), torch.int32)
            if can_prune
            else None,
        )

    def _check(self, q, kv, weights, ks, ke, kv_scales, out):
        if q.device != self.device:
            raise ValueError(f"q is on {q.device}, the selector on {self.device}")
        if not launch.supported(q, kv):
            raise ValueError(
                f"unsupported operands: q {tuple(q.shape)} {q.dtype}, "
                f"kv {tuple(kv.shape)} {kv.dtype} on SM{launch.arch(q)}0 "
                "(need FP8 E4M3 [num_q, 32, 128] and [seq_len_kv, 128], SM90 or SM100)"
            )
        num_q = q.shape[0]
        if weights.shape != (num_q, 32) or weights.dtype != torch.float32:
            raise ValueError(
                f"weights must be float32 [{num_q}, 32], "
                f"got {weights.dtype} {tuple(weights.shape)}"
            )
        if ks.shape != (num_q,) or ke.shape != (num_q,):
            raise ValueError(
                f"ks and ke must be [{num_q}], "
                f"got {tuple(ks.shape)} and {tuple(ke.shape)}"
            )
        if kv_scales.numel() != kv.shape[0] or kv_scales.dtype != torch.float32:
            raise ValueError(f"kv_scales must be float32 of {kv.shape[0]} elements")
        if out.shape != (num_q, self.topk) or out.dtype != torch.int32:
            raise ValueError(f"out must be int32 [{num_q}, {self.topk}]")

    @torch.no_grad()
    def run(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        weights: torch.Tensor,
        ks: torch.Tensor,
        ke: torch.Tensor,
        kv_scales: torch.Tensor,
        out: torch.Tensor,
        scratch: torch.Tensor,
    ) -> None:
        """Select into out [num_q, topk] int32 the top-k of q [num_q, 32, 128] against
        kv [seq_len_kv, 128] (both FP8 E4M3, kv_scales [seq_len_kv] float32), weights
        [num_q, 32] float32 and row windows [ks, ke) (int32, or converted): absolute
        key indices, -1 padded. scratch: uint8, 256-byte aligned, at least
        scratch_bytes(num_q, seq_len_kv)."""
        self._check(q, kv, weights, ks, ke, kv_scales, out)
        num_q, seq_len_kv = q.shape[0], kv.shape[0]
        if num_q == 0:
            return
        self.plan(num_q, seq_len_kv)
        need = self.scratch_bytes(num_q, seq_len_kv)
        if scratch.dtype != torch.uint8 or scratch.numel() < need:
            raise ValueError(f"run() needs {need} bytes of uint8 scratch")
        q, kv, weights = q.contiguous(), kv.contiguous(), weights.contiguous()
        ks = ks.to(torch.int32).contiguous()
        ke = ke.to(torch.int32).contiguous()
        kv_scales = kv_scales.reshape(-1).contiguous()
        for a in range(0, num_q, self.chunk_size):
            b = min(num_q, a + self.chunk_size)
            self._run_rows(
                q[a:b], kv, kv_scales, weights[a:b], ks[a:b], ke[a:b], out[a:b], scratch
            )

    def _run_rows(self, q, kv, kv_scales, weights, ks, ke, indices, scratch):
        """One pass over at most chunk_size rows, writing their selection to indices."""
        num_q, seq_len_kv = q.shape[0], kv.shape[0]
        topk = self.topk
        cap = candidate_cap(topk, num_q, seq_len_kv)
        num_samples = sample_count(seq_len_kv)
        width = math.ceil(num_samples / 4) * 4
        s = self._carve(_Take(scratch), num_q, seq_len_kv)
        arena = s.arena
        sample = arena[: num_q * width].view(num_q, width)
        cand_val = arena[: num_q * cap].view(num_q, cap)
        cand_idx = (
            arena[num_q * cap : 2 * num_q * cap].view(torch.int32).view(num_q, cap)
        )
        origin, inv_delta = self._origin[:num_q], self._inv_delta[:num_q]
        th_bucket, cand_cnt = self._th_bucket[:num_q], self._cand_cnt[:num_q]
        health, bad_rows = self._health, self._bad_rows
        lo, hi = self._lo[:num_q], self._hi[:num_q]
        seed_topk = min(topk + SEED_MARGIN, cap)
        gate = (origin, inv_delta, th_bucket, cand_val, cand_idx, cand_cnt)

        with torch.accelerator.device_index(q.device.index), launch.sm90_q_scope(q):
            launch.sample_scores(
                q, kv, kv_scales, weights, ks, ke, num_samples, sample, cand_cnt, lo,
                hi, health,
            )  # fmt: skip
            launch.seed_prep(
                sample, ks, ke, lo, hi, seed_topk, GATE_SLACK * topk,
                min(K_MIN, seed_topk), origin, inv_delta, th_bucket, cand_cnt,
            )  # fmt: skip
            threshold = planner.adaptive_threshold(q.device.index, num_q, seq_len_kv)
            if threshold <= 0.0:
                launch.scan(q, kv, kv_scales, weights, ks, ke, *gate)
            else:
                # Both paths launch; the scan sees empty windows when the call prunes,
                # the pruning kernels when it does not.
                scan_ks, prune_ks = self._scan_ks[:num_q], self._prune_ks[:num_q]
                plan = planner.adaptive_estimate(
                    q, weights, kv, kv_scales, ks, ke, origin, inv_delta, th_bucket,
                    self._flag, scan_ks, prune_ks, threshold, s.plan,
                )  # fmt: skip
                launch.scan(q, kv, kv_scales, weights, scan_ks, ke, *gate)
                q_prune, w_prune, allowance, knsc = planner.adaptive_finish(
                    plan, kv, self._flag
                )
                launch.prune_scan(
                    q, q_prune, kv, kv_scales, knsc, weights, w_prune, allowance,
                    prune_ks, ke, *gate, s.mask,
                )  # fmt: skip
            launch.select(
                cand_val, cand_idx, cand_cnt, ks, ke, indices, health, bad_rows
            )
            assert self._recovery is not None  # planned above
            self._recovery.run(
                q, kv, kv_scales, weights, ks, ke, health, bad_rows, indices,
                arena, s.stage_scores, s.stage_indices,
            )  # fmt: skip
