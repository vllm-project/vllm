# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""The selection: every query row's exact top-k keys under the indexer score, in canonical order.

For row r with window [ks[r], ke[r]) of the keys, the score of key t is

    s(r, t) = kv_scale[t] * sum_h weights[r, h] * relu(q[r, h] . kv[t])

(kv_scale is 1 for BF16 keys), and the output is the topk keys of the window with the largest scores, sorted by score
descending, then by key index ascending; -0.0 ties +0.0. A window shorter than topk is padded with (-inf, -1).

One call runs, all on one stream and without reading anything back:

  1. the shared sample: the scores of a few thousand keys spread over the KV (the dense scan, the scan's bits);
  2. calibration (seed_prep): each row's bucket map and a gate aimed at a little more than topk candidates;
  3. the scan, or key pruning (an upper bound's mask, then the full score of the keys it keeps): the keys whose
     score passes the gate become the row's candidates. With pruning="adaptive" the GPU chooses from the planner's
     estimate of the share of keys the bound keeps;
  4. the selection: the exact top-k of each row's candidates, in canonical order, with the call's health;
  5. recovery: rows that overflowed their candidate buffer or came up short are rescored densely and reselected.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Optional, Tuple, Union

import torch

from . import plan as planner
from .kernels import launch
from .kernels.selection import H_CAND_MAX, HEALTH_WORDS, RECOVERY_ROWS
from .recovery import Recovery

NUM_BUCKETS = 256
SAMPLE_LEN = 8192
SHARED_MIN_SAMPLES = 2048
KEYS_PER_SAMPLE = 128
# A sampled row's gate starts at about GATE_SLACK * topk candidates (recovery covers a miss), and no tighter than
# the K_MIN-th best sample.
GATE_SLACK = 2.0
K_MIN = 32
# A row read whole aims its gate at its (topk + SEED_MARGIN)-th best key, so a gate bucket of exactly topk keys
# cannot drop the topk-th.
SEED_MARGIN = 1
MAX_TOPK = 4096
PRUNING = ("off", "always", "adaptive")
# Query rows a selector works on at once: a call with more rows runs in slices of this many, which bounds the workspace
# (about 8 * chunk_size * max(4 * topk, 16384) bytes); rows are independent, so the slices select the same bits.
DEFAULT_CHUNK_SIZE = 65536


def sample_count(seq_len_kv: int) -> int:
    """Keys in the shared sample: one per KEYS_PER_SAMPLE, within [SHARED_MIN_SAMPLES, SAMPLE_LEN] and the KV."""
    return min(max(seq_len_kv // KEYS_PER_SAMPLE, SHARED_MIN_SAMPLES), SAMPLE_LEN, seq_len_kv)


def candidate_cap(topk: int, num_q: int, seq_len_kv: int) -> int:
    """Candidate slots per row. The scan tightens each row's gate as it goes, which takes a wave of row tiles: calls
    of fewer rows keep their sampled gates longer and get more slots, as do calls past 1M keys, whose steady state
    alone nears 16K candidates."""
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
    return max(2 * rows * candidate_cap(topk, rows, seq_len_kv), rows * width, RECOVERY_ROWS * seq_len_kv)


def workspace_bytes(max_rows: int, max_kv: int, topk: int = 2048, chunk_size: Optional[int] = None,
                    return_scores: bool = False) -> int:
    """Device memory a Selector planned for (max_rows, max_kv) holds, about: dominated by the candidate buffers
    (2 * 4 bytes * rows * candidate_cap), for at most chunk_size rows, plus a slice's scores without return_scores."""
    max_rows = min(max_rows, chunk_size or DEFAULT_CHUNK_SIZE)
    rows = [r for r in (max_rows, min(max_rows, 1023), min(max_rows, 127)) if r > 0]
    arena = max(_arena_elements(r, max_kv, topk) for r in rows) * 4
    scores = 0 if return_scores else max_rows * topk * 4
    return arena + scores + max_rows * (4 * 4 + 3 * 4) + RECOVERY_ROWS * 32 * 128 * 2


def windows_from_cu_seqlens(cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                              torch.Tensor]:
    """Causal windows (ks, ke) of packed sequences: query t of sequence i (its queries the last of its keys) sees keys
    cu_k[i] .. cu_k[i] + (len_k - len_q) + t. int32, on the inputs' device."""
    cu_q = cu_seqlens_q.to(torch.int64)
    cu_k = cu_seqlens_k.to(torch.int64)
    len_q = cu_q[1:] - cu_q[:-1]
    len_k = cu_k[1:] - cu_k[:-1]
    seq = torch.repeat_interleave(torch.arange(len_q.numel(), device=cu_q.device), len_q)
    t = torch.arange(int(cu_q[-1]), device=cu_q.device) - cu_q[:-1][seq]
    ks = cu_k[:-1][seq]
    ke = ks + (len_k - len_q)[seq] + t + 1
    return ks.to(torch.int32), ke.to(torch.int32)


def supported(q: torch.Tensor, kv: Optional[torch.Tensor] = None) -> bool:
    """Whether the kernels take these operands: [num_q, 32, 128] queries and [seq_len_kv, 128] keys, both FP8 E4M3
    or both BF16, on an SM90 or SM100 GPU."""
    return launch.supported(q, kv)


@dataclasses.dataclass
class Info:
    """The last call's record, on the device until read (reading synchronizes); the next call overwrites it.

    health: int64[8] (0 rows whose candidates overflowed, 1 rows that came up short, 3 the largest candidate count
    past the buffer, 7 bad rows: the rows recovery rescored); pruned; estimate."""

    health: torch.Tensor
    _pruned: object  # a bool, or the device flag adaptive pruning set
    _estimate: Optional[torch.Tensor] = None  # adaptive pruning's group estimate

    @property
    def pruned(self) -> bool:
        """Whether the call pruned keys rather than scanning every key."""
        return bool(self._pruned) if isinstance(self._pruned, bool) else bool(self._pruned[0])

    @property
    def estimate(self) -> Optional[float]:
        """Adaptive pruning's estimate of the share of keys pruning keeps a row group (None without the estimate)."""
        return None if self._estimate is None else float(self._estimate[0])

    @property
    def bad_rows(self) -> int:
        return int(self.health[7])


def _env_default(name: str, default: str) -> str:
    return os.environ.get("FLASHTOPK_" + name, default)


class Selector:
    """Top-k selection with its workspace.

    plan(max_rows, max_kv) allocates for calls up to that size and compiles every kernel those calls take; run(...)
    then holds no memory past the plan and its outputs (`out` reuses the caller's; a few small per-call buffers go
    through PyTorch's caching allocator) and never synchronizes, so it can be captured in a CUDA graph for one shape.

    pruning: "adaptive" (the GPU chooses per call, from 128K keys), "always" or "off" (default FLASHTOPK_PRUNING, else
    "adaptive"). The rows the sampled gate misses are always rescored (recovery). chunk_size: the rows worked on at
    once (default DEFAULT_CHUNK_SIZE); a call with more rows runs in slices of this many, with the same bits.
    return_scores: True to return the scores with the indices. By default a call returns the indices alone, and the
    scores go to a buffer of one slice's rows in the workspace rather than to an output of every row.
    """

    def __init__(self, topk: int = 2048, num_heads: int = 32, dtype: torch.dtype = torch.float8_e4m3fn,
                 device: Optional[torch.device] = None, pruning: Optional[str] = None,
                 chunk_size: Optional[int] = None, return_scores: bool = False):
        if num_heads != 32:
            raise ValueError(f"num_heads must be 32, got {num_heads}")
        if not 1 <= topk <= MAX_TOPK:
            raise ValueError(f"topk must be in [1, {MAX_TOPK}], got {topk}")
        if dtype not in (torch.float8_e4m3fn, torch.bfloat16):
            raise ValueError(f"dtype must be torch.float8_e4m3fn or torch.bfloat16, got {dtype}")
        pruning = pruning if pruning is not None else _env_default("PRUNING", "adaptive")
        if pruning not in PRUNING:
            raise ValueError(f"pruning must be one of {PRUNING}, got {pruning!r}")
        chunk_size = DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = chunk_size
        self.return_scores = return_scores
        self.topk = topk
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
        self.pruning = pruning
        self.max_rows = 0
        self.max_kv = 0
        self.last_info: Optional[Info] = None
        self._arena = None
        self._recovery = None

    def plan(self, max_rows: int, max_kv: int, warmup: bool = True) -> "Selector":
        """Allocate the workspace for calls of up to max_rows rows (at most chunk_size of them at once) and max_kv
        keys, growing an earlier plan, and with warmup compile every CuTe kernel variant the calls can take, so run()
        never compiles one (the Triton planner compiles at its first use and caches on disk)."""
        if warmup:
            words_per_head = 128 * torch.empty((), dtype=self.dtype).element_size() // 4
            launch.warmup(torch.cuda.get_device_capability(self.device)[0], self.dtype == torch.bfloat16, self.topk,
                          words_per_head)
        max_rows = min(max_rows, self.chunk_size)
        max_rows, max_kv = max(max_rows, self.max_rows), max(max_kv, self.max_kv)
        if max_rows == self.max_rows and max_kv == self.max_kv and self._arena is not None:
            return self
        dev = self.device
        rows = [r for r in (max_rows, min(max_rows, 1023), min(max_rows, 127)) if r > 0]
        elements = max(_arena_elements(r, max_kv, self.topk) for r in rows)
        i32 = dict(dtype=torch.int32, device=dev)
        self._arena = torch.empty(elements, dtype=torch.float32, device=dev)
        self._origin = torch.empty(max_rows, dtype=torch.float32, device=dev)
        self._inv_delta = torch.empty(max_rows, dtype=torch.float32, device=dev)
        self._th_bucket = torch.empty(max_rows, **i32)
        self._cand_cnt = torch.empty(max_rows, **i32)
        self._lo = torch.empty(max_rows, **i32)
        self._hi = torch.empty(max_rows, **i32)
        self._scan_ks = torch.empty(max_rows, **i32)
        self._prune_ks = torch.empty(max_rows, **i32)
        self._flag = torch.zeros(1, **i32)
        self._estimate = torch.zeros(1, dtype=torch.float32, device=dev)
        self._health = torch.zeros(HEALTH_WORDS, dtype=torch.int64, device=dev)
        self._call_health = torch.zeros(HEALTH_WORDS, dtype=torch.int64, device=dev)
        self._call_pruned = torch.zeros(1, **i32)
        self._bad_rows = torch.zeros(RECOVERY_ROWS, **i32)
        self._scores = None if self.return_scores else torch.empty(max_rows, self.topk, dtype=torch.float32, device=dev)
        self._recovery = Recovery(max_rows, max_kv, self.topk, self.dtype, self._arena)
        self.max_rows, self.max_kv = max_rows, max_kv
        return self

    def _check(self, q, kv, weights, ks, ke, kv_scales):
        if q.dtype != self.dtype or kv.dtype != self.dtype:
            raise ValueError(f"q and kv must be {self.dtype}, got {q.dtype} and {kv.dtype}")
        if q.device != self.device:
            raise ValueError(f"q is on {q.device}, the selector on {self.device}")
        if not launch.supported(q, kv):
            raise ValueError(f"unsupported operands: q {tuple(q.shape)} {q.dtype}, kv {tuple(kv.shape)} {kv.dtype} on "
                             f"SM{launch.arch(q)}0 (need [num_q, 32, 128] and [seq_len_kv, 128], SM90 or SM100)")
        num_q = q.shape[0]
        if weights.shape != (num_q, 32) or weights.dtype != torch.float32:
            raise ValueError(f"weights must be float32 [{num_q}, 32], got {weights.dtype} {tuple(weights.shape)}")
        if ks.shape != (num_q,) or ke.shape != (num_q,):
            raise ValueError(f"ks and ke must be [{num_q}], got {tuple(ks.shape)} and {tuple(ke.shape)}")
        fp8 = self.dtype == torch.float8_e4m3fn
        if fp8 and (kv_scales is None or kv_scales.numel() != kv.shape[0] or kv_scales.dtype != torch.float32):
            raise ValueError(f"FP8 keys need float32 kv_scales of {kv.shape[0]} elements")
        if not fp8 and kv_scales is not None:
            raise ValueError("BF16 keys take no kv_scales")

    @torch.no_grad()
    def run(self, q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor,
            kv_scales: Optional[torch.Tensor] = None,
            out: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, None] = None
            ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """(scores [num_q, topk] float32, indices [num_q, topk] int32) of q [num_q, 32, 128] against kv
        [seq_len_kv, 128] (FP8 with kv_scales [seq_len_kv] float32, or BF16), weights [num_q, 32] float32 and row
        windows [ks, ke) (int32, or converted) with return_scores; else the indices alone (`out` then the indices)."""
        self._check(q, kv, weights, ks, ke, kv_scales)
        num_q, seq_len_kv = q.shape[0], kv.shape[0]
        rows = min(num_q, self.chunk_size)
        if rows > self.max_rows or seq_len_kv > self.max_kv:
            self.plan(rows, seq_len_kv, warmup=False)
        topk = self.topk
        if out is None:
            scores = torch.empty(num_q, topk, dtype=torch.float32, device=q.device) if self.return_scores else None
            indices = torch.empty(num_q, topk, dtype=torch.int32, device=q.device)
        else:
            scores, indices = out if self.return_scores else (None, out)
        result = (scores, indices) if self.return_scores else indices
        if num_q == 0:
            return result
        q, kv, weights = q.contiguous(), kv.contiguous(), weights.contiguous()
        ks = ks.to(torch.int32).contiguous()
        ke = ke.to(torch.int32).contiguous()
        if kv_scales is not None:
            kv_scales = kv_scales.reshape(-1).contiguous()
        if num_q <= self.chunk_size:
            slice_scores = scores if self.return_scores else self._scores[:num_q]
            self.last_info = Info(*self._run_rows(q, kv, kv_scales, weights, ks, ke, slice_scores, indices))
            return result

        # Slices of chunk_size rows. Each slice's health is folded into the call's on the device (word H_CAND_MAX is
        # a maximum, the rest are counts), and the call pruned if any slice did.
        total, pruned_any = self._call_health, self._call_pruned
        total.zero_()
        pruned_any.zero_()
        estimate = None
        for a in range(0, num_q, self.chunk_size):
            b = min(num_q, a + self.chunk_size)
            slice_scores = scores[a:b] if self.return_scores else self._scores[:b - a]
            health, pruned, estimate = self._run_rows(q[a:b], kv, kv_scales, weights[a:b], ks[a:b], ke[a:b],
                                                      slice_scores, indices[a:b])
            cand_max = torch.maximum(total[H_CAND_MAX:H_CAND_MAX + 1], health[H_CAND_MAX:H_CAND_MAX + 1])
            total.add_(health)
            total[H_CAND_MAX:H_CAND_MAX + 1].copy_(cand_max)
            if pruned is True:
                pruned_any.fill_(1)
            elif pruned is not False:
                torch.maximum(pruned_any, pruned, out=pruned_any)
        self.last_info = Info(total, pruned_any, estimate)
        return result

    def _run_rows(self, q, kv, kv_scales, weights, ks, ke, scores, indices):
        """One pass over at most chunk_size rows, writing their selection into scores and indices: (health, pruned,
        estimate) as Info takes them."""
        num_q, seq_len_kv = q.shape[0], kv.shape[0]
        topk = self.topk
        cap = candidate_cap(topk, num_q, seq_len_kv)
        num_samples = sample_count(seq_len_kv)
        width = math.ceil(num_samples / 4) * 4
        arena = self._arena
        sample = arena[: num_q * width].view(num_q, width)
        cand_val = arena[: num_q * cap].view(num_q, cap)
        cand_idx = arena[num_q * cap: 2 * num_q * cap].view(torch.int32).view(num_q, cap)
        origin, inv_delta = self._origin[:num_q], self._inv_delta[:num_q]
        th_bucket, cand_cnt = self._th_bucket[:num_q], self._cand_cnt[:num_q]
        health, bad_rows = self._health, self._bad_rows
        lo, hi = self._lo[:num_q], self._hi[:num_q]
        seed_topk = min(topk + SEED_MARGIN, cap)
        slack = GATE_SLACK * topk
        pruned = False
        estimate = None

        scope = launch.sm90_q_scope(q)
        with torch.cuda.device(q.device), scope:
            launch.sample_scores(q, kv, kv_scales, weights, ks, ke, num_samples, sample, cand_cnt, lo, hi, health)
            launch.seed_prep(sample, ks, ke, lo, hi, seed_topk, slack, min(K_MIN, seed_topk), origin, inv_delta,
                             th_bucket, cand_cnt)
            gate = (origin, inv_delta, th_bucket, cand_val, cand_idx, cand_cnt)
            mode = self.pruning
            if mode == "adaptive" and not planner.adaptive_eligible(q, seq_len_kv):
                mode = "off"
            if mode == "off":
                launch.scan(q, kv, kv_scales, weights, ks, ke, *gate)
            elif mode == "always":
                q_prune, w_prune, allowance, knsc, _ = planner.plan_operands(q, weights, kv, kv_scales, ks, ke, origin,
                                                                             inv_delta, th_bucket)
                launch.prune_scan(q, q_prune, kv, kv_scales, knsc, weights, w_prune, allowance, ks, ke, *gate)
                pruned = True
            else:
                # The scan sees empty windows when the call prunes, the pruning kernels when it does not.
                scan_ks, prune_ks = self._scan_ks[:num_q], self._prune_ks[:num_q]
                plan = planner.adaptive_estimate(q, weights, kv, kv_scales, ks, ke, origin, inv_delta, th_bucket,
                                                 self._flag, scan_ks, prune_ks, self._estimate)
                estimate = self._estimate
                launch.scan(q, kv, kv_scales, weights, scan_ks, ke, *gate)
                q_prune, w_prune, allowance, knsc, _ = planner.adaptive_finish(plan, kv, self._flag)
                launch.prune_scan(q, q_prune, kv, kv_scales, knsc, weights, w_prune, allowance, prune_ks, ke, *gate)
                pruned = self._flag
            launch.select(cand_val, cand_idx, cand_cnt, ks, ke, scores, indices, health, bad_rows)
            self._recovery.run(q, kv, kv_scales, weights, ks, ke, health, bad_rows, scores, indices)
        return health, pruned, estimate


_SELECTORS: dict = {}


def select(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor,
           kv_scales: Optional[torch.Tensor] = None, topk: int = 2048, pruning: Optional[str] = None,
           out: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, None] = None, chunk_size: Optional[int] = None,
           return_scores: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Selector(...).run(...) with a selector kept per (device, dtype, topk, pruning, chunk_size, return_scores), its
    workspace grown to the largest call seen."""
    key = (q.device, q.dtype, topk, pruning, chunk_size, return_scores)
    selector = _SELECTORS.get(key)
    if selector is None:
        selector = _SELECTORS[key] = Selector(topk, dtype=q.dtype, device=q.device, pruning=pruning,
                                              chunk_size=chunk_size, return_scores=return_scores)
    return selector.run(q, kv, weights, ks, ke, kv_scales, out)
