# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Host side of the CuTe DSL kernels: their KV-split rules and one compiled kernel per variant, called through TVM FFI
with the torch tensors themselves.

  * scan: every key of a row's window scored, the ones past the row's gate appended to its candidates;
  * scan_dense: every score of a row's window written at its key (the shared sample, recovery);
  * prune_scan: the score upper bound's mask, then the refine of the keys it keeps (the scan's candidates);

on SM100 (scan_sm100.py, prune_sm100.py, refine_sm100.py) or SM90 (the _sm90 files). Lengths are symbolic, so a
variant compiles once per process.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import math
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from .common import BLOCK_KV, HEAD_DIM, NEXT_N, NUM_HEADS, PRUNE_BLOCK_Q, PRUNE_COLS, REFINE_BLOCK_Q
from .prune_sm100 import PruneSm100
from .prune_sm90 import PruneSm90
from .refine_sm100 import RefineSm100
from .refine_sm100 import max_blocks as refine_max_blocks
from .refine_sm90 import MAX_BLOCKS as REFINE_SM90_MAX_BLOCKS
from .refine_sm90 import RefineSm90
from .scan_sm100 import ScanSm100
from .scan_sm90 import ScanSm90
from .prologue import EST_WORDS, Decide, SampleWindows
from .seed_prep import SeedPrep
from .selection import CANDIDATES, DENSE, HEALTH_WORDS, RECOVERY_ROWS, STAGED, Select
from .triage import Triage


def make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1):
    """A compile-time tensor with symbolic strides (but a unit stride on `leading_dim`), aligned to `divisibility`
    elements."""
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    stride = tuple(cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1 for i in range(len(shape)))
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8)


def _stream():
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


def scan_kv_splits(num_q: int, seq_len_kv: int, sm_count: int) -> int:
    """One split once the 8-row tiles alone fill the GPU, else the fewest splits whose CTAs fill their last wave best
    (within 5%), up to eight waves, never fewer than four 128-key blocks per split."""
    total_kv_blocks = math.ceil(seq_len_kv / BLOCK_KV)
    num_q_blocks = math.ceil(num_q / NEXT_N)
    if num_q_blocks >= sm_count:
        return 1
    max_useful = max(1, (total_kv_blocks + 3) // 4)
    best_splits, best = 1, 0.0
    s = 1
    while s <= max_useful and (s == 1 or num_q_blocks * s <= 8 * sm_count):
        ctas = num_q_blocks * s
        eff = ctas / (math.ceil(ctas / sm_count) * sm_count)
        if eff > best + 0.05:
            best, best_splits = eff, s
        s += 1
    return best_splits


def prune_kv_splits(num_q: int, seq_len_kv: int, block_q: int, min_splits: int, sm_count: int) -> int:
    """The fewest KV splits whose CTAs fill their last wave best (within 5%), up to eight waves and at least 256 KV
    blocks a split, and at least min_splits; unlike the scan's rule it also splits once the tiles alone fill the GPU."""
    total_kv_blocks = math.ceil(seq_len_kv / BLOCK_KV)
    num_q_blocks = math.ceil(num_q / block_q)
    best_splits, best = 1, 0.0
    s = 1
    while s <= max(1, total_kv_blocks // 256) and (s == 1 or num_q_blocks * s <= 8 * sm_count):
        ctas = num_q_blocks * s
        eff = ctas / (math.ceil(ctas / sm_count) * sm_count)
        if eff > best + 0.05:
            best, best_splits = eff, s
        s += 1
    return max(best_splits, min_splits)


@functools.cache
def sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def arch(t: torch.Tensor) -> int:
    return torch.cuda.get_device_capability(t.device)[0]


@functools.cache
def _ones(device_index: int, n: int) -> torch.Tensor:
    return torch.ones(n, dtype=torch.float32, device=torch.device("cuda", device_index))


def unit_scales(kv: torch.Tensor) -> torch.Tensor:
    """BF16 keys have no scale: ones (x * 1.0 is exact), in a buffer kept per device and grown in powers of two."""
    n = max(4096, 1 << (kv.shape[0] - 1).bit_length())
    return _ones(kv.device.index, n)


def padded_scales(kv_scales: torch.Tensor) -> torch.Tensor:
    """FP8 kv scales as the kernels' TMA reads them: a multiple of 4 floats (the tail is read, never used)."""
    scales = kv_scales.reshape(-1)
    if scales.numel() % 4:
        scales = torch.nn.functional.pad(scales, (0, 4 - scales.numel() % 4))
    return scales


def as_operand(x: torch.Tensor) -> torch.Tensor:
    """FP8 operands go to the kernels as their bytes."""
    return x if x.dtype == torch.bfloat16 else x.view(torch.uint8)



def _elem(bf16: bool):
    return cutlass.BFloat16 if bf16 else cutlass.Uint8


# ----------------------------------------------------------------------------------------------------------------
# SM100
# ----------------------------------------------------------------------------------------------------------------


@functools.cache
def _compile_scan_sm100(single_split: bool, bf16: bool, dense: bool):
    num_q, num_q_heads, seq_len_kv, num_scales, cand_cap = (cute.sym_int() for _ in range(5))
    elem = _elem(bf16)
    return cute.compile(
        ScanSm100(single_split=single_split, bf16=bf16, dense=dense),
        make_fake_tensor(elem, (seq_len_kv, HEAD_DIM), divisibility=16),
        make_fake_tensor(elem, (num_q_heads, HEAD_DIM, 1), divisibility=16, leading_dim=1),
        make_fake_tensor(Float32, (num_q, NUM_HEADS), divisibility=4),
        make_fake_tensor(Float32, (num_scales,), divisibility=4),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Float32, (num_q, cand_cap)),
        make_fake_tensor(Int32, (num_q, cand_cap)),
        make_fake_tensor(Int32, (num_q,)),
        Int32(0),
        Int32(0),
        Int32(0),
        _stream(),
        options="--enable-tvm-ffi",
    )


def _scan_sm100(q, kv, kv_scales, weights, ks, ke, origin, inv_delta, th_bucket, cand_val, cand_idx, cand_cnt,
                dense: bool, q_operand=None):
    num_q, seq_len_kv = q.shape[0], kv.shape[0]
    num_splits = scan_kv_splits(num_q, seq_len_kv, sm_count(q.device.index))
    bf16 = kv_scales is None
    scales = unit_scales(kv) if bf16 else padded_scales(kv_scales)
    _compile_scan_sm100(num_splits == 1 and not dense, bf16, dense)(
        as_operand(kv), as_operand(q).view(num_q * NUM_HEADS, HEAD_DIM, 1), weights, scales, ks, ke, origin, inv_delta,
        th_bucket, cand_val, cand_idx, cand_cnt, num_q, seq_len_kv, num_splits,
    )


@functools.cache
def _compile_prune_sm100(bf16: bool):
    num_q, pair_heads, seq_len_kv, num_scales, num_words = (cute.sym_int() for _ in range(5))
    elem = _elem(bf16)
    return cute.compile(
        PruneSm100(bf16=bf16),
        make_fake_tensor(elem, (seq_len_kv, HEAD_DIM), divisibility=16),
        make_fake_tensor(elem, (pair_heads, HEAD_DIM, 1), divisibility=16, leading_dim=1),
        make_fake_tensor(Float32, (num_q, PRUNE_COLS), divisibility=4),
        make_fake_tensor(Float32, (num_scales,), divisibility=4),
        make_fake_tensor(Float32, (num_scales,), divisibility=4),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Int32, (num_words,)),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        _stream(),
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_refine_sm100(bf16: bool):
    num_q, num_q_heads, seq_len_kv, num_scales, num_words, cand_cap = (cute.sym_int() for _ in range(6))
    elem = _elem(bf16)
    return cute.compile(
        RefineSm100(bf16=bf16),
        make_fake_tensor(elem, (seq_len_kv, HEAD_DIM), divisibility=16),
        make_fake_tensor(elem, (num_q_heads, HEAD_DIM, 1), divisibility=16, leading_dim=1),
        make_fake_tensor(Float32, (num_q, NUM_HEADS), divisibility=4),
        make_fake_tensor(Float32, (num_scales,), divisibility=4),
        make_fake_tensor(Int32, (num_words,), divisibility=4),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Float32, (num_q, cand_cap)),
        make_fake_tensor(Int32, (num_q, cand_cap)),
        make_fake_tensor(Int32, (num_q,)),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        _stream(),
        options="--enable-tvm-ffi",
    )


def _pruning_scales(kv, kv_scales, knsc):
    # The pruning kernel reads scales and key norms through one symbolic length: as long as knsc.
    return unit_scales(kv)[: knsc.numel()] if kv_scales is None else padded_scales(kv_scales)


def _mask_sm100(q_prune, kv, kv_scales, knsc, w_prune, allowance, ks, ke, origin, inv_delta, th_bucket, mask):
    num_q, seq_len_kv = ks.shape[0], kv.shape[0]
    _compile_prune_sm100(kv_scales is None)(
        as_operand(kv), as_operand(q_prune).view(-1, HEAD_DIM, 1), w_prune, _pruning_scales(kv, kv_scales, knsc),
        knsc.reshape(-1), ks, ke, origin, inv_delta, th_bucket, allowance, mask.view(-1), num_q, seq_len_kv,
        mask.shape[1], prune_kv_splits(num_q, seq_len_kv, PRUNE_BLOCK_Q, 1, sm_count(kv.device.index)),
    )


def _refine_sm100(q, kv, kv_scales, knsc, weights, mask, ks, ke, origin, inv_delta, th_bucket, cand_val, cand_idx,
                  cand_cnt):
    num_q, seq_len_kv = q.shape[0], kv.shape[0]
    bf16 = kv_scales is None
    mask_blocks = mask.shape[1]
    min_splits = math.ceil(mask_blocks / refine_max_blocks(bf16))
    _compile_refine_sm100(bf16)(
        as_operand(kv), as_operand(q).view(num_q * NUM_HEADS, HEAD_DIM, 1), weights,
        _pruning_scales(kv, kv_scales, knsc), mask.view(-1), ks, ke, origin, inv_delta, th_bucket, cand_val, cand_idx,
        cand_cnt, num_q, seq_len_kv, mask_blocks,
        prune_kv_splits(num_q, seq_len_kv, REFINE_BLOCK_Q, min_splits, sm_count(q.device.index)),
    )


# ----------------------------------------------------------------------------------------------------------------
# SM90
# ----------------------------------------------------------------------------------------------------------------


@functools.cache
def _compile_scan_sm90(bf16: bool, single_split: bool, dense: bool):
    num_q, num_q_heads, seq_len_kv, num_scales, cand_cap = (cute.sym_int() for _ in range(5))
    elem = _elem(bf16)
    return cute.compile(
        ScanSm90(bf16=bf16, single_split=single_split, dense=dense),
        make_fake_tensor(elem, (seq_len_kv, HEAD_DIM), divisibility=16),
        make_fake_tensor(elem, (num_q_heads, HEAD_DIM, 1), divisibility=16, leading_dim=1),
        make_fake_tensor(Float32, (num_q, NUM_HEADS), divisibility=4),
        make_fake_tensor(Float32, (num_scales,), divisibility=4),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Float32, (num_q, cand_cap)),
        make_fake_tensor(Int32, (num_q, cand_cap)),
        make_fake_tensor(Int32, (num_q,)),
        Int32(0),
        Int32(0),
        Int32(0),
        _stream(),
        options="--enable-tvm-ffi",
    )


def _scan_sm90(q, kv, kv_scales, weights, ks, ke, origin, inv_delta, th_bucket, cand_val, cand_idx, cand_cnt,
               dense: bool, q_operand=None):
    num_q, seq_len_kv = q.shape[0], kv.shape[0]
    num_splits = scan_kv_splits(num_q, seq_len_kv, sm_count(q.device.index))
    bf16 = kv_scales is None
    scales = unit_scales(kv) if bf16 else padded_scales(kv_scales)
    _compile_scan_sm90(bf16, num_splits == 1 and not dense, dense)(
        as_operand(kv), q_operand_sm90(q) if q_operand is None else q_operand, weights, scales, ks, ke, origin,
        inv_delta, th_bucket, cand_val, cand_idx, cand_cnt, num_q, seq_len_kv, num_splits,
    )


@functools.cache
def _compile_prune_sm90(bf16: bool):
    num_q, pair_heads, seq_len_kv, num_scales, num_words = (cute.sym_int() for _ in range(5))
    elem = _elem(bf16)
    return cute.compile(
        PruneSm90(bf16=bf16),
        make_fake_tensor(elem, (seq_len_kv, HEAD_DIM), divisibility=16),
        make_fake_tensor(elem, (pair_heads, HEAD_DIM, 1), divisibility=16, leading_dim=1),
        make_fake_tensor(Float32, (num_q, PRUNE_COLS), divisibility=4),
        make_fake_tensor(Float32, (num_scales,), divisibility=4),
        make_fake_tensor(Float32, (num_scales,), divisibility=4),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Int32, (num_words,)),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        _stream(),
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_refine_sm90(bf16: bool, single_split: bool):
    num_q, num_q_heads, seq_len_kv, num_scales, num_words, cand_cap = (cute.sym_int() for _ in range(6))
    elem = _elem(bf16)
    return cute.compile(
        RefineSm90(bf16=bf16, single_split=single_split),
        make_fake_tensor(elem, (seq_len_kv, HEAD_DIM), divisibility=16),
        make_fake_tensor(elem, (num_q_heads, HEAD_DIM, 1), divisibility=16, leading_dim=1),
        make_fake_tensor(Float32, (num_q, NUM_HEADS), divisibility=4),
        make_fake_tensor(Float32, (num_scales,), divisibility=4),
        make_fake_tensor(Int32, (num_words,), divisibility=4),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Float32, (num_q,)),
        make_fake_tensor(Int32, (num_q,)),
        make_fake_tensor(Float32, (num_q, cand_cap)),
        make_fake_tensor(Int32, (num_q, cand_cap)),
        make_fake_tensor(Int32, (num_q,)),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        _stream(),
        options="--enable-tvm-ffi",
    )


def _mask_sm90(q_prune, kv, kv_scales, knsc, w_prune, allowance, ks, ke, origin, inv_delta, th_bucket, mask):
    num_q, seq_len_kv = ks.shape[0], kv.shape[0]
    _compile_prune_sm90(kv_scales is None)(
        as_operand(kv), as_operand(q_prune).view(-1, HEAD_DIM, 1), w_prune, _pruning_scales(kv, kv_scales, knsc),
        knsc.reshape(-1), ks, ke, origin, inv_delta, th_bucket, allowance, mask.view(-1), num_q, seq_len_kv,
        mask.shape[1], prune_kv_splits(num_q, seq_len_kv, PRUNE_BLOCK_Q, 1, sm_count(kv.device.index)),
    )


def _refine_sm90(q, kv, kv_scales, knsc, weights, mask, ks, ke, origin, inv_delta, th_bucket, cand_val, cand_idx,
                 cand_cnt):
    num_q, seq_len_kv = q.shape[0], kv.shape[0]
    mask_blocks = mask.shape[1]
    min_splits = math.ceil(mask_blocks / REFINE_SM90_MAX_BLOCKS)
    splits = prune_kv_splits(num_q, seq_len_kv, REFINE_BLOCK_Q, min_splits, sm_count(q.device.index))
    _compile_refine_sm90(kv_scales is None, splits == 1)(
        as_operand(kv), q_operand_sm90(q), weights, _pruning_scales(kv, kv_scales, knsc), mask.view(-1), ks, ke,
        origin, inv_delta, th_bucket, cand_val, cand_idx, cand_cnt, num_q, seq_len_kv, mask_blocks, splits,
    )


_Q_SCOPE: contextvars.ContextVar = contextvars.ContextVar("flashtopk_sm90_q_scope", default=None)


@contextlib.contextmanager
def sm90_q_scope(q: torch.Tensor):
    """Within one selection call: the SM90 kernels that take q (shared sample, scan, refine) share one Q operand,
    built at first use and released when the call ends."""
    token = _Q_SCOPE.set([q, None])
    try:
        yield
    finally:
        _Q_SCOPE.reset(token)


def q_operand_sm90(q: torch.Tensor) -> torch.Tensor:
    scope = _Q_SCOPE.get()
    if scope is None or scope[0] is not q:
        return _q_operand_sm90(q)
    if scope[1] is None:
        scope[1] = _q_operand_sm90(q)
    return scope[1]


def _q_operand_sm90(q: torch.Tensor) -> torch.Tensor:
    """q as the SM90 kernels' Q operand, [tiles * 256, 128, 1] (uint8 for FP8): each 8-row tile's rows permuted so a
    thread of the accumulator holds whole rows (scan_sm90.q_n_of); a partial last tile padded with zero rows."""
    num_q = q.shape[0]
    tiles = math.ceil(num_q / NEXT_N)
    qq = as_operand(q)
    if tiles * NEXT_N != num_q:
        qq = torch.nn.functional.pad(qq, (0, 0, 0, 0, 0, tiles * NEXT_N - num_q))
    # q_n_of as a transpose: row 2b + a, head 2c + e of a tile is operand row 128a + 8c + 2b + e.
    return _permute_rows(qq, (tiles, 4, 2, 16, 2))


def _permute_rows(x: torch.Tensor, shape) -> torch.Tensor:
    """x's 128-element rows viewed as `shape` and transposed (0, 2, 3, 1, 4), as the [-1, 128, 1] operand; the rows
    move as 8-byte words, which the strided copy moves far faster than single bytes."""
    words = x.contiguous().view(torch.int64)
    moved = words.view(*shape, words.shape[-1]).permute(0, 2, 3, 1, 4, 5).reshape(-1, words.shape[-1])
    return moved.view(x.dtype).view(-1, HEAD_DIM, 1)


# ----------------------------------------------------------------------------------------------------------------
# Either architecture
# ----------------------------------------------------------------------------------------------------------------


def scan(q, kv, kv_scales, weights, ks, ke, origin, inv_delta, th_bucket, cand_val, cand_idx, cand_cnt) -> None:
    """Append every key of row r's window [ks[r], ke[r]) whose score passes the row's raw gate (from origin,
    inv_delta and th_bucket) to its candidates as (-score, kv index). cand_cnt[r] ends as the row's hit count (slots
    past the capacity are counted, not written); with KV splits it accumulates, so it must start at zero. BF16
    operands when kv_scales is None."""
    run = _scan_sm90 if arch(q) == 9 else _scan_sm100
    run(q, kv, kv_scales, weights, ks, ke, origin, inv_delta, th_bucket, cand_val, cand_idx, cand_cnt, dense=False)


def scan_dense(q, kv, kv_scales, weights, ks, ke, out: torch.Tensor, scratch: torch.Tensor,
               q_operand: Optional[torch.Tensor] = None) -> None:
    """out[r, key] = row r's score of every key of its window [ks[r], ke[r]), in the scan's bits; other entries are
    left as they were. scratch: any [num_q] 4-byte tensor (the gate's inputs, read and unused). On SM90 q_operand,
    when given, is q's operand already laid out (q_operand_sm90)."""
    gate = scratch.view(torch.float32)
    bucket = scratch.view(torch.int32)
    run = _scan_sm90 if arch(q) == 9 else _scan_sm100
    run(q, kv, kv_scales, weights, ks, ke, gate, gate, bucket, out, out.view(torch.int32), bucket, dense=True,
        q_operand=q_operand)


def prune_scan(q, q_prune, kv, kv_scales, knsc, weights, w_prune, allowance, ks, ke, origin, inv_delta, th_bucket,
               cand_val, cand_idx, cand_cnt) -> None:
    """Key pruning in place of the scan, the same candidates: the score upper bound's mask, then the refine, which
    appends the kept keys whose full score passes the row's raw gate (counts must start at zero). q_prune is the
    planner's operand, already in the pruning kernel's layout."""
    mask = prune_mask(q_prune, kv, kv_scales, knsc, w_prune, allowance, ks, ke, origin, inv_delta, th_bucket)
    prune_refine(q, kv, kv_scales, knsc, weights, mask, ks, ke, origin, inv_delta, th_bucket, cand_val, cand_idx,
                 cand_cnt)


def mask_shape(num_q: int, seq_len_kv: int) -> Tuple[int, int, int]:
    """The pruning mask: per 8-row half of a 16-row tile and 128-key block, four words of one bit per key."""
    return 2 * math.ceil(num_q / PRUNE_BLOCK_Q), math.ceil(seq_len_kv / BLOCK_KV), 4


def prune_mask(q_prune, kv, kv_scales, knsc, w_prune, allowance, ks, ke, origin, inv_delta, th_bucket,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pruning's first kernel: the mask (mask_shape) of the keys whose score upper bound passes some row's gate in
    each 8-row half."""
    if mask is None:
        mask = torch.empty(mask_shape(ks.shape[0], kv.shape[0]), dtype=torch.int32, device=kv.device)
    run = _mask_sm90 if arch(kv) == 9 else _mask_sm100
    run(q_prune, kv, kv_scales, knsc, w_prune, allowance, ks, ke, origin, inv_delta, th_bucket, mask)
    return mask


def prune_refine(q, kv, kv_scales, knsc, weights, mask, ks, ke, origin, inv_delta, th_bucket, cand_val, cand_idx,
                 cand_cnt) -> None:
    """Pruning's second kernel: the full score of the keys the mask keeps, appended to the candidates as the scan
    would."""
    run = _refine_sm90 if arch(q) == 9 else _refine_sm100
    run(q, kv, kv_scales, knsc, weights, mask, ks, ke, origin, inv_delta, th_bucket, cand_val, cand_idx, cand_cnt)


@functools.lru_cache(maxsize=64)
def _sample_positions(seq_len_kv: int, num_samples: int, device_index: int) -> torch.Tensor:
    """The shared sample's keys p_j = floor((2j + 1) L / (2M)), int64."""
    j = torch.arange(num_samples, device=torch.device("cuda", device_index), dtype=torch.int64)
    return torch.div((2 * j + 1) * seq_len_kv, 2 * num_samples, rounding_mode="floor")


@functools.cache
def _compile_sample_windows():
    num_q = cute.sym_int()
    vec = (num_q,)
    return cute.compile(
        SampleWindows(),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(cutlass.Int64, (HEALTH_WORDS,)),
        Int32(0),
        Int32(0),
        Int32(0),
        _stream(),
        options="--enable-tvm-ffi",
    )


def sample_scores(q, kv, kv_scales, weights, ks, ke, num_samples: int, out: torch.Tensor, scratch: torch.Tensor,
                  lo: torch.Tensor, hi: torch.Tensor, health: torch.Tensor) -> None:
    """The shared sample's keys p_j = floor((2j + 1) L / (2M)) packed, each row's sample window [lo[r], hi[r]) (and
    the call's health zeroed, in the same launch), then every row's sample scores out[r, lo[r]:hi[r]] by the dense
    scan (the scan's bits, so calibration needs no seed margin)."""
    seq_len_kv = kv.shape[0]
    pos = _sample_positions(seq_len_kv, num_samples, kv.device.index)
    kv_sample = as_operand(kv).index_select(0, pos).view(kv.dtype)
    scales = None if kv_scales is None else kv_scales.index_select(0, pos)
    _compile_sample_windows()(ks, ke, lo, hi, health, q.shape[0], seq_len_kv, num_samples)
    scan_dense(q, kv_sample, scales, weights, lo, hi, out, scratch)


@functools.cache
def _compile_decide():
    num_q, est_rows = cute.sym_int(), cute.sym_int()
    vec = (num_q,)
    return cute.compile(
        Decide(),
        make_fake_tensor(Int32, (est_rows, EST_WORDS)),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, (1,)),
        make_fake_tensor(Float32, (1,)),
        Int32(0),
        Float32(0.0),
        _stream(),
        options="--enable-tvm-ffi",
    )


@functools.cache
def _estimate_scratch(device_index: int) -> torch.Tensor:
    return torch.empty(1, dtype=torch.float32, device=torch.device("cuda", device_index))


def decide(est, ks, ke, scan_ks, prune_ks, flag, threshold: float, estimate=None) -> None:
    """Adaptive pruning's choice on the device from the estimate rows' sample masks (plan.estimate_shares): flag[0]
    = (group estimate <= threshold), and the windows for the scan (scan_ks: ke when pruning, else ks) and for the
    pruning kernels (prune_ks: the other way round); estimate[0], when given, receives the group estimate."""
    if estimate is None:
        estimate = _estimate_scratch(ks.device.index)
    _compile_decide()(est, ks, ke, scan_ks, prune_ks, flag, estimate, ks.shape[0], threshold)


@functools.cache
def _compile_seed_prep():
    num_q, width = cute.sym_int(), cute.sym_int()
    vec = (num_q,)
    return cute.compile(
        SeedPrep(),
        make_fake_tensor(Float32, (num_q, width)),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Float32, vec),
        make_fake_tensor(Float32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        Int32(0),
        Int32(0),
        Float32(0.0),
        Int32(0),
        _stream(),
        options="--enable-tvm-ffi",
    )


def seed_prep(sample, ks, ke, col_lo, col_hi, topk: int, slack_topk: float, k_min: int, origin, inv_delta, th_bucket,
              cand_cnt) -> None:
    """Each row's bucket map (origin, inv_delta) and seed gate (th_bucket) from its samples sample[r, col_lo[r]:
    col_hi[r]], aiming at the topk-th best key of a row read whole and at about slack_topk candidates otherwise (at
    least the k_min-th best sample); cand_cnt is zeroed."""
    _compile_seed_prep()(sample, ks, ke, col_lo, col_hi, origin, inv_delta, th_bucket, cand_cnt, sample.shape[0],
                         topk, slack_topk, k_min)


@functools.cache
def _compile_select(topk: int, source: int):
    rows, cap, slots, out_rows = (cute.sym_int() for _ in range(4))
    vec = (rows,)
    return cute.compile(
        Select(topk, source),
        make_fake_tensor(Float32, (rows, cap)),
        make_fake_tensor(Int32, (rows, cap)),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, (slots,)),
        make_fake_tensor(Float32, (out_rows, topk)),
        make_fake_tensor(Int32, (out_rows, topk)),
        make_fake_tensor(cutlass.Int64, (HEALTH_WORDS,)),
        make_fake_tensor(Int32, (RECOVERY_ROWS,)),
        Int32(0),
        Int32(0),
        Int32(0),
        _stream(),
        options="--enable-tvm-ffi",
    )


@functools.cache
def _unused_health(device_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda", device_index)
    return (torch.zeros(HEALTH_WORDS, dtype=torch.int64, device=device),
            torch.zeros(RECOVERY_ROWS, dtype=torch.int32, device=device))


def select(cand_val, cand_idx, cand_cnt, ks, ke, scores, indices, health, bad_rows, stats: bool = False) -> None:
    """Each row's exact top-k of its candidates into scores / indices in canonical order (score descending, index
    ascending; -inf / -1 padding last), accumulating the call's health and recording its first bad rows."""
    _compile_select(scores.shape[1], CANDIDATES)(cand_val, cand_idx, cand_cnt, ks, ke, cand_cnt, scores, indices,
                                                 health, bad_rows, cand_val.shape[0], 1, int(stats))


def select_dense(dense, ks, ke, row_map, scores, indices, splits: int = 1) -> None:
    """Recovery: slot s's exact top-k over its window [ks[s], ke[s]) of dense[s] (the index is the column), in
    canonical order, to row row_map[s] of scores / indices; slots mapped to -1 are skipped. With splits > 1 each slot
    is cut into that many slices, and slice p of slot s writes its own top-k to row s * splits + p of the stage
    buffers scores / indices instead (select_staged finishes)."""
    health, bad_rows = _unused_health(dense.device.index)
    _compile_select(scores.shape[1], DENSE)(dense, dense.view(torch.int32), ks, ks, ke, row_map, scores, indices,
                                            health, bad_rows, dense.shape[0] * splits, splits, 0)


def select_staged(stage_scores, stage_indices, row_map, scores, indices) -> None:
    """Recovery's second stage: slot s's exact top-k over its slices' selections (stage rows s * splits ... s * splits
    + splits - 1), to row row_map[s] of scores / indices."""
    health, bad_rows = _unused_health(stage_scores.device.index)
    slots = row_map.shape[0]
    _compile_select(scores.shape[1], STAGED)(stage_scores.view(slots, -1), stage_indices.view(slots, -1), row_map,
                                             row_map, row_map, row_map, scores, indices, health, bad_rows, slots, 1, 0)


@functools.cache
def _compile_triage(words_per_head: int, sm90: bool):
    num_q, slots_flat = cute.sym_int(), cute.sym_int()
    vec = (num_q,)
    slots = (RECOVERY_ROWS,)
    return cute.compile(
        Triage(words_per_head, sm90),
        make_fake_tensor(cutlass.Int64, (HEALTH_WORDS,)),
        make_fake_tensor(Int32, slots),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, (num_q, NUM_HEADS * words_per_head)),
        make_fake_tensor(Float32, (num_q, NUM_HEADS)),
        make_fake_tensor(Int32, slots),
        make_fake_tensor(Int32, slots),
        make_fake_tensor(Int32, slots),
        make_fake_tensor(Int32, (slots_flat,)),
        make_fake_tensor(Float32, (RECOVERY_ROWS, NUM_HEADS)),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        make_fake_tensor(Int32, vec),
        Int32(0),
        _stream(),
        options="--enable-tvm-ffi",
    )


def triage(health, bad_rows, ks, ke, q, weights, row_map, slot_ks, slot_ke, q_gather, w_gather, whole_map, whole_ks,
           whole_ke) -> None:
    """Recovery's plan from the health (triage.py): per-row slots (map, windows, gathered q and weights; q_gather
    holds the SM90 Q operand on SM90) and whole-call windows, each empty unless its recovery applies."""
    num_q = q.shape[0]
    words = q.contiguous().view(num_q, -1).view(torch.int32)
    _compile_triage(words.shape[1] // NUM_HEADS, arch(q) == 9)(
        health, bad_rows, ks, ke, words, weights, row_map, slot_ks, slot_ke, q_gather, w_gather, whole_map, whole_ks,
        whole_ke, num_q)


def warmup(arch_major: int, bf16: bool, topk: int, words_per_head: int) -> None:
    """Compile every kernel variant a selection on this architecture and dtype can take (each once per process)."""
    if arch_major == 9:
        for single_split in (True, False):
            _compile_scan_sm90(bf16, single_split, False)
            _compile_refine_sm90(bf16, single_split)
        _compile_scan_sm90(bf16, False, True)
        _compile_prune_sm90(bf16)
    else:
        for single_split in (True, False):
            _compile_scan_sm100(single_split, bf16, False)
        _compile_scan_sm100(False, bf16, True)
        _compile_prune_sm100(bf16)
        _compile_refine_sm100(bf16)
    _compile_sample_windows()
    _compile_decide()
    _compile_seed_prep()
    for source in (CANDIDATES, DENSE, STAGED):
        _compile_select(topk, source)
    _compile_triage(words_per_head, arch_major == 9)


def supported(q: torch.Tensor, kv: Optional[torch.Tensor] = None) -> bool:
    """The kernels' shapes: [num_q, 32, 128] queries, 128-wide keys, FP8 or BF16, on SM90 or SM100."""
    if not q.is_cuda or q.dim() != 3 or q.shape[1] != NUM_HEADS or q.shape[2] != HEAD_DIM:
        return False
    if q.dtype not in (torch.float8_e4m3fn, torch.bfloat16) or arch(q) not in (9, 10):
        return False
    return kv is None or (kv.dim() == 2 and kv.shape[1] == HEAD_DIM and kv.dtype == q.dtype)
