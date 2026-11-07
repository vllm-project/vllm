# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused FP8 MQA-logits + top-k prefill selection.

SM100 family: the single-launch ``torch.ops._C.fused_mqa_topk_prefill``.
SM90: FlashTopK (https://github.com/Tencent-Hunyuan/FlashTopK), when the optional
``flashtopk`` package is installed.
"""

import functools
from types import ModuleType

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

logger = init_logger(__name__)

# Current restrictions of the fused_mqa_topk kernel:
FUSED_MQA_TOPK_HEADS = 32
FUSED_MQA_TOPK_HEAD_DIM = 128
FUSED_MQA_TOPK_TOPKS = (512, 1024, 2048)
# Most gathered keys (all requests of a chunk) one call accepts: candidates
# store the key index in 20 bits (kMaxKeys in csrc/fused_mqa_topk/constants.h).
# Chunks with more keys take the dense path; the planner never packs past it.
FUSED_MQA_TOPK_MAX_KEYS = 1 << 20
# Rows per fused_mqa_topk call. The kernel has no row limit, but prefill chunks above
# 16384 rows (--max-num-batched-tokens > 16384) are not validated end to end yet,
# so the chunk planner splits fused_mqa_topk chunks at this many rows.
FUSED_MQA_TOPK_MAX_ROWS = 16384


# SM90 (FlashTopK, measured on H200 with random and GLM-5.1 indexer inputs):
# fused from 64K keys in the longest row for chunks of at least 4096 rows, from
# 256K keys for smaller chunks, whose fixed per-call work dense outruns below.
# A call works on FLASHTOPK_CHUNK_ROWS rows at a time, which bounds its
# workspace (larger slices were no faster).
FLASHTOPK_MIN_KEYS = 1 << 16
FLASHTOPK_MIN_KEYS_FEW_ROWS = 1 << 18
FLASHTOPK_CHUNK_ROWS = 4096


def fused_mqa_topk_min_keys(topk: int) -> int:
    """Longest-row key count from which fused_mqa_topk beats the dense path
    (shorter chunks stay dense): on SM100f 32K, or 64K for top-2048; on SM90
    FLASHTOPK_MIN_KEYS (FLASHTOPK_MIN_KEYS_FEW_ROWS below 4096 rows)."""
    if _flashtopk() is not None:
        return FLASHTOPK_MIN_KEYS
    return 1 << 16 if topk > 1024 else 1 << 15


# Mirrors csrc/fused_mqa_topk/constants.h (the op checks the workspace shapes).
_CANDIDATES_PER_ROW = 16384  # kCandidatesPerRow
_RETAINED_PER_ROW = 4096  # kRetainedPerRow
_SLOTS_PER_SM = 2  # candidate slots per persistent CTA
_BLOCK_ROWS = 8  # kBlockRows: query rows per block, one block per CTA at a time


@functools.cache
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


@functools.cache
def _flashtopk() -> ModuleType | None:
    """The FlashTopK package on SM90, else None (not installed, or another GPU)."""
    if not (current_platform.is_cuda() and current_platform.is_device_capability(90)):
        return None
    try:
        import flashtopk
    except ImportError:
        return None
    return flashtopk


@functools.cache
def fused_mqa_topk_available() -> bool:
    """True when a fused kernel serves this GPU: the compiled op on SM100-family,
    or FlashTopK on SM90."""
    sm100 = (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and hasattr(torch.ops._C, "fused_mqa_topk_prefill")
    )
    return sm100 or _flashtopk() is not None


def use_fused_mqa_topk(
    *,
    query_shape: tuple[int, ...],
    topk: int,
    fp4_cache: bool,
    full_context: bool,
    candidate_selection: bool = False,
) -> bool:
    """Whether an indexer may use the fused_mqa_topk kernel: the single
    eligibility rule of the prefill chunk planner and the indexer layers (each
    prefill chunk then also needs fused_mqa_topk_chunk).

    query_shape: one token's indexer queries, (heads, head_dim) = (32, 128);
    topk: the per-row top-k (pools for kpool indexers) in {512, 1024, 2048};
    fp4_cache: the MXFP4 indexer cache (and its FP4 queries) stays dense;
    full_context: every rank ranks its rows over the whole context (no DCP, or
    PCP after the KV all-gather); pure DCP needs the per-shard top-k merge;
    candidate_selection: v4.1 two-level selection needs the logits.
    VLLM_BATCH_INVARIANT keeps the dense path: the fused gate depends on the
    chunk's total keys, and ties at the k-th score are taken in atomic order.
    """
    return (
        envs.VLLM_USE_FUSED_MQA_TOPK
        and not envs.VLLM_BATCH_INVARIANT
        and not fp4_cache
        and full_context
        and not candidate_selection
        and tuple(query_shape) == (FUSED_MQA_TOPK_HEADS, FUSED_MQA_TOPK_HEAD_DIM)
        and topk in FUSED_MQA_TOPK_TOPKS
        and fused_mqa_topk_available()
    )


def fused_mqa_topk_chunk(
    max_seqlen_k: int, keys: int, topk: int, rows: int, dense_rows: int
) -> bool:
    """Whether a prefill chunk of a fused_mqa_topk-eligible layer takes the
    fused_mqa_topk kernel: its longest row sees at least fused_mqa_topk_min_keys
    keys (the dense path is as fast below), its gathered keys fit the kernel,
    and its rows keep the SMs busy.

    Each SM scans every key of one 8-row block at a time, so the blocks run in
    waves of one per SM. A last wave that fills less than half the SMs costs a
    whole wave and makes the kernel slower than dense, unless dense needs more
    than one logits call (rows > dense_rows, its rows per call).

    SM90 (FlashTopK) depends on the rows and keys only (see FLASHTOPK_MIN_KEYS).
    """
    if _flashtopk() is not None:
        few_rows = rows < FLASHTOPK_CHUNK_ROWS
        min_keys = FLASHTOPK_MIN_KEYS_FEW_ROWS if few_rows else FLASHTOPK_MIN_KEYS
        return max_seqlen_k >= min_keys and keys <= FUSED_MQA_TOPK_MAX_KEYS
    sms = _sm_count(torch.accelerator.current_device_index())
    last_wave = (-(-rows // _BLOCK_ROWS) - 1) % sms + 1  # blocks, 1..sms
    return (
        max_seqlen_k >= fused_mqa_topk_min_keys(topk)
        and keys <= FUSED_MQA_TOPK_MAX_KEYS
        and (rows > dense_rows or 2 * last_wave >= sms)
    )


def reserve_fused_mqa_topk(
    device: torch.device, rows: int, keys: int, topk: int
) -> None:
    """Allocate, then free, the workspaces of the largest fused_mqa_topk call
    over ``rows`` rows and ``keys`` keys, so the profiling run accounts for
    them (they overlap the dense logits). On SM90 this also compiles
    FlashTopK's kernels."""
    if (flashtopk := _flashtopk()) is not None:
        _flashtopk_selector(flashtopk, topk, device).plan(rows, keys, warmup=True)
    else:
        get_fused_mqa_topk_workspaces(device, rows)


def _flashtopk_selector(flashtopk: ModuleType, topk: int, device: torch.device):
    return flashtopk.Selector(
        topk, dtype=torch.float8_e4m3fn, device=device, chunk_size=FLASHTOPK_CHUNK_ROWS
    )


def get_fused_mqa_topk_workspaces(
    device: torch.device, rows: int
) -> list[torch.Tensor]:
    """The workspaces of one fused_mqa_topk call over ``rows`` query rows, in the
    op's argument order (see fused_mqa_topk.cu), all fresh: the zeroed block
    counter, then the scratch."""
    index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    blocks = -(-rows // _BLOCK_ROWS)
    # One persistent CTA per SM, at most one per block; S = 2 slots per CTA.
    slot_rows = _BLOCK_ROWS * _SLOTS_PER_SM * min(blocks, _sm_count(index))
    empty = functools.partial(torch.empty, device=device)
    return [
        torch.zeros(2, dtype=torch.int32, device=device),  # block counter
        empty((slot_rows, _CANDIDATES_PER_ROW), dtype=torch.uint16),  # fp24 lows
        empty((slot_rows, _CANDIDATES_PER_ROW), dtype=torch.int32),  # indices
        empty((slot_rows, _RETAINED_PER_ROW), dtype=torch.float32),  # retained
        empty((slot_rows, _RETAINED_PER_ROW), dtype=torch.int32),  # their indices
        empty(blocks * _BLOCK_ROWS, dtype=torch.int32),  # per-row candidate counts
        empty(blocks, dtype=torch.int32),  # blocks queued for the exact rescan
    ]


def fused_mqa_topk_key_scales(scales: torch.Tensor) -> torch.Tensor:
    """Per-key fp32 scales from the gathered indexer K-cache scales.

    scales: [keys, 4] uint8. SM100 caches store each power-of-two scale as four
    equal UE8M0 bytes; the fp32 value 2**(e - 127) has bit pattern ``e << 23``.
    Other caches store the fp32 scale itself.
    """
    if not mqa_logits_use_mx_scales():
        return scales.view(torch.float32).squeeze(-1)
    exponents = scales.view(torch.int32).squeeze(-1) & 0xFF
    return (exponents << 23).view(torch.float32)


def fused_mqa_topk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Fused replacement of ``fp8_fp4_mqa_logits`` + ``top_k_per_row_prefill``.

    q: [rows, 32, 128] fp8 e4m3; k: [keys, 128] fp8 e4m3 (keys <=
    FUSED_MQA_TOPK_MAX_KEYS); scales: [keys] fp32; weights: [rows, 32] fp32;
    starts/ends: [rows] int32, the visible key range ``[start, end)`` of each
    row; output: [rows, k] int32 with k (the top-k) 512, 1024 or 2048. All
    contiguous; q, k, scales, weights and output 16-byte aligned (the op checks
    the contract).

    Writes the same convention as ``ops.top_k_per_row_prefill``: indices
    relative to the row's ``start`` (i.e. ``key - start``), unordered, ``-1``
    padded when fewer than k keys are visible.
    """
    if (flashtopk := _flashtopk()) is not None:
        _flashtopk_prefill(flashtopk, q, k, scales, weights, starts, ends, output)
        return
    logger.info_once(
        "Using the single-launch fused MQA+top-k prefill kernel "
        "(FP8, 32 heads, top-k %d).",
        output.shape[1],
    )
    torch.ops._C.fused_mqa_topk_prefill(
        q,
        k,
        scales,
        weights,
        starts,
        ends,
        output,
        *get_fused_mqa_topk_workspaces(q.device, q.shape[0]),
    )


def _flashtopk_prefill(
    flashtopk: ModuleType,
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """fused_mqa_topk_prefill on SM90: FlashTopK's exact top-k of the same
    scores, ``scales[key] * sum_h weights[row, h] * relu(q[row, h] . k[key])``.

    FlashTopK writes absolute key indices (``-1`` past a short window); they are
    made relative to the row's start in place.
    """
    if not output.is_contiguous():
        raise RuntimeError("fused_mqa_topk_prefill: output must be contiguous")
    logger.info_once(
        "Using FlashTopK for the fused MQA+top-k prefill (SM90, FP8, top-k %d).",
        output.shape[1],
    )
    _flashtopk_selector(flashtopk, output.shape[1], q.device).run(
        q, k, weights, starts, ends, scales, out=output
    )
    output.sub_(starts[:, None]).clamp_(min=-1)
