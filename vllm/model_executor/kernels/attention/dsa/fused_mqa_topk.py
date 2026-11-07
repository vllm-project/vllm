# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused FP8 MQA-logits + top-k prefill selection (LiteTopK, SM100 family)"""

import functools

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Current restrictions of the fused_mqa_topk kernel:
FUSED_MQA_TOPK_HEADS = (32, 64)
FUSED_MQA_TOPK_HEAD_DIM = 128
FUSED_MQA_TOPK_TOPKS = (512, 1024, 2048)
# Most gathered keys (all requests of a chunk) one call accepts (kMaxKeys in
# csrc/fused_mqa_topk/fused_mqa_topk.cu). Chunks with more keys take the dense
# path; the planner never packs past it.
FUSED_MQA_TOPK_MAX_KEYS = 1 << 20
# Rows per fused_mqa_topk call. The kernel has no row limit, but prefill chunks above
# 16384 rows (--max-num-batched-tokens > 16384) are not validated end to end yet,
# so the chunk planner splits fused_mqa_topk chunks at this many rows.
FUSED_MQA_TOPK_MAX_ROWS = 16384
# Rows per op call: fused_mqa_topk_prefill runs windows of at most this many rows
# over one set of workspaces, which stay below the 512 MiB dense prefill logits
# budget the profile run reserves at the default VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
# (466.6 MiB on 148 SMs).
FUSED_MQA_TOPK_WINDOW_ROWS = 4096


def fused_mqa_topk_min_keys(topk: int, heads: int) -> int:
    """Longest-row key count from which fused_mqa_topk is at least 3% faster
    than the dense path on SM100f at 1024 to 16384 rows on kernel time and
    CUDA events, and not slower on GPU span from the call start: 20K, 24K and
    28K keys for top-512, 1024 and 2048 at 32 heads, measured against FP32
    dense logits (BF16 logits are slower at 32 heads), and 4K more at 64 heads,
    measured against BF16 dense logits."""
    return 16384 + 4096 * (topk.bit_length() - 10 + heads // 32)


# Mirrors csrc/fused_mqa_topk (the op checks the workspace shapes).
_CODES_PER_CTA = 32768  # int32 threshold-sample codes per producer CTA
_PIECES_PER_BLOCK = 32  # most scan pieces of a block (the producer's last wave)
_RECOVERY_SLOTS = 32  # most recovery slots, one block's dense scores each
_RECOVERY_FLOATS = 1 << 24  # recovery pool bound (64 MiB of fp32 scores)
# Candidate slab width, the op's largest (8 bytes per entry: 96 KiB per row). A
# row with more candidates is recovered exactly but slowly, so the width trades
# memory for speed only: 1.3x the most candidates per row measured at top-2048
# on GLM-5.2, DeepSeek-V4 and random indexer inputs (1.8x at top-1024, 2.2x at
# top-512).
_CANDIDATES_PER_ROW = 12288


@functools.cache
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


@functools.cache
def fused_mqa_topk_available() -> bool:
    """True when the fused_mqa_topk op is compiled in and the GPU is SM100-family."""
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and hasattr(torch.ops._C, "fused_mqa_topk_prefill")
    ):
        return False
    schema = torch.ops._C.fused_mqa_topk_prefill.default._schema
    if "row_state" not in (arg.name for arg in schema.arguments):
        logger.warning(
            "vllm._C holds an older fused_mqa_topk_prefill (%s); the sparse "
            "indexer prefill takes the dense path. Rebuild the extension.",
            schema,
        )
        return False
    return True


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

    query_shape: one token's indexer queries, (heads, head_dim) with heads 32
    or 64 and head_dim 128;
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
        and len(query_shape) == 2
        and query_shape[0] in FUSED_MQA_TOPK_HEADS
        and query_shape[1] == FUSED_MQA_TOPK_HEAD_DIM
        and topk in FUSED_MQA_TOPK_TOPKS
        and fused_mqa_topk_available()
    )


def fused_mqa_topk_chunk(
    max_seqlen_k: int, keys: int, topk: int, rows: int, *, heads: int
) -> bool:
    """Whether a prefill chunk of a fused_mqa_topk-eligible layer takes the
    fused_mqa_topk kernels: its gathered keys fit the kernel and its longest
    row sees at least fused_mqa_topk_min_keys keys in chunks of at least 1024
    rows, or at least 64K keys in chunks of fewer than 1024 rows, where a
    call's fixed cost needs longer rows."""
    min_keys = fused_mqa_topk_min_keys(topk, heads) if rows >= 1024 else 1 << 16
    return max_seqlen_k >= min_keys and keys <= FUSED_MQA_TOPK_MAX_KEYS


def get_fused_mqa_topk_workspaces(
    device: torch.device,
    rows: int,
    heads: int = 64,
    keys: int = FUSED_MQA_TOPK_MAX_KEYS,
) -> list[torch.Tensor]:
    """The workspaces of a fused_mqa_topk_prefill call over ``rows`` query rows,
    in the op's argument order (see fused_mqa_topk.cu), all fresh: the zeroed
    block counters, then the scratch. They are sized for one op call over at
    most FUSED_MQA_TOPK_WINDOW_ROWS of the rows. The defaults give the largest
    workspaces of a call over at most ``rows`` rows."""
    index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    rows = min(rows, FUSED_MQA_TOPK_WINDOW_ROWS)
    blocks = -(-rows // (4 if heads > 32 else 8))  # 256 row x head columns each
    # At most one producer CTA per SM and 32 pieces per block.
    ctas = min(_sm_count(index), _PIECES_PER_BLOCK * blocks)
    # The recovery rescores each block with a bad row over all keys in a slot
    # of its pool: at most 32 slots and 64 MiB, at least one slot. Sized in
    # slots of 8 rows (a block at 32 heads, two at 64) for either head count.
    keys = max(keys, 1)
    slots = min(-(-rows // 8), _RECOVERY_SLOTS, _RECOVERY_FLOATS // (8 * keys))
    empty = functools.partial(torch.empty, device=device)
    return [
        torch.zeros(2 * blocks, dtype=torch.int32, device=device),  # block counters
        empty(ctas * _CODES_PER_CTA, dtype=torch.int32),  # threshold-sample codes
        empty((rows, _CANDIDATES_PER_ROW), dtype=torch.float32),  # candidate scores
        empty((rows, _CANDIDATES_PER_ROW), dtype=torch.int32),  # their keys
        empty((3, rows), dtype=torch.int32),  # candidate counts, thresholds, overflow
        empty(max(1, slots) * 8 * keys, dtype=torch.float32),  # recovery
    ]


def fused_mqa_topk_key_scales(scales: torch.Tensor) -> torch.Tensor:
    """Per-key fp32 scales from the gathered indexer K-cache scales.

    scales: [keys, 4] uint8. SM100 caches store each power-of-two scale as four
    equal UE8M0 bytes; the fp32 value 2**(e - 127) has bit pattern ``e << 23``.
    """
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
    """Fused replacement of ``fp8_fp4_mqa_logits`` + ``top_k_per_row_prefill``
    (LiteTopK sample -> filter -> select): per block of rows, the candidate
    producer scores a strided sample of the keys, takes a threshold per row
    from it and keeps the keys that score at least that; the selector takes
    each row's exact top-k of its candidates; rows whose candidates overflowed
    or fell short of the top-k are rescored densely and selected exactly. All
    on the GPU: no logits matrix, no host synchronization.

    q: [rows, heads, 128] fp8 e4m3 with heads 32 or 64; k: [keys, 128] fp8 e4m3
    (keys <= FUSED_MQA_TOPK_MAX_KEYS); scales: [keys] fp32; weights: [rows,
    heads] fp32; starts/ends: [rows] int32, the visible key range ``[start,
    end)`` of each row; output: [rows, k] int32 with k (the top-k) 512, 1024 or
    2048. All contiguous; q, k, scales, weights and output 16-byte aligned (the
    op checks the contract).

    Writes the same convention as ``ops.top_k_per_row_prefill``: indices
    relative to the row's ``start`` (i.e. ``key - start``), unordered, ``-1``
    padded when fewer than k keys are visible. Exact on the kernels' fp32
    scores up to ties at the k-th score. The thresholds are fp16 distances to a
    sampled score of each row (a common offset of the scores does not matter);
    rows whose top scores spread over more than 65504 (or are packed within
    2^-24) may overflow and take the recovery: exact, but seconds at 1M keys.
    """
    rows, heads = q.shape[0], q.shape[1]
    logger.info_once(
        "Using the fused LiteTopK MQA+top-k prefill kernels (FP8, %d heads, top-k %d).",
        heads,
        output.shape[1],
    )
    workspaces = get_fused_mqa_topk_workspaces(q.device, rows, heads, k.shape[0])
    if rows <= FUSED_MQA_TOPK_WINDOW_ROWS:
        # One window: the workspaces are sized for exactly these rows.
        torch.ops._C.fused_mqa_topk_prefill(
            q, k, scales, weights, starts, ends, output, *workspaces
        )
        return
    counter, codes, values, indices, row_state, recovery = workspaces
    # Op calls over row windows reuse the workspaces in stream order (each call
    # leaves the counters zeroed).
    for start in range(0, rows, FUSED_MQA_TOPK_WINDOW_ROWS):
        n = min(FUSED_MQA_TOPK_WINDOW_ROWS, rows - start)
        window = slice(start, start + n)
        torch.ops._C.fused_mqa_topk_prefill(
            q[window],
            k,
            scales,
            weights[window],
            starts[window],
            ends[window],
            output[window],
            counter,
            codes,
            values[:n],
            indices[:n],
            row_state.view(-1)[: 3 * n].view(3, n),
            recovery,
        )
