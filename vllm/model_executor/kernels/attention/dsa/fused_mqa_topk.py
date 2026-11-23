# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused FP8 MQA-logits + top-k prefill selection (LiteTopK, SM100 family)"""

import functools

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import get_num_sms, has_deep_gemm

logger = init_logger(__name__)

# Current restrictions of the fused_mqa_topk kernel:
FUSED_MQA_TOPK_HEADS = (32, 64)
FUSED_MQA_TOPK_MAX_HEADS = max(FUSED_MQA_TOPK_HEADS)
FUSED_MQA_TOPK_HEAD_DIM = 128
FUSED_MQA_TOPK_TOPKS = (512, 1024, 2048)
FUSED_MQA_TOPK_MAX_TOPK = 2048
# Most gathered keys (all requests of a chunk) one call accepts (kMaxKeys in
# csrc/attention/dsa/fused_mqa_topk/fused_mqa_topk.cu): the widest key span the
# strided threshold sample covers (32 sample tiles x 256 keys x stride 257).
# Chunks with more keys take the dense path; the planner never packs past it.
FUSED_MQA_TOPK_MAX_KEYS = 32 * 256 * 257
# Rows per fused_mqa_topk call. The kernel has no row limit, but prefill chunks above
# 16384 rows (--max-num-batched-tokens > 16384) are not validated end to end yet,
# so the chunk planner splits fused_mqa_topk chunks at this many rows.
FUSED_MQA_TOPK_MAX_ROWS = 16384
# Rows per op call: fused_mqa_topk_prefill runs windows of at most this many rows
# over one set of workspaces, which stay below the 512 MiB dense prefill logits
# budget the profile run reserves at the default VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
# (466.8 MiB on 148 SMs).
FUSED_MQA_TOPK_WINDOW_ROWS = 4096


def is_fused_mqa_topk_k_supported(topk: int) -> bool:
    """Whether the fp32 candidate selector accepts `topk` (any multiple of 4 in
    [4, 2048]; 512 / 1024 / 2048 use specialized instantiations, other values
    such as top-k minus forced tokens use the runtime-k instantiation)."""
    return topk % 4 == 0 and 4 <= topk <= FUSED_MQA_TOPK_MAX_TOPK


def _padded_heads(heads: int) -> int:
    """Native kernel head width (32 or 64) for `1 <= heads <= 64`."""
    return 32 if heads <= 32 else 64


def _pad_heads(
    q: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad head counts below a native width (32 or 64).

    Zero heads contribute nothing to `sum_h w_h relu(q_h . k)`.
    """
    heads = q.shape[1]
    if heads <= 0 or heads > FUSED_MQA_TOPK_MAX_HEADS:
        return q, weights
    width = _padded_heads(heads)
    if width == heads:
        return q, weights
    pad = width - heads
    return (
        F.pad(q.view(torch.uint8), (0, 0, 0, pad)).view(q.dtype),
        F.pad(weights, (0, pad)),
    )


def _forced_columns(
    *,
    lengths: torch.Tensor,
    num_init_tokens: int,
    num_forced_tokens: int,
) -> torch.Tensor:
    """`[rows, num_forced_tokens]` int32 of request-relative forced keys:
    `[0, init)` and `[len - local, len)` for rows of at least `num_forced_tokens`
    keys; shorter rows list `[0, len)` and pad with `-1`."""
    cols = torch.arange(num_forced_tokens, dtype=torch.int32, device=lengths.device)
    lens = lengths[:, None]
    short = lens < num_forced_tokens
    key = torch.where(
        short | (cols < num_init_tokens), cols, lens - num_forced_tokens + cols
    )
    return torch.where(short & (cols >= lens), -1, key)


def fused_mqa_topk_min_keys(topk: int, heads: int) -> int:
    """Longest-row key count from which fused_mqa_topk is at least 3% faster
    than the dense path on SM100f at 1024 to 16384 rows on kernel time and
    CUDA events, and not slower on GPU span from the call start: 20K, 24K and
    28K keys for top-512, 1024 and 2048 at 32 heads (or fewer, zero-padded to
    32), measured against FP32 dense logits (BF16 logits are slower at 32
    heads), and 4K more at 64 heads (33..64), measured against BF16 dense
    logits."""
    tier = (max(topk, 512) - 1).bit_length() - 9
    return 16384 + 4096 * (tier + _padded_heads(heads) // 32)


# Mirrors csrc/attention/dsa/fused_mqa_topk (the op checks the workspace shapes).
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


def fused_mqa_topk_num_ctas() -> int:
    """Live SM budget for the persistent producer grid.

    Follows DeepGEMM's SM budget when available so dual-batch overlap (which
    reserves communication SMs via `deep_gemm.set_num_sms`) does not starve
    co-resident communication kernels; returns 0 (all SMs) otherwise.
    """
    if has_deep_gemm():
        return get_num_sms()
    return 0


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
    arg_names = {arg.name for arg in schema.arguments}
    if not {"row_state", "num_ctas"} <= arg_names:
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
    num_forced_tokens: int = 0,
) -> bool:
    """Whether an indexer may use the fused_mqa_topk kernel: the single
    eligibility rule of the prefill chunk planner and the indexer layers (each
    prefill chunk then also needs fused_mqa_topk_chunk).

    query_shape: one token's indexer queries, (heads, head_dim) with
    1 <= heads <= 64 (zero-padded to 32 or 64) and head_dim == 128;
    topk: the per-row top-k (pools for kpool indexers), multiple of 4 in
    [4, 2048];
    num_forced_tokens: forced init + local keys kept outside the selection
    (`topk - num_forced_tokens` must also be a supported top-k);
    fp4_cache: the MXFP4 indexer cache (and its FP4 queries) stays dense;
    full_context: every rank ranks its rows over the whole context (no DCP, or
    PCP / DCP ShardQ after the KV all-gather); pure DCP needs the per-shard
    top-k merge;
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
        and 0 < query_shape[0] <= FUSED_MQA_TOPK_MAX_HEADS
        and query_shape[1] == FUSED_MQA_TOPK_HEAD_DIM
        and num_forced_tokens >= 0
        and is_fused_mqa_topk_k_supported(topk)
        and (
            num_forced_tokens == 0
            or is_fused_mqa_topk_k_supported(topk - num_forced_tokens)
        )
        and fused_mqa_topk_available()
    )


def use_fused_mqa_topk_for_config(
    cfg: object | None,
    *,
    fp4_cache: bool,
    full_context: bool,
    candidate_selection: bool = False,
) -> bool:
    """Config-level `use_fused_mqa_topk` check for prefill chunk planners.

    Reads the top-level DSA attributes (`index_n_heads`, `index_head_dim`,
    `index_topk`, `index_kpool`, `index_init_tokens`, `index_local_tokens`)
    with fallback to `dsa_config` (Kimi K3.1).
    """
    if cfg is None:
        return False
    dsa_cfg = getattr(cfg, "dsa_config", None)
    n_heads = getattr(cfg, "index_n_heads", None) or getattr(
        dsa_cfg, "index_num_heads", 0
    )
    head_dim = getattr(cfg, "index_head_dim", None) or getattr(
        dsa_cfg, "index_head_dim", 0
    )
    topk = getattr(cfg, "index_topk", None) or getattr(dsa_cfg, "index_topk", 0)
    kpool = getattr(cfg, "index_kpool", None) or 1
    num_forced = (getattr(cfg, "index_init_tokens", None) or 0) + (
        getattr(cfg, "index_local_tokens", None) or 0
    )
    return use_fused_mqa_topk(
        query_shape=(n_heads, head_dim),
        topk=topk // kpool,
        fp4_cache=fp4_cache,
        full_context=full_context,
        candidate_selection=candidate_selection,
        num_forced_tokens=num_forced,
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
    num_ctas: int = 0,
) -> list[torch.Tensor]:
    """The workspaces of a fused_mqa_topk_prefill call over `rows` query rows,
    in the op's argument order (see fused_mqa_topk.cu), all fresh: the zeroed
    block counters, then the scratch. They are sized for one op call over at
    most FUSED_MQA_TOPK_WINDOW_ROWS of the rows. The defaults give the largest
    workspaces of a call over at most `rows` rows."""
    index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    rows = min(rows, FUSED_MQA_TOPK_WINDOW_ROWS)
    padded_heads = _padded_heads(heads)
    blocks = -(-rows // (4 if padded_heads > 32 else 8))  # 256 row x head cols
    dev_sms = _sm_count(index)
    sms = min(num_ctas, dev_sms) if num_ctas > 0 else dev_sms
    # At most one producer CTA per SM (capped by num_ctas) and 32 pieces/block.
    ctas = min(sms, _PIECES_PER_BLOCK * blocks)
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
    equal UE8M0 bytes; the fp32 value 2**(e - 127) has bit pattern `e << 23`.
    """
    exponents = scales.view(torch.int32).squeeze(-1) & 0xFF
    return (exponents << 23).view(torch.float32)


def _run_fused_mqa_topk_windows(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    output: torch.Tensor,
    num_ctas: int,
) -> None:
    rows, heads = q.shape[0], q.shape[1]
    workspaces = get_fused_mqa_topk_workspaces(
        q.device, rows, heads, k.shape[0], num_ctas=num_ctas
    )
    if rows <= FUSED_MQA_TOPK_WINDOW_ROWS:
        q_pad, w_pad = _pad_heads(q, weights)
        torch.ops._C.fused_mqa_topk_prefill(
            q_pad, k, scales, w_pad, starts, ends, output, *workspaces, num_ctas
        )
        return
    counter, codes, values, indices, row_state, recovery = workspaces
    # Op calls over row windows reuse the workspaces in stream order (each call
    # leaves the counters zeroed).
    for start in range(0, rows, FUSED_MQA_TOPK_WINDOW_ROWS):
        n = min(FUSED_MQA_TOPK_WINDOW_ROWS, rows - start)
        window = slice(start, start + n)
        q_pad, w_pad = _pad_heads(q[window], weights[window])
        torch.ops._C.fused_mqa_topk_prefill(
            q_pad,
            k,
            scales,
            w_pad,
            starts[window],
            ends[window],
            output[window],
            counter,
            codes,
            values[:n],
            indices[:n],
            row_state.view(-1)[: 3 * n].view(3, n),
            recovery,
            num_ctas,
        )


def fused_mqa_topk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    output: torch.Tensor,
    *,
    num_init_tokens: int = 0,
    num_local_tokens: int = 0,
    num_ctas: int | None = None,
) -> None:
    """Fused replacement of `fp8_fp4_mqa_logits` + `top_k_per_row_prefill`
    (LiteTopK sample -> filter -> select): per block of rows, the candidate
    producer scores a strided sample of the keys, takes a threshold per row
    from it and keeps the keys that score at least that; the selector takes
    each row's exact top-k of its candidates; rows whose candidates overflowed
    or fell short of the top-k are rescored densely and selected exactly. All
    on the GPU: no logits matrix, no host synchronization.

    q: [rows, heads, 128] fp8 e4m3 with 1 <= heads <= 64 (zero-padded to 32 or
    64); k: [keys, 128] fp8 e4m3 (keys <= FUSED_MQA_TOPK_MAX_KEYS); scales:
    [keys] fp32; weights: [rows, heads] fp32; starts/ends: [rows] int32, the
    visible key range `[start, end)` of each row; output: [rows, k] int32 with
    k (the top-k) a multiple of 4 in [4, 2048] and row stride a multiple of 4.
    q, k, scales, weights and output are 16-byte aligned (the op checks the
    contract).

    With forced tokens (`num_init_tokens`, `num_local_tokens`), rows longer
    than `k` keep their first `num_init_tokens` and last `num_local_tokens`
    keys in the first columns and the top `k - num_init_tokens -
    num_local_tokens` of the keys between; shorter rows keep every key, packed
    in front.

    `num_ctas` caps the persistent producer grid (None: follow DeepGEMM's live
    SM budget via `fused_mqa_topk_num_ctas()`; 0: all SMs).

    Writes the same convention as `ops.top_k_per_row_prefill`: indices
    relative to the row's `start` (i.e. `key - start`), unordered, `-1`
    padded when fewer than k keys are visible. Exact on the kernels' fp32
    scores up to ties at the k-th score. The thresholds are fp16 distances to a
    sampled score of each row (a common offset of the scores does not matter);
    rows whose top scores spread over more than 65504 (or are packed within
    2^-24) may overflow and take the recovery: exact, but seconds at 1M keys.
    """
    if num_init_tokens < 0 or num_local_tokens < 0:
        raise ValueError(
            "num_init_tokens and num_local_tokens must be nonnegative, "
            f"got ({num_init_tokens}, {num_local_tokens})"
        )
    if q.shape[0] == 0:
        return
    resolved_ctas = fused_mqa_topk_num_ctas() if num_ctas is None else num_ctas
    logger.info_once(
        "Using the fused LiteTopK MQA+top-k prefill kernels (FP8, %d heads, top-k %d).",
        q.shape[1],
        output.shape[1],
    )
    num_forced = num_init_tokens + num_local_tokens
    if num_forced == 0:
        _run_fused_mqa_topk_windows(
            q, k, scales, weights, starts, ends, output, resolved_ctas
        )
        return
    inner_k = output.shape[1] - num_forced
    if num_forced % 4 != 0 or not is_fused_mqa_topk_k_supported(inner_k):
        raise ValueError(
            f"unsupported topk={output.shape[1]} with num_forced={num_forced} "
            "(both num_forced and topk - num_forced must be multiples of 4, "
            f"with 4 <= topk - num_forced <= {FUSED_MQA_TOPK_MAX_TOPK})"
        )
    # Score only the keys between the forced ones: their top-(k - forced) lands
    # after the forced columns (rows of at most num_forced keys emit -1 there).
    ks = starts.clamp(0, k.shape[0])
    ke = torch.maximum(ends.clamp(0, k.shape[0]), ks)
    inner_ks = (ks + num_init_tokens).clamp(max=k.shape[0])
    inner_ke = torch.maximum(ke - num_local_tokens, inner_ks)
    inner_out = output[:, num_forced:]
    _run_fused_mqa_topk_windows(
        q, k, scales, weights, inner_ks, inner_ke, inner_out, resolved_ctas
    )
    if num_init_tokens > 0:
        torch.where(
            inner_out >= 0, inner_out + num_init_tokens, inner_out, out=inner_out
        )
    output[:, :num_forced] = _forced_columns(
        lengths=ke - ks,
        num_init_tokens=num_init_tokens,
        num_forced_tokens=num_forced,
    )
