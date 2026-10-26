# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-launch fused FP8 MQA-logits + top-k prefill selection (SM100 family).

One kernel (``torch.ops._C.fused_mqa_topk_prefill``) scores every visible key
of each query row with the DeepSeek sparse-attention indexer and selects the
row's top-k keys (k = 512, 1024 or 2048), without materializing the
``[rows, keys]`` logits matrix that the dense path (``fp8_fp4_mqa_logits`` +
``top_k_per_row_prefill``) needs.

The kernel is persistent (one CTA per SM); each CTA owns two 8-row candidate
slots. The scratch (~296 MiB on a 148-SM GPU) is allocated once per device and
reused by every call, independently of the number of rows or keys. It is tuned
for the chunks vLLM sends it (rows that see >= fused_mqa_topk_min_keys keys);
smaller key counts are supported but not optimized.
"""

import functools
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

FUSED_MQA_TOPK_HEADS = 32
FUSED_MQA_TOPK_HEAD_DIM = 128
FUSED_MQA_TOPK_TOPKS = (512, 1024, 2048)
# Largest gathered key range a single call accepts (no row limit: the per-row
# workspace is sized from the call, see get_fused_mqa_topk_workspace).
FUSED_MQA_TOPK_MAX_KEYS = 1 << 20
# Rows per fused call. The kernel has no row limit, but prefill chunks above
# 16384 rows (--max-num-batched-tokens > 16384) are not validated end to end,
# so the chunk planner splits fused chunks at this many rows.
FUSED_MQA_TOPK_MAX_ROWS = 16384


def fused_mqa_topk_min_keys(topk: int) -> int:
    """Keys per row below which the dense logits + top-k path is as fast or
    faster, so callers keep prefill chunks whose longest row sees fewer keys on
    the dense path. B300 real captures, fused vs dense: top-512 0.89x at 16K
    keys and 1.06x at 32K; top-1024 0.93x / 1.01x; top-2048 0.92x at 32K and
    1.13x at 64K."""
    return 1 << 16 if topk > 1024 else 1 << 15


# Mirrors csrc/fused_mqa_topk/constants.h (the op checks the workspace shapes).
_CANDIDATES_PER_ROW = 16384  # kCandidatesPerRow
_RETAINED_PER_ROW = 4096  # kRetainedPerRow
_SLOTS_PER_SM = 2  # candidate slots per persistent CTA


@dataclass
class FusedMQATopKWorkspace:
    """Caller-owned scratch of the fused kernel (see fused_mqa_topk.cu)."""

    counter: torch.Tensor  # [2] int32, zero between calls (the kernel restores it)
    values: torch.Tensor  # [8S, 16384] uint16 (low 16 bits of fp24 codes), S = 2/SM
    indices: torch.Tensor  # [8S, 16384] int32
    retained: torch.Tensor  # [8S, 4096] fp32
    retained_indices: torch.Tensor  # [8S, 4096] int32
    counts: torch.Tensor  # [rows] int32, per-row candidate counts
    defer: torch.Tensor  # [rows / 8] int32, blocks queued for the exact rescan

    def as_args(self) -> tuple[torch.Tensor, ...]:
        return (
            self.counter,
            self.values,
            self.indices,
            self.retained,
            self.retained_indices,
            self.counts,
            self.defer,
        )


_WORKSPACES: dict[int, FusedMQATopKWorkspace] = {}


@functools.cache
def fused_mqa_topk_available() -> bool:
    """True when the fused op is compiled in and the GPU is SM100-family."""
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and hasattr(torch.ops._C, "fused_mqa_topk_prefill")
    )


def use_fused_mqa_topk(
    *,
    query_shape: tuple[int, ...],
    topk: int,
    fp4_cache: bool,
    full_context: bool,
    candidate_selection: bool = False,
) -> bool:
    """The one eligibility rule of the fused kernel, shared by the prefill
    chunk planner and the indexer layers (each prefill chunk then also needs
    fused_mqa_topk_chunk).

    query_shape: one token's indexer queries, (heads, head_dim) = (32, 128);
    topk: the per-row
    top-k (pools for kpool indexers) in {512, 1024, 2048}; fp4_cache: the
    MXFP4 indexer cache (and its FP4 queries) stays dense; full_context: every
    rank ranks its rows over the whole context (no DCP, or PCP after the KV
    all-gather); pure DCP needs the per-shard top-k merge; candidate_selection:
    v4.1 two-level selection needs the logits.
    """
    return (
        envs.VLLM_USE_FUSED_MQA_TOPK
        and not fp4_cache
        and full_context
        and not candidate_selection
        and tuple(query_shape) == (FUSED_MQA_TOPK_HEADS, FUSED_MQA_TOPK_HEAD_DIM)
        and topk in FUSED_MQA_TOPK_TOPKS
        and fused_mqa_topk_available()
    )


def fused_mqa_topk_chunk(max_seqlen_k: int, keys: int, topk_tokens: int) -> bool:
    """Whether a prefill chunk of a fused-eligible layer takes the fused
    kernel: its longest row sees at least fused_mqa_topk_min_keys keys (the
    dense path is as fast below) and its gathered keys fit the kernel."""
    return (
        max_seqlen_k >= fused_mqa_topk_min_keys(topk_tokens)
        and keys <= FUSED_MQA_TOPK_MAX_KEYS
    )


def get_fused_mqa_topk_workspace(
    device: torch.device, rows: int
) -> FusedMQATopKWorkspace:
    """Return the per-device workspace, allocating it on first use.

    The per-row part (counts, defer) holds at least ``rows`` query rows and grows
    when a larger call arrives; the profiling run asks for the largest prefill
    chunk (max_num_batched_tokens rows) so serving never grows it. The block
    counter is zeroed once; every launch leaves it zeroed again. The workspace is
    shared by every call on the device, so calls must not run concurrently (they
    are issued on one stream).
    """
    rows = (rows + 7) // 8 * 8
    index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    ws = _WORKSPACES.get(index)
    if ws is None:
        dev = torch.device("cuda", index)
        num_sms = torch.cuda.get_device_properties(index).multi_processor_count
        slots = _SLOTS_PER_SM * num_sms
        ws = FusedMQATopKWorkspace(
            counter=torch.zeros(2, dtype=torch.int32, device=dev),
            values=torch.empty(
                (8 * slots, _CANDIDATES_PER_ROW), dtype=torch.uint16, device=dev
            ),
            indices=torch.empty(
                (8 * slots, _CANDIDATES_PER_ROW), dtype=torch.int32, device=dev
            ),
            retained=torch.empty(
                (8 * slots, _RETAINED_PER_ROW), dtype=torch.float32, device=dev
            ),
            retained_indices=torch.empty(
                (8 * slots, _RETAINED_PER_ROW), dtype=torch.int32, device=dev
            ),
            counts=torch.empty(rows, dtype=torch.int32, device=dev),
            defer=torch.empty(rows // 8, dtype=torch.int32, device=dev),
        )
        nbytes = sum(t.numel() * t.element_size() for t in ws.as_args())
        logger.info(
            "Allocated fused MQA+top-k workspace on cuda:%d (%d slots, %.1f MiB).",
            index,
            slots,
            nbytes / 2**20,
        )
        _WORKSPACES[index] = ws
    elif ws.counts.numel() < rows:
        ws.counts = torch.empty(rows, dtype=torch.int32, device=ws.counts.device)
        ws.defer = torch.empty(rows // 8, dtype=torch.int32, device=ws.defer.device)
        logger.info(
            "Grew the fused MQA+top-k row workspace on cuda:%d to %d rows.", index, rows
        )
    return ws


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
    logger.info_once(
        "Using the single-launch fused MQA+top-k prefill kernel "
        "(FP8, 32 heads, top-k %d).",
        output.shape[1],
    )
    ws = get_fused_mqa_topk_workspace(q.device, q.shape[0])
    torch.ops._C.fused_mqa_topk_prefill(
        q, k, scales, weights, starts, ends, output, *ws.as_args()
    )
