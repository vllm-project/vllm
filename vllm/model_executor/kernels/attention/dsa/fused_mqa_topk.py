# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-launch fused FP8 MQA-logits + top-k prefill selection (SM100 family).

One kernel (``torch.ops._C.fused_mqa_topk_prefill``) scores every visible key
of each query row with the DeepSeek sparse-attention indexer and selects the
row's top-k keys (k = 512, 1024 or 2048), without materializing the
``[rows, keys]`` logits matrix that the dense path (``fp8_fp4_mqa_logits`` +
``top_k_per_row_prefill``) needs.

Above 4K keys the kernel is persistent (one CTA per SM); each CTA owns two
8-row candidate slots. The scratch (~296 MiB on a 148-SM GPU) is allocated once
per device and reused by every call, independently of the number of rows or
keys.
"""

from dataclasses import dataclass

import torch

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
# so the chunk planner splits fused chunks at this many rows. None: one fused
# call per chunk, any size.
# TODO: check for other prefill chunk sizes.
FUSED_MQA_TOPK_MAX_ROWS: int | None = 16384


def fused_mqa_topk_min_keys(topk: int) -> int:
    """Keys per row below which the dense logits + top-k path is as fast or
    faster, so callers keep prefill chunks whose longest row sees fewer keys on
    the dense path. B300 real captures, fused vs dense: top-512 0.89x at 16K
    keys and 1.06x at 32K; top-1024 0.93x / 1.01x; top-2048 0.92x at 32K and
    1.13x at 64K."""
    return 1 << 16 if topk > 1024 else 1 << 15


_CANDIDATES_PER_ROW = 16384
_RETAINED_PER_ROW = 4096
_SLOTS_PER_SM = 2


@dataclass
class FusedMQATopKWorkspace:
    """Caller-owned scratch of the fused kernel (see fused_mqa_topk.cu)."""

    counter: torch.Tensor  # [2] int32, zero between calls (the kernel restores it)
    values: torch.Tensor  # [8S, 16384] fp16, S = 2 slots per SM
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


def fused_mqa_topk_available() -> bool:
    """True when the fused op is compiled in and the GPU is SM100-family."""
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and hasattr(torch.ops._C, "fused_mqa_topk_prefill")
    )


def get_fused_mqa_topk_workspace(
    device: torch.device, rows: int
) -> FusedMQATopKWorkspace:
    """Return the per-device workspace, allocating it on first use.

    The per-row part (counts, defer) holds at least ``rows`` query rows and grows
    when a larger call arrives; the profiling run asks for the largest prefill
    chunk (max_num_batched_tokens rows) so serving never grows it. The block
    counter is zeroed once; every launch leaves it zeroed again.
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
                (8 * slots, _CANDIDATES_PER_ROW), dtype=torch.float16, device=dev
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
        _WORKSPACES[index] = ws
    return ws


def fused_mqa_topk_config_supported(q: torch.Tensor, topk_tokens: int) -> bool:
    """Whether this indexer's query layout and top-k fit the fused kernel."""
    return (
        topk_tokens in FUSED_MQA_TOPK_TOPKS
        and q.dtype == torch.float8_e4m3fn
        and q.dim() == 3
        and tuple(q.shape[1:]) == (FUSED_MQA_TOPK_HEADS, FUSED_MQA_TOPK_HEAD_DIM)
        and q.is_cuda
        and fused_mqa_topk_available()
    )


def fused_mqa_topk_supported(
    q: torch.Tensor, k: torch.Tensor, topk_tokens: int
) -> bool:
    """Static eligibility of an indexer prefill chunk for the fused kernel."""
    return (
        k.dtype == torch.float8_e4m3fn
        and k.dim() == 2
        and k.shape[1] == FUSED_MQA_TOPK_HEAD_DIM
        and q.shape[0] > 0
        and 0 < k.shape[0] <= FUSED_MQA_TOPK_MAX_KEYS
        and fused_mqa_topk_config_supported(q, topk_tokens)
    )


def fused_mqa_topk_key_scales(scales: torch.Tensor) -> torch.Tensor:
    """Per-key fp32 scales from the indexer K cache.

    SM100 caches store each power-of-two scale as four equal UE8M0 bytes in an
    int32; the fp32 value 2**(e - 127) has bit pattern ``e << 23``.
    """
    if scales.dtype == torch.int32:
        return ((scales & 0xFF) << 23).view(torch.float32)
    return scales


def fused_mqa_topk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    output: torch.Tensor,
    *,
    rows_start_at_zero: bool = False,
) -> bool:
    """Fused replacement of ``fp8_fp4_mqa_logits`` + ``top_k_per_row_prefill``.

    q: [rows, 32, 128] fp8 e4m3; k: [keys, 128] fp8 e4m3; scales: [keys] fp32;
    weights: [rows, 32] fp32; starts/ends: [rows] int32, the visible key range
    ``[start, end)`` of each row; output: [rows, k] int32 with k (the top-k)
    512, 1024 or 2048.

    Writes the same convention as ``ops.top_k_per_row_prefill``: indices
    relative to the row's ``start`` (i.e. ``key - start``), unordered, ``-1``
    padded when fewer than k keys are visible. ``rows_start_at_zero=True``
    promises every ``start`` is 0 (single-request chunk) and skips the
    relative-index conversion.

    Returns False (and leaves ``output`` untouched) for unsupported inputs so
    the caller can fall back to the dense path.
    """
    rows, keys = q.shape[0], k.shape[0]
    if (
        not fused_mqa_topk_supported(q, k, output.shape[1] if output.dim() == 2 else 0)
        or scales.dtype != torch.float32
        or scales.numel() != keys
        or weights.dtype != torch.float32
        or tuple(weights.shape) != (rows, FUSED_MQA_TOPK_HEADS)
        or starts.dtype != torch.int32
        or ends.dtype != torch.int32
        or starts.numel() != rows
        or ends.numel() != rows
        or output.dtype != torch.int32
        or output.shape[0] != rows
    ):
        return False

    logger.info_once(
        "Using the single-launch fused MQA+top-k prefill kernel "
        "(FP8, 32 heads, top-k %d).",
        output.shape[1],
    )
    q = q.contiguous()
    k = k.contiguous()
    scales = scales.contiguous()
    weights = weights.contiguous()
    starts = starts.contiguous()
    ends = ends.contiguous()

    # The kernel writes the output with 16-byte stores; any other layout goes
    # through a temporary (never the case for the indexer's own buffers).
    direct = output.is_contiguous() and output.data_ptr() % 16 == 0
    out = (
        output
        if direct
        else torch.empty_like(output, memory_format=torch.contiguous_format)
    )
    ws = get_fused_mqa_topk_workspace(q.device, rows)
    # The kernel emits absolute key indices; the dense top-k (and the sparse
    # attention consumer) use indices relative to the row start. relative=True
    # converts in place (rows with start 0 are skipped on the GPU).
    torch.ops._C.fused_mqa_topk_prefill(
        q,
        k,
        scales,
        weights,
        starts,
        ends,
        out,
        *ws.as_args(),
        not rows_start_at_zero,
    )
    if not direct:
        output.copy_(out)
    return True
