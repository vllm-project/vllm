# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-launch fused FP8 MQA-logits + top-k prefill selection (SM100 family).

One kernel (``torch.ops._C.fused_mqa_topk_prefill``) scores every visible key
of each query row with the DeepSeek sparse-attention indexer and selects the
row's top-2048 keys, without materializing the ``[rows, keys]`` logits matrix
that the dense path (``fp8_fp4_mqa_logits`` + ``top_k_per_row_prefill``) needs.

Scratch memory is a fixed pool of 8-row slots (2 per SM, ~311 MB on a 148-SM
GPU) that is allocated once per device and reused by every call, independently
of the number of rows or keys.
"""

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

FUSED_MQA_TOPK_HEADS = 32
FUSED_MQA_TOPK_HEAD_DIM = 128
FUSED_MQA_TOPK_TOPK = 2048
# Largest query chunk / gathered key range a single call accepts.
FUSED_MQA_TOPK_MAX_ROWS = 16384
FUSED_MQA_TOPK_MAX_KEYS = 1 << 20

_CANDIDATES_PER_ROW = 16384
_RETAINED_PER_ROW = 4096
_SLOTS_PER_SM = 2
_STATS_PER_CTA = 7


@dataclass
class FusedMQATopKWorkspace:
    """Caller-owned scratch of the fused kernel (see fused_mqa_topk.cu)."""

    slot_flags: torch.Tensor  # [S] int32, zero between calls
    values: torch.Tensor  # [8S, 16384] fp16
    indices: torch.Tensor  # [8S, 16384] int32
    retained: torch.Tensor  # [8S, 4096] fp32
    retained_indices: torch.Tensor  # [8S, 4096] int32
    counts: torch.Tensor  # [16384] int32
    gates: torch.Tensor  # [16384] fp32
    stats: torch.Tensor  # [16384 / 8 * 7] int32, per-CTA diagnostics

    def as_args(self) -> tuple[torch.Tensor, ...]:
        return (
            self.slot_flags,
            self.values,
            self.indices,
            self.retained,
            self.retained_indices,
            self.counts,
            self.gates,
            self.stats,
        )


_WORKSPACES: dict[int, FusedMQATopKWorkspace] = {}


def fused_mqa_topk_available() -> bool:
    """True when the fused op is compiled in and the GPU is SM100-family."""
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and hasattr(torch.ops._C, "fused_mqa_topk_prefill")
    )


def get_fused_mqa_topk_workspace(device: torch.device) -> FusedMQATopKWorkspace:
    """Return the per-device workspace, allocating it on first use.

    The slot flags are zeroed once; every kernel launch releases the slots it
    claimed, so the flags are zero again whenever the stream is idle.
    """
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
        rows = FUSED_MQA_TOPK_MAX_ROWS
        ws = FusedMQATopKWorkspace(
            slot_flags=torch.zeros(slots, dtype=torch.int32, device=dev),
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
            gates=torch.empty(rows, dtype=torch.float32, device=dev),
            stats=torch.empty(
                rows // 8 * _STATS_PER_CTA, dtype=torch.int32, device=dev
            ),
        )
        nbytes = sum(t.numel() * t.element_size() for t in ws.as_args())
        logger.info(
            "Allocated fused MQA+top-k workspace on cuda:%d (%d slots, %.1f MiB).",
            index,
            slots,
            nbytes / 2**20,
        )
        _WORKSPACES[index] = ws
    return ws


def fused_mqa_topk_config_supported(q: torch.Tensor, topk_tokens: int) -> bool:
    """Whether this indexer's query layout and top-k fit the fused kernel."""
    return (
        topk_tokens == FUSED_MQA_TOPK_TOPK
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
        and 0 < q.shape[0] <= FUSED_MQA_TOPK_MAX_ROWS
        and 0 < k.shape[0] <= FUSED_MQA_TOPK_MAX_KEYS
        and fused_mqa_topk_config_supported(q, topk_tokens)
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
    rows_start_at_zero: bool = False,
) -> bool:
    """Fused replacement of ``fp8_fp4_mqa_logits`` + ``top_k_per_row_prefill``.

    q: [rows, 32, 128] fp8 e4m3; k: [keys, 128] fp8 e4m3; scales: [keys] fp32;
    weights: [rows, 32] fp32; starts/ends: [rows] int32, the visible key range
    ``[start, end)`` of each row; output: [rows, 2048] int32.

    Writes the same convention as ``ops.top_k_per_row_prefill``: indices
    relative to the row's ``start`` (i.e. ``key - start``), unordered, ``-1``
    padded when fewer than 2048 keys are visible. ``rows_start_at_zero=True``
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
        or tuple(output.shape) != (rows, FUSED_MQA_TOPK_TOPK)
    ):
        return False

    logger.info_once(
        "Using the single-launch fused MQA+top-k prefill kernel "
        "(FP8, 32 heads, top-k 2048)."
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
    ws = get_fused_mqa_topk_workspace(q.device)
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
