# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unscaled FP8 QSA prefill through contiguous DeepGEMM MQA logits."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    get_mqa_logits_metadata,
    is_deep_gemm_supported,
)
from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from ...common.qsa_indexer import QSAIndexerMetadata


def can_use_deepgemm_prefill(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    return (
        current_platform.is_device_capability_family(100)
        and q.dtype == k_cache.dtype == torch.float8_e4m3fn
        and q.shape[1:] == (4, 128)
        and q.is_contiguous()
        and k_cache.shape[2:] == (1, 128)
        and k_cache.stride(-1) == 1
        and is_deep_gemm_supported()
    )


@triton.jit(do_not_specialize=["stride_table_req", "width"])
def _gather_prefill_keys_kernel(
    cache,
    table,
    packed,
    starts,
    visible,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    width,
    PAGE_SIZE: tl.constexpr,
    CP: tl.constexpr,
):
    request = tl.program_id(0)
    tile = tl.program_id(1)
    tiles_per_page: tl.constexpr = triton.cdiv(PAGE_SIZE, CP)
    page = tile // tiles_per_page
    row = tile % tiles_per_page * CP + tl.arange(0, CP)
    column = page * PAGE_SIZE + row
    base = tl.load(starts)
    begin = tl.load(starts + request) - base
    end = tl.load(starts + request + 1) - base
    # Empty requests may occur in padded batches; never read visible[-1].
    useful = tl.load(visible + end - 1, end > begin, other=0)
    physical = tl.load(table + request * stride_table_req + page)
    dim = tl.arange(0, 128)
    live = (row[:, None] < PAGE_SIZE) & (column[:, None] < tl.minimum(width, useful))
    values = tl.load(
        cache
        + physical.to(tl.int64) * stride_cache_block
        + row[:, None] * stride_cache_token
        + dim[None, :],
        live,
        other=0.0,
    )
    tl.store(
        packed
        + request.to(tl.int64) * width * 128
        + column[:, None].to(tl.int64) * 128
        + dim[None, :],
        values,
        live,
    )


@dataclass
class QSAIndexerPrefillMetadata:
    width: int
    rows_per_chunk: int
    ks: torch.Tensor
    ke: torch.Tensor
    schedules: tuple[torch.Tensor, ...]


# One GPU per worker process. Initialized once; never resized or replaced.
_constants: tuple[torch.Tensor, torch.Tensor] | None = None
_limits = (0, 0)


def init_deepgemm_prefill_constants(
    device: torch.device,
    max_tokens: int,
    max_keys: int,
) -> None:
    """Allocate maximum-size unit scales and weights during builder initialization."""
    global _constants, _limits
    if _constants is not None:
        _get_constants(max_tokens, max_keys)
        return
    scales = torch.full(
        (max(4 * max_tokens, max_keys),),
        0x7F7F7F7F,
        dtype=torch.int32,
        device=device,
    )
    weights = torch.ones((max_tokens, 4), dtype=torch.float32, device=device)
    _constants = scales, weights
    _limits = max_tokens, max_keys


def _get_constants(rows: int, keys: int) -> tuple[torch.Tensor, torch.Tensor]:
    if _constants is None:
        raise RuntimeError("QSA prefill constants have not been initialized")
    if rows > _limits[0] or keys > _limits[1]:
        raise RuntimeError(
            f"QSA prefill constants requested ({rows} tokens, {keys} keys), "
            f"exceeding initialized limits {_limits}; buffers cannot grow"
        )
    return _constants


def prepare_deepgemm_prefill(
    metadata: "QSAIndexerMetadata",
    width: int,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> QSAIndexerPrefillMetadata:
    """Prepare schedules for query ranges emitted by the QSA metadata kernel."""
    offset = metadata.num_decode_tokens
    visible = metadata.visible_blocks[offset:]
    requests = metadata.block_table.shape[0] - metadata.num_decodes
    _get_constants(visible.numel(), requests * width)
    # Pinned DeepGEMM: 384-column KV tiles, 256-float pitch, 32-row Q tiles.
    pitch = round_up(round_up(width, 384), 256)
    max_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024**2
    rows_per_chunk = max(1, max_bytes // (pitch * 4) // 32 * 32)
    schedules = tuple(
        get_mqa_logits_metadata(
            ks[start : start + rows_per_chunk],
            ke[start : start + rows_per_chunk],
            requests * width,
            4,
        )
        for start in range(0, visible.numel(), rows_per_chunk)
    )
    return QSAIndexerPrefillMetadata(width, rows_per_chunk, ks, ke, schedules)


def gather_prefill_keys(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    width: int,
) -> torch.Tensor:
    packed = k_cache.new_empty(page_table.shape[0] * width, 128)
    page_size = k_cache.shape[1]
    _gather_prefill_keys_kernel[
        (
            page_table.shape[0],
            triton.cdiv(width, page_size) * triton.cdiv(page_size, 64),
        )
    ](
        k_cache,
        page_table,
        packed,
        query_start_loc,
        visible_blocks,
        *k_cache.stride()[:2],
        page_table.stride(0),
        width,
        PAGE_SIZE=page_size,
        CP=64,
        num_warps=4,
    )
    return packed


def deepgemm_prefill_logits(
    q: torch.Tensor,
    packed: torch.Tensor,
    metadata: QSAIndexerPrefillMetadata,
    start: int,
    end: int,
) -> torch.Tensor:
    q = q[start:end]
    scales, weights = _get_constants(q.shape[0], packed.shape[0])
    q_scales = scales[: q.shape[0] * q.shape[1]].view(q.shape[:2])
    k_scales = scales[: packed.shape[0]]
    logits = fp8_fp4_mqa_logits(
        (q, q_scales),
        (packed, k_scales),
        weights[: end - start],
        metadata.ks[start:end],
        metadata.ke[start:end],
        metadata.width,
        logits_dtype=torch.float32,
        schedule_meta=metadata.schedules[start // metadata.rows_per_chunk],
    )
    return logits[: end - start, : metadata.width]


def warmup_deepgemm_prefill(k_cache: torch.Tensor, num_heads: int) -> None:
    """Compile the shape-independent SM100 scorer, scheduler, and gather paths."""
    q = k_cache.new_zeros((1, num_heads, 128))
    if not can_use_deepgemm_prefill(q, k_cache):
        return
    # One query/key fits every configured limit and compiles the same scorer.
    width = 1
    # Prefill slices after decode rows can have unaligned metadata pointers.
    for offset in range(2):
        table = torch.zeros((2, 1), device=q.device, dtype=torch.int32)[offset:]
        starts = torch.tensor(
            [0, 0, 1] if offset else [0, 1],
            device=q.device,
            dtype=torch.int32,
        )[offset:]
        visible = torch.full((2,), width, device=q.device, dtype=torch.int32)[
            offset : offset + 1
        ]
        ks = torch.zeros(1, device=q.device, dtype=torch.int32)
        ke = torch.full_like(ks, width)
        schedule = get_mqa_logits_metadata(ks, ke, width, 4)
        metadata = QSAIndexerPrefillMetadata(width, 1, ks, ke, (schedule,))
        packed = gather_prefill_keys(k_cache, table[:1], starts, visible, width)
        deepgemm_prefill_logits(q, packed, metadata, 0, 1)
