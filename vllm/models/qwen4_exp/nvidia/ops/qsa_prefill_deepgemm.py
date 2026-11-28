# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unscaled FP8 QSA prefill through contiguous DeepGEMM MQA logits.

Prefill requests are planned into groups. Each group's compressed keys are
gathered once, packed back to back, and its query rows are scored in chunks.
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB bounds the gathered keys and a chunk's
logits combined (the DeepSeek indexer applies it to logits only), so the
profiling run's touch of that budget covers both. Only a request too long to
fit with one Q tile of logits exceeds it, and every eligible step still takes
this path.
"""

import itertools
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
from vllm.utils.math_utils import cdiv, round_up

if TYPE_CHECKING:
    from ..qsa_indexer_metadata import QSAIndexerMetadata

_NUM_HEADS = 4
_HEAD_DIM = 128
# Pinned DeepGEMM pads logits rows to 32-row Q tiles and the row pitch to its
# 384-column KV tiles, then to 256 floats.
_Q_TILE = 32


def deepgemm_prefill_supported(
    dtype: torch.dtype, num_heads: int, head_dim: int
) -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and dtype == torch.float8_e4m3fn
        and num_heads == _NUM_HEADS
        and head_dim == _HEAD_DIM
        and is_deep_gemm_supported()
    )


def can_use_deepgemm_prefill(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    return (
        current_platform.is_device_capability_family(100)
        and q.dtype == k_cache.dtype == torch.float8_e4m3fn
        and q.shape[1:] == (_NUM_HEADS, _HEAD_DIM)
        and q.is_contiguous()
        and k_cache.shape[2:] == (1, _HEAD_DIM)
        and k_cache.stride(-1) == 1
        and is_deep_gemm_supported()
    )


def _logits_width(num_keys: int) -> int:
    # 64-column multiple keeps the logits stride cooperative_topk-compatible.
    return max(64, round_up(num_keys, 64))


def _logits_bytes(rows: int, width: int) -> int:
    pitch = round_up(round_up(width, 384), 256)
    return round_up(rows, _Q_TILE) * pitch * 4


def _scratch_bytes(rows: int, keys: int, longest: int) -> int:
    """Gathered keys plus the logits of `rows` query rows over them."""
    return keys * _HEAD_DIM + _logits_bytes(rows, _logits_width(longest))


def plan_prefill_chunks(
    query_lens: list[int],
    key_lens: list[int],
    max_bytes: int,
) -> list[tuple[int, int, int, int]]:
    """Split prefill requests into (req_start, req_end, row_start, row_end).

    Requests are grouped while their keys take at most half of `max_bytes`,
    so a lone request is the only group past it. A group's rows, across its
    requests, are split into even chunks whose logits fit beside its keys,
    down to one Q tile. Rows are relative to the group's first row, so a
    chunk with row_start > 0 reuses the keys gathered for the group's first
    chunk.
    """
    chunks = []
    num_requests = len(query_lens)
    start = 0
    while start < num_requests:
        end, keys = start + 1, key_lens[start]
        while (
            end < num_requests and (keys + key_lens[end]) * _HEAD_DIM <= max_bytes // 2
        ):
            keys += key_lens[end]
            end += 1
        rows = sum(query_lens[start:end])
        width = _logits_width(max(key_lens[start:end]))
        row_bytes = _logits_bytes(_Q_TILE, width) // _Q_TILE
        logits_bytes = max_bytes - keys * _HEAD_DIM
        step = max(_Q_TILE, logits_bytes // row_bytes // _Q_TILE * _Q_TILE)
        # Even chunks: a short tail chunk underfills the GPU.
        # max: a group with no query rows (an empty padded request) gets no chunks.
        step = max(_Q_TILE, round_up(cdiv(rows, max(1, cdiv(rows, step))), _Q_TILE))
        for row in range(0, rows, step):
            chunks.append((start, end, row, min(row + step, rows)))
        start = end
    return chunks


@dataclass
class QSAPrefillChunk:
    # Rows of the prefill query slice scored by this chunk.
    row_start: int
    row_end: int
    # Prefill requests whose keys are packed for this chunk.
    req_start: int
    req_end: int
    # Host upper bounds on the packed keys and the longest request.
    num_keys: int
    width: int
    gather: bool
    ks: torch.Tensor
    ke: torch.Tensor
    schedule: torch.Tensor


@dataclass
class QSADeepGEMMPrefillPlan:
    # Exclusive cumsum of exact per-request compressed lengths, [requests + 1].
    cu_keys: torch.Tensor
    chunks: list[QSAPrefillChunk]


def prepare_deepgemm_prefill(
    metadata: "QSAIndexerMetadata",
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
) -> QSADeepGEMMPrefillPlan:
    """Plan the step's prefill chunks and their DeepGEMM schedules.

    `query_start_loc_cpu` and `seq_lens_cpu` cover the whole batch; host
    lengths only size the plan, while device lengths place the keys.
    """
    num_decodes = metadata.num_decodes
    num_prefills = metadata.num_prefills
    compress_ratio = metadata.compress_ratio
    requests = slice(num_decodes, num_decodes + num_prefills)
    query_start_loc_cpu = query_start_loc_cpu[num_decodes : requests.stop + 1]
    query_lens = torch.diff(query_start_loc_cpu).tolist()
    key_lens = (seq_lens_cpu[requests] // compress_ratio).tolist()
    first_rows = (query_start_loc_cpu - query_start_loc_cpu[0]).tolist()

    cu_keys = metadata.seq_lens.new_zeros(num_prefills + 1)
    torch.cumsum(metadata.seq_lens[requests] // compress_ratio, 0, out=cu_keys[1:])
    rows = slice(metadata.num_decode_tokens, metadata.num_actual_tokens)
    # Rows past the last query token are padding; they belong to no chunk.
    row_keys = cu_keys[(metadata.token_to_req[rows].long() - num_decodes).clamp(0)]
    visible = metadata.visible_blocks[rows]

    max_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    chunks = []
    for req_start, req_end, row_start, row_end in plan_prefill_chunks(
        query_lens, key_lens, max_bytes
    ):
        chunk_rows = slice(
            first_rows[req_start] + row_start, first_rows[req_start] + row_end
        )
        ks = row_keys[chunk_rows] - cu_keys[req_start]
        ke = ks + visible[chunk_rows]
        # DeepGEMM needs a nonempty K; a group with no compressed key yet has
        # only rows with zero visible blocks, which top-k never reads.
        num_keys = max(1, sum(key_lens[req_start:req_end]))
        chunks.append(
            QSAPrefillChunk(
                row_start=chunk_rows.start,
                row_end=chunk_rows.stop,
                req_start=req_start,
                req_end=req_end,
                num_keys=num_keys,
                width=_logits_width(max(key_lens[req_start:req_end])),
                gather=row_start == 0,
                ks=ks,
                ke=ke,
                schedule=get_mqa_logits_metadata(ks, ke, num_keys, _NUM_HEADS),
            )
        )
    return QSADeepGEMMPrefillPlan(cu_keys=cu_keys, chunks=chunks)


# Unit MX scales and FP32 head weights, allocated during memory profiling.
_constants: tuple[torch.Tensor, torch.Tensor] | None = None


def reserve_deepgemm_prefill(
    device: torch.device,
    max_model_len: int,
    compress_ratio: int,
    max_tokens: int,
) -> int:
    """Allocate the unit scales and weights during memory profiling.

    Returns the prefill scratch peak past the budget: one max-length request's
    keys and one Q tile of its logits, which the caller touches together with
    the other scratch alive at the same time.
    """
    global _constants
    max_request_keys = cdiv(max_model_len, compress_ratio)
    # plan_prefill_chunks grows a group only while its keys fit half the
    # budget, so a group is at most that or one max-length request.
    max_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    max_keys = max(max_request_keys, max_bytes // 2 // _HEAD_DIM)
    if _constants is None:
        scales = torch.full(
            (max(_NUM_HEADS * max_tokens, max_keys),),
            0x7F7F7F7F,  # four UE8M0 bytes encoding 1.0
            dtype=torch.int32,
            device=device,
        )
        weights = torch.ones(
            (max_tokens, _NUM_HEADS), dtype=torch.float32, device=device
        )
        _constants = scales, weights
    return _scratch_bytes(_Q_TILE, max_request_keys, max_request_keys)


def _get_constants(rows: int, keys: int) -> tuple[torch.Tensor, torch.Tensor]:
    if _constants is None:
        raise RuntimeError("QSA prefill constants were not reserved")
    scales, weights = _constants
    if rows > weights.shape[0] or max(_NUM_HEADS * rows, keys) > scales.shape[0]:
        raise RuntimeError(
            f"QSA prefill needs {rows} rows and {keys} keys, beyond the "
            f"reserved {weights.shape[0]} rows and {scales.shape[0]} scales"
        )
    return scales, weights


@triton.jit(do_not_specialize=["stride_table_req"])
def _gather_prefill_keys_kernel(
    cache,
    table,
    packed,
    cu_keys,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    PAGE_SIZE: tl.constexpr,
    CP: tl.constexpr,
):
    request = tl.program_id(0)
    tile = tl.program_id(1)
    tiles_per_page: tl.constexpr = triton.cdiv(PAGE_SIZE, CP)
    page = tile // tiles_per_page
    row = tile % tiles_per_page * CP + tl.arange(0, CP)
    column = page * PAGE_SIZE + row
    # cu_keys is a slice of the step's offsets; pack from the group's start.
    base = tl.load(cu_keys)
    begin = tl.load(cu_keys + request) - base
    length = tl.load(cu_keys + request + 1) - base - begin
    # The grid spans the host bound on the longest request; never read a
    # block-table entry past the request's own keys.
    physical = tl.load(
        table + request * stride_table_req + page, page * PAGE_SIZE < length, other=0
    )
    dim = tl.arange(0, 128)
    live = (row[:, None] < PAGE_SIZE) & (column[:, None] < length)
    values = tl.load(
        cache
        + physical.to(tl.int64) * stride_cache_block
        + row[:, None] * stride_cache_token
        + dim[None, :],
        live,
        other=0.0,
    )
    tl.store(
        packed + (begin + column[:, None]).to(tl.int64) * 128 + dim[None, :],
        values,
        live,
    )


def gather_prefill_keys(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    plan: QSADeepGEMMPrefillPlan,
    chunk: QSAPrefillChunk,
) -> torch.Tensor:
    """Pack the chunk's requests' compressed keys back to back."""
    packed = k_cache.new_empty((chunk.num_keys, _HEAD_DIM))
    page_size = k_cache.shape[1]
    _gather_prefill_keys_kernel[
        (
            chunk.req_end - chunk.req_start,
            triton.cdiv(chunk.width, page_size) * triton.cdiv(page_size, 64),
        )
    ](
        k_cache,
        page_table[chunk.req_start : chunk.req_end],
        packed,
        plan.cu_keys[chunk.req_start : chunk.req_end + 1],
        *k_cache.stride()[:2],
        page_table.stride(0),
        PAGE_SIZE=page_size,
        CP=64,
        num_warps=4,
    )
    return packed


def deepgemm_prefill_logits(
    q: torch.Tensor,
    packed: torch.Tensor,
    chunk: QSAPrefillChunk,
) -> torch.Tensor:
    q = q[chunk.row_start : chunk.row_end]
    scales, weights = _get_constants(q.shape[0], chunk.num_keys)
    logits = fp8_fp4_mqa_logits(
        (q, scales[: q.shape[0] * _NUM_HEADS].view(q.shape[:2])),
        (packed[: chunk.num_keys], scales[: chunk.num_keys]),
        weights[: q.shape[0]],
        chunk.ks,
        chunk.ke,
        chunk.width,
        logits_dtype=torch.float32,
        schedule_meta=chunk.schedule,
    )
    return logits[:, : chunk.width]


def warmup_deepgemm_prefill(k_cache: torch.Tensor, num_heads: int) -> None:
    """Compile the shape-independent SM100 scorer, scheduler, and gather paths."""
    q = k_cache.new_zeros((1, num_heads, _HEAD_DIM))
    if not can_use_deepgemm_prefill(q, k_cache):
        return
    ks = torch.zeros(1, device=q.device, dtype=torch.int32)
    ke = torch.ones_like(ks)
    chunk = QSAPrefillChunk(
        row_start=0,
        row_end=1,
        req_start=0,
        req_end=1,
        num_keys=1,
        width=_logits_width(1),
        gather=True,
        ks=ks,
        ke=ke,
        schedule=get_mqa_logits_metadata(ks, ke, 1, _NUM_HEADS),
    )
    # Groups after the first slice the offsets and block table at addresses
    # Triton may not see as 16-byte aligned, and it compiles each separately.
    for table_offset, keys_offset in itertools.product(range(2), repeat=2):
        table = torch.zeros((2, 1), device=q.device, dtype=torch.int32)
        cu_keys = torch.tensor([0, 0, 1], device=q.device, dtype=torch.int32)
        plan = QSADeepGEMMPrefillPlan(cu_keys[keys_offset:], [chunk])
        packed = gather_prefill_keys(k_cache, table[table_offset:], plan, chunk)
    deepgemm_prefill_logits(q, packed, chunk)
