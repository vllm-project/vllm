# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax CUTLASS sparse decode using per-query-token page indices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_MAX_NUM_Q_HEADS = 64
_MAX_NUM_KV_HEADS = 4
_HEAD_DIM = 128
_PAGE_SIZE = 128
_TOPK = 16
# fmha_sm100 plans one row per query head. Keep every cached plan within the
# fixed planner allocation used by the MSA decode kernel.
_MAX_QUERY_HEAD_ROWS = 65536
_MAX_DECODE_QUERY_LEN = 32
# Kernel benchmarks put the CUTLASS crossover at 16 requests for TP1 and TP4.
_MIN_CUTLASS_BATCH_SIZE = 16


@dataclass
class MSACutlassDecodeMetadata:
    plan: Any
    page_table: torch.Tensor


@triton.jit
def _update_runtime_metadata_kernel(
    seq_lens_ptr,
    kv_segment_lens_ptr,
    qo_offset_ptr,
    num_rows: tl.constexpr,
    decode_query_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    request = offsets // decode_query_len
    local_query = offsets % decode_query_len
    seq_len = tl.load(seq_lens_ptr + request, mask=mask)
    tl.store(kv_segment_lens_ptr + offsets, seq_len, mask=mask)
    tl.store(
        qo_offset_ptr + offsets,
        seq_len - decode_query_len + local_query,
        mask=mask,
    )


@dataclass
class MSACutlassDecodePlanCache:
    """Reusable plans whose mutable tensors retain cudagraph-stable addresses."""

    plans: dict[tuple[int, ...], Any] = field(init=False, default_factory=dict)

    def _build_plan(
        self,
        *,
        batch: int,
        decode_query_len: int,
        page_table_stride: int,
        initial_seq_lens: torch.Tensor,
        device: torch.device,
        num_q_heads: int,
        num_kv_heads: int,
        page_size: int,
        topk_blocks: int,
    ) -> Any:
        from vllm.third_party.fmha_sm100.api import fmha_sm100_plan

        qo_lens_cpu = torch.full((batch,), decode_query_len, dtype=torch.int32)
        kv_lens_cpu = initial_seq_lens.to(device="cpu", dtype=torch.int32)
        plan = fmha_sm100_plan(
            qo_lens_cpu,
            kv_lens_cpu,
            num_q_heads,
            num_kv_heads=num_kv_heads,
            qo_offset=kv_lens_cpu - qo_lens_cpu,
            page_size=page_size,
            output_maxscore=False,
            kv_block_num=topk_blocks,
            causal=True,
            sparse_kernel_mode="decode",
            use_fp8_kvcache=True,
            split_prefill_decode=False,
            device=device,
        )

        plan_info = plan[3]
        row_starts = (
            torch.arange(batch, dtype=torch.int32)
            .mul_(page_table_stride)
            .repeat_interleave(decode_query_len)
        )
        page_indptr = torch.cat(
            (
                row_starts,
                torch.tensor([batch * page_table_stride], dtype=torch.int32),
            )
        )
        plan_info["kv_page_indptr"].copy_(
            page_indptr.to(device=device, non_blocking=True)
        )
        return plan

    def prepare(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        decode_query_len: int,
        *,
        num_q_heads: int,
        num_kv_heads: int,
        page_size: int,
        topk_blocks: int,
    ) -> MSACutlassDecodeMetadata:
        batch = int(seq_lens.shape[0])
        if (
            block_table.device.type != "cuda"
            or block_table.dtype != torch.int32
            or not block_table.is_contiguous()
            or block_table.shape[0] != batch
        ):
            raise ValueError(
                "MSA sparse decode requires a contiguous CUDA int32 block "
                "table with one row per request"
            )
        if (
            seq_lens.dtype != torch.int32
            or not seq_lens.is_contiguous()
            or seq_lens.device != block_table.device
        ):
            raise ValueError(
                "MSA sparse decode requires contiguous CUDA int32 sequence "
                "lengths on the block table device"
            )

        page_table_stride = int(block_table.stride(0))
        key = (
            batch,
            decode_query_len,
            page_table_stride,
            num_q_heads,
            num_kv_heads,
            page_size,
            topk_blocks,
        )
        plan = self.plans.get(key)
        if plan is None:
            plan = self._build_plan(
                batch=batch,
                decode_query_len=decode_query_len,
                page_table_stride=page_table_stride,
                initial_seq_lens=seq_lens,
                device=seq_lens.device,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                page_size=page_size,
                topk_blocks=topk_blocks,
            )
            self.plans[key] = plan

        plan_info = plan[3]
        num_rows = batch * decode_query_len
        _update_runtime_metadata_kernel[(triton.cdiv(num_rows, 128),)](
            seq_lens,
            plan_info["kv_segment_lens"],
            plan_info["qo_offset"],
            num_rows=num_rows,
            decode_query_len=decode_query_len,
            BLOCK_SIZE=128,
        )
        return MSACutlassDecodeMetadata(
            plan=plan,
            page_table=block_table.view(-1),
        )


def _cutlass_decode_requested() -> bool:
    return envs.VLLM_MINIMAX_M3_MSA_DECODE_BACKEND == "cutlass"


def _supported_head_geometry(num_q_heads: int, num_kv_heads: int) -> bool:
    return (
        0 < num_q_heads <= _MAX_NUM_Q_HEADS
        and 0 < num_kv_heads <= _MAX_NUM_KV_HEADS
        and num_q_heads % num_kv_heads == 0
    )


def should_prepare_decode_metadata(
    batch_size: int,
    decode_query_len: int,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    kv_cache_dtype: str,
    page_size: int,
    topk_blocks: int,
) -> bool:
    """Return whether a graph shape can use the CUTLASS decode path."""
    total_q = batch_size * decode_query_len
    return (
        _cutlass_decode_requested()
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and kv_cache_dtype in ("fp8", "fp8_e4m3")
        and 1 <= decode_query_len <= _MAX_DECODE_QUERY_LEN
        and batch_size >= _MIN_CUTLASS_BATCH_SIZE
        and total_q * num_q_heads <= _MAX_QUERY_HEAD_ROWS
        and _supported_head_geometry(num_q_heads, num_kv_heads)
        and page_size == _PAGE_SIZE
        and topk_blocks == _TOPK
    )


@torch.no_grad()
def prepare_decode_metadata(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
    topk_blocks: int,
    plan_cache: MSACutlassDecodePlanCache | None = None,
) -> MSACutlassDecodeMetadata:
    """Prepare graph-stable runtime metadata for one sparse decode step."""
    cache = plan_cache or MSACutlassDecodePlanCache()
    return cache.prepare(
        block_table,
        seq_lens,
        decode_query_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        topk_blocks=topk_blocks,
    )


def _static_fallback_reason(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    topk: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    metadata: MSACutlassDecodeMetadata | None,
    *,
    num_kv_heads: int,
    block_size: int,
    topk_blocks: int,
    decode_query_len: int,
    q_scale: torch.Tensor | None,
) -> str | None:
    if not _cutlass_decode_requested():
        return "backend was not requested"
    if metadata is None:
        return "MSA decode metadata is unavailable"
    if query.device.type != "cuda":
        return "query is not on CUDA"
    if torch.cuda.get_device_capability(query.device) not in ((10, 0), (10, 3)):
        return "GPU is not SM100 or SM103"
    batch = int(seq_lens.shape[0])
    total_q = batch * decode_query_len
    if batch < _MIN_CUTLASS_BATCH_SIZE:
        return f"batch size is below {_MIN_CUTLASS_BATCH_SIZE}"
    if not 1 <= decode_query_len <= _MAX_DECODE_QUERY_LEN:
        return f"decode query length is outside [1, {_MAX_DECODE_QUERY_LEN}]"
    if query.ndim != 3:
        return "query geometry is unsupported"
    num_q_heads = int(query.shape[1])
    if not _supported_head_geometry(num_q_heads, num_kv_heads):
        return "query and KV head geometry is unsupported"
    if total_q * num_q_heads > _MAX_QUERY_HEAD_ROWS:
        return "total query-head rows exceed the CUTLASS planner limit"
    if (
        query.shape != (total_q, num_q_heads, _HEAD_DIM)
        or output.shape != query.shape
        or query.dtype not in (torch.bfloat16, torch.float16)
        or output.dtype != query.dtype
        or block_size != _PAGE_SIZE
        or topk_blocks != _TOPK
        or topk.shape != (total_q, num_kv_heads, _TOPK)
        or topk.dtype != torch.int32
        or not topk.is_contiguous()
        or seq_lens.dtype != torch.int32
    ):
        return "query, output, or sparse metadata geometry is unsupported"
    if (
        kv_cache.dtype != torch.float8_e4m3fn
        or kv_cache.ndim != 4
        or kv_cache.shape[1:] != (num_kv_heads, _PAGE_SIZE, 2 * _HEAD_DIM)
        or kv_cache.stride(-1) != 1
    ):
        return "KV cache is not interleaved FP8 E4M3 with the expected geometry"
    if any(
        tensor.device != query.device
        for tensor in (
            kv_cache,
            topk,
            seq_lens,
            output,
            metadata.page_table,
        )
    ):
        return "decode tensors are on different devices"
    if not query.is_contiguous() or not output.is_contiguous():
        return "query or output is not contiguous"
    if (
        q_scale is None
        or q_scale.numel() != 1
        or q_scale.dtype != torch.float32
        or q_scale.device != query.device
    ):
        return "a scalar device FP32 query scale is required"
    return None


@dataclass
class _QueryBufferAllocation:
    capacity: int
    num_heads: int
    query_fp8: torch.Tensor


@dataclass
class _QueryBufferPool:
    current: _QueryBufferAllocation | None = field(init=False, default=None)
    retired: list[_QueryBufferAllocation] = field(init=False, default_factory=list)

    def get(self, query: torch.Tensor) -> torch.Tensor:
        tokens = int(query.shape[0])
        num_heads = int(query.shape[1])
        if (
            self.current is None
            or tokens > self.current.capacity
            or num_heads != self.current.num_heads
        ):
            if self.current is not None:
                self.retired.append(self.current)
            capacity = 1 << (tokens - 1).bit_length()
            self.current = _QueryBufferAllocation(
                capacity=capacity,
                num_heads=num_heads,
                query_fp8=torch.empty(
                    (capacity, num_heads, _HEAD_DIM),
                    dtype=torch.float8_e4m3fn,
                    device=query.device,
                ),
            )
        return self.current.query_fp8[:tokens]


class MSACutlassSparseDecodeRunner:
    """Layer-owned FP8 query buffers for MSA ``fmha_sm100`` sparse decode."""

    def __init__(self) -> None:
        self._query_pools: dict[torch.device, _QueryBufferPool] = {}

    def _get_query_buffer(self, query: torch.Tensor) -> torch.Tensor:
        pool = self._query_pools.get(query.device)
        if pool is None:
            pool = _QueryBufferPool()
            self._query_pools[query.device] = pool
        return pool.get(query)

    @torch.no_grad()
    def try_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk: torch.Tensor,
        seq_lens: torch.Tensor,
        output: torch.Tensor,
        metadata: MSACutlassDecodeMetadata | None,
        *,
        num_kv_heads: int,
        scale: float,
        block_size: int,
        topk_blocks: int,
        decode_query_len: int,
        q_scale: torch.Tensor | None,
        q_scale_float: float,
        k_scale_float: float,
        v_scale_float: float,
    ) -> bool:
        reason = _static_fallback_reason(
            query,
            kv_cache,
            topk,
            seq_lens,
            output,
            metadata,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            topk_blocks=topk_blocks,
            decode_query_len=decode_query_len,
            q_scale=q_scale,
        )
        if reason is not None:
            if (
                _cutlass_decode_requested()
                and reason != "MSA decode metadata is unavailable"
            ):
                logger.warning_once(
                    "MiniMax CUTLASS sparse decode fallback: %s; using Triton",
                    reason,
                )
            return False
        assert metadata is not None
        assert q_scale is not None

        query_fp8 = self._get_query_buffer(query)
        ops.scaled_fp8_quant(
            query.view(query.shape[0], -1),
            scale=q_scale,
            output=query_fp8.view(query.shape[0], -1),
        )
        key, value = kv_cache.split(_HEAD_DIM, dim=-1)

        from vllm.third_party.fmha_sm100.api import fmha_sm100

        fmha_sm100(
            query_fp8,
            key,
            value,
            metadata.plan,
            kv_indices=metadata.page_table,
            kv_block_indexes=topk,
            out=output,
            output_maxscore=False,
            output_o=True,
            sm_scale=scale,
            q_scale=q_scale_float,
            k_scale=k_scale_float,
            v_scale=v_scale_float,
            o_scale=1.0,
        )
        logger.info_once(
            "MiniMax M3 sparse decode dispatched the MSA CUTLASS "
            "fmha_sm100 sparse kernel"
        )
        return True
