# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental FlashInfer TRTLLM-GEN sparse decode for MiniMax M3."""

from __future__ import annotations

import inspect
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_BACKEND_ENV = "VLLM_MINIMAX_M3_SPARSE_DECODE_BACKEND"
_FLASHINFER_BACKEND = "flashinfer_trtllm_gen"
_NUM_Q_HEADS = 64
_NUM_KV_HEADS = 4
_HEAD_DIM = 128
_PAGE_SIZE = 128
_TOPK = 16
_WORKSPACE_BYTES = 256 * 1024 * 1024
_REQUIRED_DECODE_PARAMETERS = frozenset(
    {
        "bmm1_scale_log2",
        "enable_block_sparse_attention",
        "multi_ctas_kv_counter_buffer",
    }
)

_Api = tuple[Callable[..., Any], Callable[[int, int, int], int]]
_api: _Api | None = None
_api_checked = False


@triton.jit(do_not_specialize=["batch_size", "decode_query_len", "max_logical_pages"])
def _prepare_sparse_metadata_kernel(
    topk_ptr,
    block_table_ptr,
    seq_lens_ptr,
    k_scale_ptr,
    physical_pages_ptr,
    sparse_seq_lens_ptr,
    metadata_ok_ptr,
    metadata_status_ptr,
    bmm1_scale_log2_ptr,
    batch_size,
    decode_query_len,
    max_logical_pages,
    num_physical_pages,
    stride_topk_h,
    stride_topk_b,
    stride_topk_k,
    stride_bt_b,
    stride_bt_k,
    stride_seq_b,
    stride_out_h,
    stride_out_b,
    stride_out_k,
    stride_len_h,
    stride_len_b,
    TOPK_SIZE: tl.constexpr,
    PAGE_SIZE_VALUE: tl.constexpr,
    HAS_KV_SCALE: tl.constexpr,
    SM_SCALE_LOG2: tl.constexpr,
):
    pid = tl.program_id(0)
    kv_head = pid // batch_size
    token = pid - kv_head * batch_size
    request = token // decode_query_len
    local_q = token - request * decode_query_len
    offsets = tl.arange(0, TOPK_SIZE)
    logical = tl.load(
        topk_ptr
        + kv_head * stride_topk_h
        + token * stride_topk_b
        + offsets * stride_topk_k
    ).to(tl.int32)
    request_seq_len = tl.load(seq_lens_ptr + request * stride_seq_b).to(tl.int32)
    seq_len = request_seq_len - decode_query_len + local_q + 1
    valid_pages = (seq_len + PAGE_SIZE_VALUE - 1) // PAGE_SIZE_VALUE
    tail_page = valid_pages - 1
    valid = (logical >= 0) & (logical < valid_pages) & (logical < max_logical_pages)
    sentinel = 0x7FFFFFFF
    remaining = tl.where(valid, logical, sentinel)
    count = tl.zeros((), tl.int32)
    physical_ok = tl.full((), True, tl.int1)

    # Fixed-size selection sort also removes duplicate logical page indices.
    for output_slot in tl.static_range(TOPK_SIZE):
        selected = tl.min(remaining, axis=0)
        has_value = selected < sentinel
        physical = tl.load(
            block_table_ptr + request * stride_bt_b + selected * stride_bt_k,
            mask=has_value,
            other=0,
        ).to(tl.int32)
        tl.store(
            physical_pages_ptr
            + kv_head * stride_out_h
            + token * stride_out_b
            + output_slot * stride_out_k,
            physical,
        )
        physical_ok &= (~has_value) | (
            (physical >= 0) & (physical < num_physical_pages)
        )
        count += has_value.to(tl.int32)
        remaining = tl.where(remaining == selected, sentinel, remaining)

    is_padding = request_seq_len <= 0
    has_tail = tl.sum((valid & (logical == tail_page)).to(tl.int32), axis=0) > 0
    metadata_ok = is_padding | (
        has_tail & physical_ok & (count > 0) & (valid_pages <= max_logical_pages)
    )
    tail_tokens = seq_len - tail_page * PAGE_SIZE_VALUE
    surviving_tokens = (count - 1) * PAGE_SIZE_VALUE + tail_tokens
    surviving_tokens = tl.where(is_padding, 0, surviving_tokens)
    surviving_tokens = tl.where(metadata_ok, surviving_tokens, 0)
    tl.store(
        sparse_seq_lens_ptr + kv_head * stride_len_h + token * stride_len_b,
        surviving_tokens,
    )
    tl.store(metadata_ok_ptr + kv_head * batch_size + token, metadata_ok)
    tl.atomic_and(metadata_status_ptr, metadata_ok.to(tl.int32))

    if HAS_KV_SCALE:
        k_scale = tl.load(k_scale_ptr).to(tl.float32)
        tl.store(
            bmm1_scale_log2_ptr,
            k_scale * SM_SCALE_LOG2,
            mask=pid == 0,
        )


@dataclass
class _Buffers:
    workspace: torch.Tensor
    physical_pages: torch.Tensor
    sparse_seq_lens: torch.Tensor
    metadata_ok: torch.Tensor
    metadata_status: torch.Tensor
    bmm1_scale_log2: torch.Tensor
    counter: torch.Tensor


@dataclass
class _BufferAllocation:
    capacity: int
    physical_pages: torch.Tensor
    sparse_seq_lens: torch.Tensor
    metadata_ok: torch.Tensor
    metadata_status: torch.Tensor
    bmm1_scale_log2: torch.Tensor
    counter: torch.Tensor

    def for_batch(self, batch: int, workspace: torch.Tensor) -> _Buffers:
        return _Buffers(
            workspace=workspace,
            physical_pages=self.physical_pages[: _NUM_KV_HEADS * batch * _TOPK].view(
                _NUM_KV_HEADS, batch, _TOPK
            ),
            sparse_seq_lens=self.sparse_seq_lens[: _NUM_KV_HEADS * batch].view(
                _NUM_KV_HEADS, batch
            ),
            metadata_ok=self.metadata_ok[: _NUM_KV_HEADS * batch].view(
                _NUM_KV_HEADS, batch
            ),
            metadata_status=self.metadata_status,
            bmm1_scale_log2=self.bmm1_scale_log2,
            counter=self.counter,
        )


@dataclass
class _DeviceBufferPool:
    device: torch.device
    workspace: torch.Tensor
    current: _BufferAllocation | None = field(init=False, default=None)
    retired: list[_BufferAllocation] = field(init=False, default_factory=list)

    def get(
        self,
        batch: int,
        counter_size: Callable[[int, int, int], int],
    ) -> _Buffers:
        if batch <= 0:
            raise ValueError("batch must be positive")
        if self.current is None or batch > self.current.capacity:
            if self.current is not None:
                # Captured CUDA graphs retain raw pointers into old allocations.
                self.retired.append(self.current)
            capacity = 1 << (batch - 1).bit_length()
            sm_count = torch.cuda.get_device_properties(
                self.device
            ).multi_processor_count
            counter_bytes = counter_size(capacity, _NUM_Q_HEADS, sm_count)
            self.current = _BufferAllocation(
                capacity=capacity,
                physical_pages=torch.empty(
                    _NUM_KV_HEADS * capacity * _TOPK,
                    dtype=torch.int32,
                    device=self.device,
                ),
                sparse_seq_lens=torch.empty(
                    _NUM_KV_HEADS * capacity,
                    dtype=torch.int32,
                    device=self.device,
                ),
                metadata_ok=torch.empty(
                    _NUM_KV_HEADS * capacity,
                    dtype=torch.int32,
                    device=self.device,
                ),
                metadata_status=torch.ones(1, dtype=torch.int32, device=self.device),
                bmm1_scale_log2=torch.empty(1, dtype=torch.float32, device=self.device),
                counter=torch.zeros(
                    counter_bytes, dtype=torch.uint8, device=self.device
                ),
            )
        current = self.current
        assert current is not None
        return current.for_batch(batch, self.workspace)


_workspace_by_device: dict[torch.device, torch.Tensor] = {}


def _get_workspace(device: torch.device) -> torch.Tensor:
    workspace = _workspace_by_device.get(device)
    if workspace is None:
        workspace = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _workspace_by_device[device] = workspace
    return workspace


def _requested() -> bool:
    value = os.environ.get(_BACKEND_ENV, "triton")
    if value not in ("triton", _FLASHINFER_BACKEND):
        logger.warning_once(
            "Unknown %s=%r; using Triton sparse decode", _BACKEND_ENV, value
        )
        return False
    return value == _FLASHINFER_BACKEND


def _load_api() -> _Api | None:
    global _api, _api_checked
    if _api_checked:
        return _api
    _api_checked = True
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes
    except Exception as error:
        logger.warning_once(
            "FlashInfer MiniMax sparse decode unavailable (%s); using Triton", error
        )
        return None
    parameters = inspect.signature(trtllm_batch_decode_with_kv_cache).parameters
    missing_parameters = _REQUIRED_DECODE_PARAMETERS.difference(parameters)
    if missing_parameters:
        logger.warning_once(
            "Installed FlashInfer lacks sparse decode parameters %s; using Triton",
            tuple(sorted(missing_parameters)),
        )
        return None
    _api = (
        trtllm_batch_decode_with_kv_cache,
        get_trtllm_gen_multi_ctas_kv_counter_bytes,
    )
    return _api


def _static_fallback_reason(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    topk: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    num_kv_heads: int,
    block_size: int,
    topk_blocks: int,
    decode_query_len: int,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> str | None:
    if decode_query_len <= 0:
        return "decode query length is not positive"
    if query.device.type != "cuda":
        return "query is not on CUDA"
    capability = torch.cuda.get_device_capability(query.device)
    if capability not in ((10, 0), (10, 3)):
        return "GPU is not SM100 or SM103"
    if (
        query.ndim != 3
        or query.shape[1:] != (_NUM_Q_HEADS, _HEAD_DIM)
        or num_kv_heads != _NUM_KV_HEADS
        or block_size != _PAGE_SIZE
        or topk_blocks != _TOPK
    ):
        return "MiniMax head/page/top-k geometry differs"
    if query.dtype not in (torch.bfloat16, torch.float16):
        return "query dtype is unsupported"
    if kv_cache.dtype != torch.float8_e4m3fn:
        return "KV cache is not FP8 E4M3"
    if (
        kv_cache.ndim != 4
        or kv_cache.shape[1:]
        != (
            _NUM_KV_HEADS,
            _PAGE_SIZE,
            2 * _HEAD_DIM,
        )
        or kv_cache.stride(-1) != 1
    ):
        return "KV cache shape or inner stride is unsupported"
    batch = block_table.shape[0] if block_table.ndim == 2 else 0
    num_query_tokens = query.shape[0]
    if (
        output.shape != query.shape
        or output.dtype != query.dtype
        or num_query_tokens != batch * decode_query_len
        or topk.shape != (_NUM_KV_HEADS, num_query_tokens, _TOPK)
        or block_table.ndim != 2
        or seq_lens.shape != (batch,)
        or topk.dtype != torch.int32
        or block_table.dtype != torch.int32
        or seq_lens.dtype != torch.int32
    ):
        return "decode tensor shape or metadata dtype is unsupported"
    tensors = (query, kv_cache, topk, block_table, seq_lens, output)
    if any(tensor.device != query.device for tensor in tensors):
        return "decode tensors are on different devices"
    if not query.is_contiguous() or not output.is_contiguous():
        return "query or output is not contiguous"
    if (k_scale is None) != (v_scale is None):
        return "only one KV scale was provided"
    if (
        k_scale is not None
        and v_scale is not None
        and (
            k_scale.numel() != 1
            or v_scale.numel() != 1
            or k_scale.dtype != torch.float32
            or v_scale.dtype != torch.float32
            or k_scale.device != query.device
            or v_scale.device != query.device
        )
    ):
        return "only scalar FP32 KV scales are supported"
    return None


class FlashInferSparseDecodeRunner:
    """Layer-owned FlashInfer sparse decode buffers and dispatch."""

    def __init__(self) -> None:
        self._buffer_pools_by_device: dict[torch.device, _DeviceBufferPool] = {}

    def _get_buffers(
        self,
        device: torch.device,
        batch: int,
        counter_size: Callable[[int, int, int], int],
    ) -> _Buffers:
        pool = self._buffer_pools_by_device.get(device)
        if pool is None:
            pool = _DeviceBufferPool(device, _get_workspace(device))
            self._buffer_pools_by_device[device] = pool
        return pool.get(batch, counter_size)

    @torch.no_grad()
    def try_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        output: torch.Tensor,
        *,
        num_kv_heads: int,
        scale: float,
        block_size: int,
        topk_blocks: int,
        decode_query_len: int,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> bool:
        """Run FlashInfer when explicitly enabled and compatible."""
        if not _requested():
            return False
        reason = _static_fallback_reason(
            query,
            kv_cache,
            topk,
            block_table,
            seq_lens,
            output,
            num_kv_heads,
            block_size,
            topk_blocks,
            decode_query_len,
            k_scale,
            v_scale,
        )
        if reason is not None:
            logger.warning_once(
                "FlashInfer MiniMax sparse decode fallback: %s; using Triton", reason
            )
            return False
        api = _load_api()
        if api is None:
            return False

        # FlashInfer sees each verification token as an independent request so
        # every token retains the sparse page set chosen by MiniMax's indexer.
        batch = query.shape[0]
        run, counter_size = api
        buffers = self._get_buffers(query.device, batch, counter_size)
        physical_pages = buffers.physical_pages
        sparse_seq_lens = buffers.sparse_seq_lens
        metadata_ok = buffers.metadata_ok
        buffers.metadata_status.fill_(1)
        has_kv_scale = k_scale is not None
        k_scale_arg = k_scale if k_scale is not None else buffers.bmm1_scale_log2
        _prepare_sparse_metadata_kernel[(_NUM_KV_HEADS * batch,)](
            topk,
            block_table,
            seq_lens,
            k_scale_arg,
            physical_pages,
            sparse_seq_lens,
            metadata_ok,
            buffers.metadata_status,
            buffers.bmm1_scale_log2,
            batch,
            decode_query_len,
            block_table.shape[1],
            kv_cache.shape[0],
            topk.stride(0),
            topk.stride(1),
            topk.stride(2),
            block_table.stride(0),
            block_table.stride(1),
            seq_lens.stride(0),
            physical_pages.stride(0),
            physical_pages.stride(1),
            physical_pages.stride(2),
            sparse_seq_lens.stride(0),
            sparse_seq_lens.stride(1),
            TOPK_SIZE=_TOPK,
            PAGE_SIZE_VALUE=_PAGE_SIZE,
            HAS_KV_SCALE=has_kv_scale,
            SM_SCALE_LOG2=scale * math.log2(math.e),
        )
        torch._assert_async(
            buffers.metadata_status,
            "MiniMax FlashInfer sparse decode metadata is invalid or misses the "
            "tail page",
        )
        k_cache, v_cache = kv_cache.split(_HEAD_DIM, dim=-1)
        run(
            query=query,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=buffers.workspace,
            block_tables=physical_pages,
            seq_lens=sparse_seq_lens,
            max_seq_len=_TOPK * _PAGE_SIZE,
            bmm1_scale=scale,
            bmm1_scale_log2=(buffers.bmm1_scale_log2 if has_kv_scale else None),
            bmm2_scale=v_scale if v_scale is not None else 1.0,
            out=output,
            kv_layout="HND",
            backend="trtllm-gen",
            q_len_per_req=1,
            multi_ctas_kv_counter_buffer=buffers.counter,
            enable_block_sparse_attention=True,
        )
        logger.info_once(
            "MiniMax M3 sparse decode dispatched FlashInfer TRTLLM-GEN block-sparse "
            "attention"
        )
        return True
