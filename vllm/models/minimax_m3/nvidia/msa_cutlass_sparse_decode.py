# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax CUTLASS sparse decode using per-query-token page indices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.config.attention import MiniMaxM3MSADecodeBackend
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

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
        initial_seq_lens_cpu: torch.Tensor,
        device: torch.device,
        num_q_heads: int,
        num_kv_heads: int,
        page_size: int,
        topk_blocks: int,
    ) -> Any:
        from vllm.third_party.fmha_sm100.api import fmha_sm100_plan

        qo_lens_cpu = torch.full((batch,), decode_query_len, dtype=torch.int32)
        kv_lens_cpu = initial_seq_lens_cpu
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
            torch.arange(batch, dtype=torch.int32, device=device)
            .mul_(page_table_stride)
            .repeat_interleave(decode_query_len)
        )
        page_indptr = torch.cat(
            (
                row_starts,
                torch.tensor(
                    [batch * page_table_stride],
                    dtype=torch.int32,
                    device=device,
                ),
            )
        )
        plan_info["kv_page_indptr"].copy_(page_indptr)
        return plan

    def prepare(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
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
        if (
            seq_lens_cpu.device.type != "cpu"
            or seq_lens_cpu.dtype != torch.int32
            or not seq_lens_cpu.is_contiguous()
            or seq_lens_cpu.shape != seq_lens.shape
        ):
            raise ValueError(
                "MSA sparse decode requires contiguous CPU int32 sequence "
                "lengths matching the device sequence lengths"
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
                initial_seq_lens_cpu=seq_lens_cpu,
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


def _supported_head_geometry(num_q_heads: int, num_kv_heads: int) -> bool:
    return (
        0 < num_q_heads <= _MAX_NUM_Q_HEADS
        and 0 < num_kv_heads <= _MAX_NUM_KV_HEADS
        and num_q_heads % num_kv_heads == 0
    )


def supports_cutlass_sparse_decode(
    *,
    decode_backend: MiniMaxM3MSADecodeBackend,
    num_q_heads: int,
    num_kv_heads: int,
    kv_cache_dtype: str,
    page_size: int,
    topk_blocks: int,
) -> bool:
    """Return whether static model geometry supports CUTLASS sparse decode."""
    return (
        decode_backend == "cutlass"
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and kv_cache_dtype in ("fp8", "fp8_e4m3")
        and _supported_head_geometry(num_q_heads, num_kv_heads)
        and page_size == _PAGE_SIZE
        and topk_blocks == _TOPK
    )


def should_prepare_decode_metadata(
    batch_size: int,
    decode_query_len: int,
    *,
    decode_backend: MiniMaxM3MSADecodeBackend,
    num_q_heads: int,
    num_kv_heads: int,
    kv_cache_dtype: str,
    page_size: int,
    topk_blocks: int,
) -> bool:
    """Return whether a graph shape can use the CUTLASS decode path."""
    total_q = batch_size * decode_query_len
    return (
        supports_cutlass_sparse_decode(
            decode_backend=decode_backend,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            page_size=page_size,
            topk_blocks=topk_blocks,
        )
        and 1 <= decode_query_len <= _MAX_DECODE_QUERY_LEN
        and batch_size >= _MIN_CUTLASS_BATCH_SIZE
        and total_q * num_q_heads <= _MAX_QUERY_HEAD_ROWS
    )


@torch.no_grad()
def prepare_decode_metadata(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
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
        seq_lens_cpu,
        decode_query_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        topk_blocks=topk_blocks,
    )


@torch.no_grad()
def msa_cutlass_sparse_decode(
    query_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    topk: torch.Tensor,
    output: torch.Tensor,
    metadata: MSACutlassDecodeMetadata,
    *,
    scale: float,
    q_scale_float: float,
    k_scale_float: float,
    v_scale_float: float,
) -> None:
    """Run CUTLASS sparse decode with metadata prepared by the MSA builder."""
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
