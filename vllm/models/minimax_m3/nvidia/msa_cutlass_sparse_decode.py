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
from vllm.v1.kv_cache_interface import get_kv_quant_mode

_MAX_NUM_Q_HEADS = 64
_MAX_NUM_KV_HEADS = 4
_HEAD_DIM = 128
_PAGE_SIZE = 128
_TOPK = 16
# fmha_sm100 plans one row per query head. Keep every cached plan within the
# fixed planner allocation used by the MSA decode kernel.
_MAX_QUERY_HEAD_ROWS = 65536
_MAX_DECODE_QUERY_LEN = 32
_DEFAULT_MIN_CUTLASS_BATCH_SIZE = 16
_B300_EAGLE3_QUERY_HEAD_ROWS = 1024
_B300_EAGLE3_MAX_QUERY_LEN = 8
_B300_EAGLE3_HEAD_GEOMETRIES = {(64, 4), (32, 2), (16, 1)}


def is_nvfp4_kv_cache(kv_cache_dtype: str) -> bool:
    return get_kv_quant_mode(kv_cache_dtype).is_nvfp4


def nvfp4_kv_cache_views(
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the K data, K scale, V data and V scale views of an NVFP4 cache.

    ``kv_cache`` is uint8 ``[pages, 2 * Hkv, page_size, 72]``; slot
    ``2 * head + side`` (K = 0, V = 1) is that head's E2M1 data
    ``[page_size, 64]`` followed by its E4M3 block scales ``[page_size, 8]``,
    the one page layout ``fmha_sm100`` reads.
    """
    from vllm.third_party.fmha_sm100.nvfp4_kv import nvfp4_head_slot_views

    return nvfp4_head_slot_views(*nvfp4_kv_cache_slots(kv_cache))


def nvfp4_kv_cache_slots(
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-head K and V slots of an NVFP4 cache."""
    return kv_cache[:, 0::2], kv_cache[:, 1::2]


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
            # The Q8KV4 route keeps plan-time lengths and its own page offsets,
            # which this cached plan does not refresh.
            decode_backend="kv_mode3",
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


def _min_cutlass_batch_size(
    decode_query_len: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> int:
    if (
        not current_platform.is_device_capability((10, 3))
        or not 1 <= decode_query_len <= _B300_EAGLE3_MAX_QUERY_LEN
        or (num_q_heads, num_kv_heads) not in _B300_EAGLE3_HEAD_GEOMETRIES
    ):
        return _DEFAULT_MIN_CUTLASS_BATCH_SIZE
    query_head_rows = decode_query_len * num_q_heads
    return (_B300_EAGLE3_QUERY_HEAD_ROWS + query_head_rows - 1) // query_head_rows


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
    nvfp4 = is_nvfp4_kv_cache(kv_cache_dtype)
    return (
        (decode_backend == "cutlass" or nvfp4)
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and (kv_cache_dtype in ("fp8", "fp8_e4m3") or nvfp4)
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
    min_batch_size: int | None = None,
) -> bool:
    """Return whether a graph shape can use the CUTLASS decode path."""
    total_q = batch_size * decode_query_len
    if min_batch_size is None:
        min_batch_size = _min_cutlass_batch_size(
            decode_query_len, num_q_heads, num_kv_heads
        )
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
        and (batch_size >= min_batch_size or is_nvfp4_kv_cache(kv_cache_dtype))
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
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Run CUTLASS sparse decode with metadata prepared by the MSA builder.

    NVFP4 caches are uint8 head-slot pages (see ``nvfp4_kv_cache_views``) and
    take the device global scales ``k_scale``/``v_scale``.
    """
    if kv_cache.dtype == torch.uint8:
        assert k_scale is not None and v_scale is not None
        key, value = nvfp4_kv_cache_slots(kv_cache)
        k_scale_arg: float | torch.Tensor = k_scale
        v_scale_arg: float | torch.Tensor = v_scale
    else:
        key, value = kv_cache.split(_HEAD_DIM, dim=-1)
        k_scale_arg, v_scale_arg = k_scale_float, v_scale_float

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
        k_scale=k_scale_arg,
        v_scale=v_scale_arg,
        o_scale=1.0,
    )
