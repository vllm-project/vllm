# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse top-k prefill KV staging orchestration for Sparse-MLA / QSA (Issue #67874)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.hisparse.prefill_kernels import (
    hisparse_build_selected_mask,
    hisparse_gather_resident_rows,
)
from vllm.v1.hisparse.runtime import _MIN_STAGED_PREFILL_KV_ROWS

if TYPE_CHECKING:
    from vllm.v1.hisparse.runtime import (
        HiSparseCacheHandle,
        HiSparsePrefillStagingPlan,
    )

__all__ = [
    "bind_sparse_prefill_staging",
    "filter_prefill_gather_inputs",
    "gather_sparse_prefill_cache",
    "hisparse_build_selected_mask",
    "hisparse_gather_resident_rows",
    "select_fp8_sparse_prefill_rows",
    "stage_sparse_prefill_rows",
]

_MAX_CACHED_PLANS = 8


def bind_sparse_prefill_staging(group: Any, vllm_config: Any) -> None:
    """Configure sparse prefill staging on the leader of an index group."""
    compilation_config = getattr(vllm_config, "compilation_config", None)
    group._topk_owner = getattr(compilation_config, "static_forward_context", None)
    hisparse_config = vllm_config.attention_config.hisparse_config
    parallel_config = getattr(vllm_config, "parallel_config", None)
    group.sparse_prefill_staging = bool(
        getattr(hisparse_config, "sparse_prefill_staging", False)
        and group.has_indexer
        and getattr(parallel_config, "decode_context_parallel_size", 1) == 1
        and getattr(parallel_config, "prefill_context_parallel_size", 1) == 1
        and not getattr(parallel_config, "use_ubatching", False)
    )


def filter_prefill_gather_inputs(
    miss_mask: torch.Tensor,
    gpu_row_ids: torch.Tensor | None,
    selected_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Restrict prefill host misses and GPU-resident copies to `selected_mask`."""
    if selected_mask is None:
        return miss_mask, gpu_row_ids
    if selected_mask.shape != miss_mask.shape:
        raise ValueError(
            f"HiSparse selected_mask shape {tuple(selected_mask.shape)} "
            f"does not match staging plan shape {tuple(miss_mask.shape)}."
        )
    keep = selected_mask != 0
    filtered_miss = torch.where(keep, miss_mask, 0)
    filtered_gpu = (
        torch.where(keep, gpu_row_ids, -1) if gpu_row_ids is not None else None
    )
    return filtered_miss, filtered_gpu


def gather_sparse_prefill_cache(
    cache_ops: Any,
    kv_cache: torch.Tensor,
    plan: HiSparsePrefillStagingPlan,
    *,
    resident_cache: torch.Tensor | None = None,
    selected_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather only `selected_mask` rows from GPU-resident shadow or host."""
    if kv_cache.shape[1] != plan.block_size:
        raise ValueError(
            f"HiSparse staging block size {plan.block_size} does not match "
            f"the KV cache block size {kv_cache.shape[1]}."
        )
    row_width = kv_cache.shape[-1]
    num_blocks = plan.row_ids.shape[1] // plan.block_size
    min_blocks = cdiv(_MIN_STAGED_PREFILL_KV_ROWS, plan.block_size)
    staged = torch.empty(
        (max(num_blocks, min_blocks), plan.block_size, row_width),
        dtype=kv_cache.dtype,
        device=plan.block_table.device,
    )[:num_blocks]
    miss_mask, gpu_row_ids = filter_prefill_gather_inputs(
        plan.miss_mask, plan.gpu_row_ids, selected_mask
    )
    if gpu_row_ids is not None and resident_cache is not None:
        hisparse_gather_resident_rows(
            resident_cache, staged.view(-1, row_width), gpu_row_ids[0]
        )
    cache_ops.hisparse_gather_plan(
        kv_cache.view(-1, row_width),
        staged,
        plan.row_ids,
        plan.dst_rows,
        miss_mask,
    )
    return staged


def _resolve_group_live_topk(group: Any) -> torch.Tensor | None:
    """Resolve the group's live top-k buffer, including MTP repointed buffers."""
    fallback = getattr(group, "logical_topk_indices", None)
    owner = getattr(group, "_topk_owner", None)
    if isinstance(owner, dict):
        caches = getattr(group, "caches", ())
        leader_cache = caches[0] if caches else None
        resolved = None
        for layer in owner.values():
            impl = getattr(layer, "impl", None)
            is_leader = (
                getattr(layer, "index_group", None) is group
                or getattr(impl, "index_group", None) is group
            ) and getattr(impl, "index_group_index", 0) == 0
            if (
                leader_cache is not None
                and getattr(layer, "hisparse_cache", None) is leader_cache
            ) or is_leader:
                resolved = layer
                group._topk_owner = layer
                break
        owner = resolved
    if owner is None:
        return fallback

    impl_buf = getattr(getattr(owner, "impl", None), "topk_indices_buffer", None)
    layer_buf = getattr(owner, "topk_indices_buffer", None)
    fallback_ptr = fallback.data_ptr() if fallback is not None else None
    if impl_buf is not None and (
        layer_buf is None or impl_buf.data_ptr() != fallback_ptr
    ):
        live = impl_buf
    elif layer_buf is not None:
        live = layer_buf
    else:
        return fallback
    if fallback is not None and live.shape[1] > fallback.shape[1]:
        return live[:, : fallback.shape[1]]
    return live


def _extract_group_prefill_topk(
    group: Any,
    attn_metadata: Any,
    *,
    num_prefill_tokens: int | None = None,
) -> torch.Tensor | None:
    """Slice `[num_prefill_tokens, top_k]` from the group's live top-k buffer."""
    if not getattr(group, "has_indexer", True):
        return None
    route = getattr(getattr(attn_metadata, "prefill", None), "route", None)
    if route is not None and getattr(route, "name", None) != "SPARSE_MQA":
        return None
    raw_topk = _resolve_group_live_topk(group)
    if raw_topk is None or attn_metadata is None:
        return None
    topk_start = int(getattr(attn_metadata, "topk_start", 0) or 0)
    start = topk_start + int(getattr(attn_metadata, "num_decode_tokens", 0))
    num_actual = getattr(attn_metadata, "num_actual_tokens", None)
    fp8_prefill = getattr(
        getattr(attn_metadata, "fp8_extra_metadata", None), "num_prefill_tokens", None
    )
    if num_actual is not None:
        end = topk_start + int(num_actual)
    elif num_prefill_tokens is not None:
        end = start + num_prefill_tokens
    elif fp8_prefill is not None:
        end = start + int(fp8_prefill)
    else:
        end = raw_topk.shape[0]
    if start < 0 or end <= start or end > raw_topk.shape[0]:
        return None
    if num_prefill_tokens is not None and (end - start) != num_prefill_tokens:
        return None
    return raw_topk[start:end]


def _drop_plan_cache(group: Any, plan: HiSparsePrefillStagingPlan) -> None:
    raw_cache = getattr(group, "_prefill_mask_cache", None)
    if isinstance(raw_cache, dict):
        raw_cache.pop((id(plan), plan.row_ids.data_ptr()), None)
        group._prefill_mask_cache = raw_cache or None
    else:
        group._prefill_mask_cache = None


def _resolve_prefill_entry(
    group: Any,
    layer_index: int,
    plan: HiSparsePrefillStagingPlan,
    attn_metadata: Any,
    *,
    req_ids: torch.Tensor | None = None,
    request_start: int = 0,
    batch_size: int | None = None,
) -> list[Any] | None:
    """Resolve `[key, mask, fp8_gpu_key, fp8_dev_rows]` bound to `(plan, group)`."""
    if not getattr(group, "sparse_prefill_staging", False):
        group._prefill_mask_cache = None
        return None

    expected_rows = req_ids.shape[0] if req_ids is not None else None
    logical_topk = _extract_group_prefill_topk(
        group, attn_metadata, num_prefill_tokens=expected_rows
    )
    if logical_topk is None:
        _drop_plan_cache(group, plan)
        return None

    plan_id = (id(plan), plan.row_ids.data_ptr())
    key = (
        plan.block_size,
        tuple(plan.block_table.shape),
        tuple(plan.row_ids.shape),
        logical_topk.data_ptr(),
        logical_topk.storage_offset(),
        tuple(logical_topk.shape),
        tuple(logical_topk.stride()),
    )
    raw_cache = getattr(group, "_prefill_mask_cache", None)
    cache_map: dict[tuple[int, int], list[Any]] = (
        raw_cache if isinstance(raw_cache, dict) else {}
    )
    entry: list[Any] | None = None
    matched_id = plan_id
    if layer_index > 0 or request_start > 0:
        cand = cache_map.get(plan_id)
        if cand is not None and cand[0] == key:
            entry = cand
        elif len(cache_map) == 1:
            only_id, only_cand = next(iter(cache_map.items()))
            if only_cand[0] == key:
                entry = only_cand
                matched_id = only_id

    if entry is None:
        if req_ids is None:
            req_ids = attn_metadata.req_id_per_token[attn_metadata.num_decode_tokens :]
            if attn_metadata.num_decodes > 0:
                req_ids = req_ids - attn_metadata.num_decodes
        if req_ids.shape[0] != logical_topk.shape[0]:
            _drop_plan_cache(group, plan)
            return None
        mask = hisparse_build_selected_mask(
            req_ids,
            plan.block_table,
            logical_topk,
            block_size=plan.block_size,
            num_staged_rows=plan.row_ids.shape[1],
        )
        entry = [key, mask, None, None]
        matched_id = plan_id

    is_last_chunk = (
        batch_size is None or request_start + batch_size >= plan.block_table.shape[0]
    )
    num_layers = getattr(group, "num_layers", 0) or len(getattr(group, "caches", ()))
    if is_last_chunk and layer_index >= max(num_layers - 1, 0):
        cache_map.pop(matched_id, None)
        group._prefill_mask_cache = cache_map or None
    else:
        if matched_id not in cache_map and len(cache_map) >= _MAX_CACHED_PLANS:
            cache_map.pop(next(iter(cache_map)))
        cache_map[matched_id] = entry
        group._prefill_mask_cache = cache_map
    return entry


def stage_sparse_prefill_rows(
    group: Any,
    layer_index: int,
    cache: HiSparseCacheHandle,
    kv_cache: torch.Tensor,
    plan: HiSparsePrefillStagingPlan,
    attn_metadata: Any,
    req_ids: torch.Tensor,
) -> torch.Tensor:
    """Stage only top-k selected KV rows, falling back to full staging if None."""
    entry = _resolve_prefill_entry(
        group, layer_index, plan, attn_metadata, req_ids=req_ids
    )
    if entry is None:
        return cache.stage_prefill_rows(kv_cache, plan, attn_metadata.num_decodes)
    resident_cache = None
    if cache.view is not None and cache.block_table is not None:
        cache.ensure_prefill_gpu_sources(plan, attn_metadata.num_decodes)
        resident_cache = cache.view.cache
    return gather_sparse_prefill_cache(
        cache.runtime.cache_ops,
        kv_cache,
        plan,
        resident_cache=resident_cache,
        selected_mask=entry[1],
    )


def select_fp8_sparse_prefill_rows(
    group: Any,
    layer_index: int,
    cache: HiSparseCacheHandle,
    plan: HiSparsePrefillStagingPlan,
    attn_metadata: Any,
    request_start: int,
    batch_size: int,
) -> torch.Tensor:
    """Redirect unselected FP8 prefill rows to dummy resident row `0`."""
    assert plan.gpu_row_ids is not None
    if cache.view is None or cache.view.cache.numel() <= 0:
        _drop_plan_cache(group, plan)
        return plan.gpu_row_ids
    entry = _resolve_prefill_entry(
        group,
        layer_index,
        plan,
        attn_metadata,
        request_start=request_start,
        batch_size=batch_size,
    )
    if entry is None:
        return plan.gpu_row_ids
    gpu_key = (
        id(plan.gpu_row_ids),
        plan.gpu_row_ids.data_ptr(),
        plan.gpu_source_key,
    )
    if entry[2] == gpu_key and entry[3] is not None:
        device_row_ids = entry[3]
    else:
        device_row_ids = torch.where(entry[1] != 0, plan.gpu_row_ids, 0)
        entry[2] = gpu_key
        entry[3] = device_row_ids
    prefill_stream = getattr(group, "prefill_stream", None)
    if device_row_ids.is_cuda and isinstance(prefill_stream, torch.Stream):
        device_row_ids.record_stream(prefill_stream)
    return device_row_ids
