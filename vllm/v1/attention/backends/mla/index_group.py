# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.hisparse.runtime import (
    FP8_DS_MLA_ROW_BYTES,
    HiSparseCacheHandle,
    HiSparsePrefillStagingPlan,
    create_hisparse_cache_handle,
)


def _create_side_stream(device: torch.device) -> torch.Stream:
    return torch.Stream(device=device)


def _create_event() -> torch.Event:
    return torch.Event()


@dataclass
class SparseMLAIndexGroup:
    """Layers that consume one sparse-indexer result."""

    logical_topk_indices: torch.Tensor
    physical_topk_indices: torch.Tensor
    valid_topk_counts: torch.Tensor
    request_ids: torch.Tensor
    side_stream: torch.Stream
    logical_topk_ready: torch.Event
    physical_topk_ready: torch.Event
    has_indexer: bool
    num_layers: int = 0

    def register_layer(
        self,
        vllm_config: VllmConfig | None = None,
        *,
        head_size: int | None = None,
        kv_cache_dtype: str | None = None,
    ) -> int:
        layer_index = self.num_layers
        self.num_layers += 1
        return layer_index

    def set_logical_topk_ready(self, layer_index: int) -> None:
        if layer_index == 0 and self.has_indexer:
            self.logical_topk_ready.record(current_stream())

    def prepare_for_batch(self, layer_index: int, attn_metadata: Any | None) -> None:
        pass

    def gather_fp8_prefill(
        self,
        layer_index: int,
        source_cache: torch.Tensor,
        dst: torch.Tensor,
        block_table: torch.Tensor,
        workspace_starts: torch.Tensor,
        batch_size: int,
        attn_metadata: Any,
        request_start: int,
    ) -> torch.Event | None:
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            source_cache,
            dst,
            block_table,
            workspace_starts,
            batch_size,
        )
        return None

    def convert_logical_to_physical_topk(
        self,
        layer_index: int,
        logical_topk_indices: torch.Tensor,
        attn_metadata: Any,
        *,
        block_stride_rows: int | None,
        return_valid_counts: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Convert one index result once for all layers in this group."""
        return self._convert_once(
            layer_index,
            logical_topk_indices,
            attn_metadata.req_id_per_token[: logical_topk_indices.shape[0]],
            attn_metadata.block_table,
            attn_metadata.block_size,
            block_stride_rows=block_stride_rows,
            return_valid_counts=return_valid_counts,
        )

    def _convert_once(
        self,
        layer_index: int,
        logical_topk_indices: torch.Tensor,
        req_id_per_token: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        *,
        block_stride_rows: int | None,
        return_valid_counts: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        num_tokens = logical_topk_indices.shape[0]
        if num_tokens > self.physical_topk_indices.shape[0]:
            return triton_convert_req_index_to_global_index(
                req_id_per_token,
                block_table,
                logical_topk_indices,
                BLOCK_SIZE=block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=logical_topk_indices.shape[1],
                return_valid_counts=return_valid_counts,
            )

        physical_topk_indices = self.physical_topk_indices[:num_tokens]
        valid_topk_counts = self.valid_topk_counts[:num_tokens]
        compute_stream = current_stream()
        if layer_index == 0:
            if self.has_indexer:
                self.side_stream.wait_event(self.logical_topk_ready)
            else:
                self.side_stream.wait_stream(compute_stream)
            with self.side_stream:
                triton_convert_req_index_to_global_index(
                    req_id_per_token,
                    block_table,
                    logical_topk_indices,
                    BLOCK_SIZE=block_size,
                    BLOCK_STRIDE_ROWS=block_stride_rows,
                    NUM_TOPK_TOKENS=logical_topk_indices.shape[1],
                    return_valid_counts=return_valid_counts,
                    out=physical_topk_indices,
                    valid_counts_out=(
                        valid_topk_counts if return_valid_counts else None
                    ),
                )
                self.physical_topk_ready.record(self.side_stream)
        compute_stream.wait_event(self.physical_topk_ready)

        if return_valid_counts:
            return physical_topk_indices, valid_topk_counts
        return physical_topk_indices


@dataclass
class HiSparseMLAIndexGroup(SparseMLAIndexGroup):
    """Sparse index group with per-layer host-backed KV caches."""

    caches: list[HiSparseCacheHandle] = field(default_factory=list)
    hisparse_group: Any | None = None
    prefill_stream: torch.Stream | None = None
    prefill_ready_events: list[torch.Event] = field(default_factory=list)

    def register_layer(
        self,
        vllm_config: VllmConfig | None = None,
        *,
        head_size: int | None = None,
        kv_cache_dtype: str | None = None,
    ) -> int:
        assert vllm_config is not None
        assert head_size is not None
        assert kv_cache_dtype is not None
        layer_index = super().register_layer()
        if self.prefill_stream is None:
            self.prefill_stream = _create_side_stream(self.logical_topk_indices.device)
        if kv_cache_dtype == "fp8_ds_mla":
            row_width = FP8_DS_MLA_ROW_BYTES
            kv_dtype = torch.uint8
        else:
            from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

            row_width = head_size
            kv_dtype = kv_cache_dtype_str_to_dtype(
                kv_cache_dtype, vllm_config.model_config
            )
        cache = create_hisparse_cache_handle(
            vllm_config,
            self.logical_topk_indices.shape[1],
            is_index_group_leader=layer_index == 0,
            row_width=row_width,
            kv_dtype=kv_dtype,
            index_group=self,
        )
        assert cache is not None
        self.caches.append(cache)
        cache.index_group_caches = self.caches
        self.prefill_ready_events.append(_create_event())
        return layer_index

    def cache(self, layer_index: int) -> HiSparseCacheHandle:
        return self.caches[layer_index]

    def physical_kv_cache(self, layer_index: int) -> torch.Tensor:
        cache = self.cache(layer_index)
        return cache.runtime.hot.attention_cache

    def prepare_for_batch(self, layer_index: int, attn_metadata: Any | None) -> None:
        if layer_index == 0:
            self.cache(0).prepare_group_for_batch(attn_metadata)

    def convert_logical_to_physical_topk(
        self,
        layer_index: int,
        logical_topk_indices: torch.Tensor,
        attn_metadata: Any,
        *,
        block_stride_rows: int | None,
        return_valid_counts: bool,
        req_id_per_token: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        cache = self.cache(layer_index)
        num_tokens = logical_topk_indices.shape[0]
        if req_id_per_token is None:
            req_id_per_token = attn_metadata.req_id_per_token[:num_tokens]
        if num_tokens > self.physical_topk_indices.shape[0]:
            # Prefill-sized batches do not fit the decode residency workspace.
            # Non-resident prefills are staged before reaching this path.
            assert cache.all_context_pages_resident
            leader = self.cache(0)
            assert leader.view is not None and leader.block_table is not None
            return self._convert_once(
                layer_index,
                logical_topk_indices,
                req_id_per_token,
                leader.block_table,
                leader.view.block_size,
                block_stride_rows=leader.view.attention_block_stride,
                return_valid_counts=return_valid_counts,
            )
        source_block_table = cache.source_block_table
        assert source_block_table is not None
        return cache.swap_in(
            req_id_per_token,
            block_table=source_block_table,
            logical_topk_indices=logical_topk_indices,
            block_size=attn_metadata.block_size,
            return_valid_counts=return_valid_counts,
        )

    def convert_decode_logical_to_physical_topk(
        self,
        layer_index: int,
        logical_topk_indices: torch.Tensor,
        attn_metadata: Any,
        *,
        return_valid_counts: bool,
        num_decodes: int | None = None,
        decode_query_len: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        num_tokens = logical_topk_indices.shape[0]
        if num_decodes is None:
            num_decodes = attn_metadata.num_decodes
        if decode_query_len is None:
            decode_query_len = attn_metadata.decode_max_query_len
        assert logical_topk_indices.shape[0] == num_tokens
        if decode_query_len == 1:
            return self.convert_logical_to_physical_topk(
                layer_index,
                logical_topk_indices,
                attn_metadata,
                block_stride_rows=None,
                return_valid_counts=return_valid_counts,
                req_id_per_token=self.request_ids[:num_decodes],
            )

        assert num_tokens == num_decodes * decode_query_len
        logical_topk_by_request = logical_topk_indices.view(
            num_decodes, decode_query_len, -1
        )
        physical_topk_by_request = self.physical_topk_indices[:num_tokens].view(
            num_decodes, decode_query_len, -1
        )
        valid_topk_by_request = self.valid_topk_counts[:num_tokens].view(
            num_decodes, decode_query_len
        )
        request_ids = self.request_ids[:num_decodes]
        for step in range(decode_query_len):
            cache = self.cache(layer_index)
            source_block_table = cache.source_block_table
            assert source_block_table is not None
            cache.swap_in(
                request_ids,
                block_table=source_block_table,
                logical_topk_indices=logical_topk_by_request[:, step],
                block_size=attn_metadata.block_size,
                return_valid_counts=return_valid_counts,
                attention_indices_out=(
                    physical_topk_by_request[:, step] if layer_index == 0 else None
                ),
                valid_counts_out=(
                    valid_topk_by_request[:, step]
                    if layer_index == 0 and return_valid_counts
                    else None
                ),
            )
        physical_topk_indices = physical_topk_by_request.view(num_tokens, -1)
        if return_valid_counts:
            return physical_topk_indices, valid_topk_by_request.view(num_tokens)
        return physical_topk_indices

    def stage_prefill_rows(
        self, layer_index: int, kv_cache: torch.Tensor, attn_metadata: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self.cache(layer_index)
        prefill = attn_metadata.prefill
        staging_plan = prefill.host_staging_plan if prefill is not None else None
        assert staging_plan is not None
        resident_cache = None
        if cache.view is not None and cache.block_table is not None:
            staging_plan.ensure_gpu_sources(
                cache.block_table[attn_metadata.num_decodes :],
                cache.view.block_size,
            )
            resident_cache = cache.view.cache
        staged_cache = cache.runtime.gather_prefill_cache(
            kv_cache,
            staging_plan,
            resident_cache=resident_cache,
        )
        req_ids = attn_metadata.req_id_per_token[attn_metadata.num_decode_tokens :]
        if attn_metadata.num_decodes > 0:
            req_ids = req_ids - attn_metadata.num_decodes
        return staged_cache, staging_plan.block_table, req_ids

    def _prefill_gather_plan(
        self, layer_index: int, attn_metadata: Any
    ) -> HiSparsePrefillStagingPlan:
        cache = self.cache(layer_index)
        prefill = attn_metadata.prefill
        plan = prefill.host_staging_plan if prefill is not None else None
        assert plan is not None
        assert cache.view is not None and cache.block_table is not None
        plan.ensure_gpu_sources(
            cache.block_table[attn_metadata.num_decodes :],
            cache.view.block_size,
        )
        assert plan.gpu_row_ids is not None
        return plan

    def gather_fp8_prefill(
        self,
        layer_index: int,
        source_cache: torch.Tensor,
        dst: torch.Tensor,
        block_table: torch.Tensor,
        workspace_starts: torch.Tensor,
        batch_size: int,
        attn_metadata: Any,
        request_start: int,
    ) -> torch.Event | None:
        cache = self.cache(layer_index)
        plan = self._prefill_gather_plan(layer_index, attn_metadata)
        assert cache.view is not None and plan.gpu_row_ids is not None
        row_width = source_cache.shape[-1]
        prefill_stream = self.prefill_stream
        assert prefill_stream is not None
        block_table = plan.block_table[request_start : request_start + batch_size]
        prefill_stream.wait_stream(current_stream())
        with prefill_stream:
            ops.cp_gather_and_upconvert_fp8_kv_cache(
                cache.view.cache,
                dst,
                block_table,
                workspace_starts,
                batch_size,
                host_cache=source_cache.view(-1, row_width),
                host_row_ids=plan.row_ids,
                device_row_ids=plan.gpu_row_ids,
            )
            ready = self.prefill_ready_events[layer_index]
            ready.record(prefill_stream)
        return ready


class SparseMLAIndexGroupBuilder:
    """Assign sparse MLA layers to their index-producing layer."""

    def __init__(
        self, logical_topk_indices: torch.Tensor, max_decode_rows: int | None = None
    ) -> None:
        self.logical_topk_indices = logical_topk_indices
        self.max_decode_rows = (
            logical_topk_indices.shape[0]
            if max_decode_rows is None
            else max_decode_rows
        )
        self.current_group: SparseMLAIndexGroup | None = None

    def register_layer(
        self,
        is_index_producing_layer: bool,
        vllm_config: VllmConfig | None = None,
        *,
        head_size: int | None = None,
        kv_cache_dtype: str | None = None,
    ) -> tuple[SparseMLAIndexGroup, int]:
        if is_index_producing_layer or self.current_group is None:
            group_cls = (
                HiSparseMLAIndexGroup
                if vllm_config is not None
                and vllm_config.attention_config.hisparse_config is not None
                else SparseMLAIndexGroup
            )
            workspace_rows = self.max_decode_rows
            physical_topk_indices = torch.empty(
                (workspace_rows, self.logical_topk_indices.shape[1]),
                dtype=self.logical_topk_indices.dtype,
                device=self.logical_topk_indices.device,
            )
            self.current_group = group_cls(
                logical_topk_indices=self.logical_topk_indices,
                physical_topk_indices=physical_topk_indices,
                valid_topk_counts=torch.empty(
                    workspace_rows,
                    dtype=torch.int32,
                    device=self.logical_topk_indices.device,
                ),
                request_ids=torch.arange(
                    workspace_rows,
                    dtype=torch.int32,
                    device=self.logical_topk_indices.device,
                ),
                side_stream=_create_side_stream(self.logical_topk_indices.device),
                logical_topk_ready=_create_event(),
                physical_topk_ready=_create_event(),
                has_indexer=is_index_producing_layer,
            )
        group = self.current_group
        group_index = group.register_layer(
            vllm_config,
            head_size=head_size,
            kv_cache_dtype=kv_cache_dtype,
        )
        return group, group_index


def get_sparse_mla_index_group_max_rows(vllm_config: VllmConfig) -> int:
    max_query_len = 1
    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is not None
        and speculative_config.num_speculative_tokens is not None
    ):
        max_query_len += speculative_config.num_speculative_tokens * (
            2 if speculative_config.parallel_drafting else 1
        )
    scheduler_config = vllm_config.scheduler_config
    return min(
        scheduler_config.max_num_batched_tokens,
        scheduler_config.max_num_seqs * max_query_len,
    )
