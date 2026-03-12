# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.config import CacheConfig, CachePolicy, CopyMethod
from vllm.v1.attention.backends import sparse_select as sparse_select_ops
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.cache_policy import LRUCache, LRUWithHotCache, LayerWiseLRUCache


@dataclass
class PreparedLayerState:
    attn_metadata: Any
    slot_mapping: torch.Tensor | None
    swap_out_mapping: torch.Tensor
    layer_idx: int
    swap_in_event: torch.cuda.Event | None = None
    kv_update_event: torch.cuda.Event | None = None
    attn_forward_event: torch.cuda.Event | None = None


class GPUCacheManager:
    """Sparse hot-cache manager for vLLM 0.16.

    This manager keeps CPU full KV caches as logical main memory and per-layer
    GPU hot KV caches as physical working set. It rewrites attention metadata,
    schedules CPU<->GPU swaps, and updates block representations for sparse
    selection.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        layer_names: tuple[str, ...],
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
    ) -> None:
        self.cache_config = cache_config
        self.layer_names = layer_names
        self.layer_name_to_idx = {
            layer_name: idx for idx, layer_name in enumerate(layer_names)
        }
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.block_size = block_size
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.max_model_len = max_model_len
        self.max_blocks_per_req = math.ceil(max_model_len / block_size)

        if cache_config.copy_method in (CopyMethod.GATHER_SCATTER, CopyMethod.CUSTOM):
            raise NotImplementedError(
                f"copy_method={cache_config.copy_method.value} is not supported."
            )

        if cache_config.cache_policy == CachePolicy.LRU_LAYERWISE:
            self.cache = LayerWiseLRUCache(len(layer_names), num_gpu_blocks)
        elif cache_config.cache_policy == CachePolicy.LRU:
            self.cache = LRUCache(len(layer_names), num_gpu_blocks)
        elif cache_config.cache_policy == CachePolicy.LRU_WITH_HOT_SCORE:
            self.cache = LRUWithHotCache(len(layer_names), num_gpu_blocks)
        else:
            raise NotImplementedError(
                f"cache_policy={cache_config.cache_policy.value} is not supported."
            )

        self.gpu_kv_caches: dict[str, torch.Tensor] = {}
        self.cpu_kv_caches: dict[str, torch.Tensor] = {}
        self.layer_name_to_kv_cache_group: dict[str, int] = {}

        self.block_repr: torch.Tensor | None = None
        self.scores: torch.Tensor | None = None
        self.last_selected_scores: dict[str, torch.Tensor] = {}

        self.num_reqs = 0
        self.num_tokens = 0
        self.query_start_loc_cpu: torch.Tensor | None = None
        self.seq_lens_cpu: torch.Tensor | None = None
        self.block_tables_cpu_by_group: dict[int, torch.Tensor] = {}
        self.slot_mappings_cpu_by_group: dict[int, torch.Tensor] = {}
        self.prepared_layers: dict[str, PreparedLayerState] = {}

        self._warned_keys: set[str] = set()
        self._triton_sparse_available = bool(
            getattr(sparse_select_ops, "_HAS_TRITON", False)
        )

        self._device: torch.device | None = None
        self.kv_swap_in_stream: torch.cuda.Stream | None = None
        self.kv_swap_out_stream: torch.cuda.Stream | None = None
        self.layer_swap_out_events: dict[str, torch.cuda.Event] = {}
        self._async_copy_enabled = False

    @property
    def enabled(self) -> bool:
        return self.cache_config.enable_sparse_hot_cache

    def attach_caches(
        self,
        gpu_kv_caches: dict[str, torch.Tensor],
        cpu_kv_caches: dict[str, torch.Tensor],
        layer_name_to_kv_cache_group: dict[str, int],
    ) -> None:
        self.gpu_kv_caches = gpu_kv_caches
        self.cpu_kv_caches = cpu_kv_caches
        self.layer_name_to_kv_cache_group = layer_name_to_kv_cache_group

        if not self.gpu_kv_caches:
            return
        first_kv_cache = next(iter(self.gpu_kv_caches.values()))
        self._device = first_kv_cache.device

        if first_kv_cache.dim() >= 5 and first_kv_cache.shape[0] == 2:
            num_kv_heads = int(first_kv_cache.shape[-2])
            head_dim = int(first_kv_cache.shape[-1])
            self.block_repr = torch.zeros(
                (len(self.layer_names), self.num_cpu_blocks, num_kv_heads, head_dim),
                dtype=first_kv_cache.dtype,
                device=first_kv_cache.device,
            )
            self.scores = torch.empty(
                (self.max_num_reqs, self.max_blocks_per_req),
                dtype=torch.float32,
                device=first_kv_cache.device,
            )

        if first_kv_cache.is_cuda:
            self._async_copy_enabled = True
            self.kv_swap_in_stream = torch.cuda.Stream(device=first_kv_cache.device)
            self.kv_swap_out_stream = torch.cuda.Stream(device=first_kv_cache.device)
            current_stream = torch.cuda.current_stream(device=first_kv_cache.device)
            for layer_name in self.layer_names:
                event = torch.cuda.Event()
                event.record(current_stream)
                self.layer_swap_out_events[layer_name] = event

    def begin_batch(
        self,
        num_reqs: int,
        num_tokens: int,
        query_start_loc_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        block_tables_cpu_by_group: dict[int, torch.Tensor],
        slot_mappings_by_group: dict[int, torch.Tensor],
    ) -> None:
        self.num_reqs = num_reqs
        self.num_tokens = num_tokens
        self.query_start_loc_cpu = query_start_loc_cpu[: num_reqs + 1].detach().cpu()
        self.seq_lens_cpu = seq_lens_cpu[:num_reqs].detach().cpu()
        self.block_tables_cpu_by_group = block_tables_cpu_by_group
        self.slot_mappings_cpu_by_group = {
            gid: slot_mapping[:num_tokens].detach().cpu()
            for gid, slot_mapping in slot_mappings_by_group.items()
        }
        self.prepared_layers.clear()
        self.last_selected_scores.clear()
        self.cache.add_timer()

    def end_batch(self) -> None:
        self.prepared_layers.clear()
        self.num_reqs = 0
        self.num_tokens = 0
        self.query_start_loc_cpu = None
        self.seq_lens_cpu = None
        self.block_tables_cpu_by_group = {}
        self.slot_mappings_cpu_by_group = {}

    def prepare_layer(
        self,
        layer_name: str,
        attn_metadata: Any,
        slot_mapping: torch.Tensor | None,
        query: torch.Tensor | None = None,
    ) -> tuple[Any, torch.Tensor | None]:
        if not self.enabled or attn_metadata is None or slot_mapping is None:
            return attn_metadata, slot_mapping
        if self.query_start_loc_cpu is None or self.seq_lens_cpu is None:
            return attn_metadata, slot_mapping
        if not isinstance(attn_metadata, FlashAttentionMetadata):
            raise NotImplementedError(
                "Sparse hot-cache path only supports FlashAttention metadata."
            )

        layer_idx = self.layer_name_to_idx[layer_name]
        kv_cache_gid = self.layer_name_to_kv_cache_group[layer_name]
        original_block_table_cpu = self.block_tables_cpu_by_group[kv_cache_gid]
        original_slot_mapping_cpu = self.slot_mappings_cpu_by_group[kv_cache_gid]

        # Start from original block-table/slot semantics and rewrite only the
        # fields needed by sparse selection/remapping.
        new_block_table_cpu = torch.full_like(
            attn_metadata.block_table.detach().cpu(), -1
        )
        new_slot_mapping_cpu = torch.remainder(
            original_slot_mapping_cpu.clone(), self.block_size
        )
        new_seq_lens_cpu = attn_metadata.seq_lens.detach().cpu().clone()

        seq_infos: list[tuple[int, int, int, int, int, int, bool]] = []
        existing_full_blocks_by_req: list[int] = []
        for req_idx in range(self.num_reqs):
            query_start = int(self.query_start_loc_cpu[req_idx])
            query_end = int(self.query_start_loc_cpu[req_idx + 1])
            query_len = query_end - query_start
            seq_len = int(self.seq_lens_cpu[req_idx])
            kv_len = seq_len - query_len
            existing_full_blocks = kv_len // self.block_size
            full_blocks = seq_len // self.block_size
            has_partial_block = seq_len % self.block_size != 0
            seq_infos.append(
                (
                    query_start,
                    query_end,
                    seq_len,
                    kv_len,
                    existing_full_blocks,
                    full_blocks,
                    has_partial_block,
                )
            )
            existing_full_blocks_by_req.append(existing_full_blocks)

        selected_existing_by_req, selected_score_maps = self._select_existing_blocks(
            layer_idx=layer_idx,
            existing_full_blocks_by_req=existing_full_blocks_by_req,
            attn_metadata=attn_metadata,
            query=query,
        )

        swap_in_pairs: list[list[int]] = []
        swap_out_pairs: list[list[int]] = []
        max_seq_len = 0
        for req_idx, seq_info in enumerate(seq_infos):
            (
                query_start,
                query_end,
                seq_len,
                kv_len,
                existing_full_blocks,
                full_blocks,
                has_partial_block,
            ) = seq_info
            selected_existing = selected_existing_by_req[req_idx]
            selected_score_map = selected_score_maps[req_idx]

            next_block_pos = 0
            for logical_block_id in selected_existing:
                cpu_block_id = int(original_block_table_cpu[req_idx, logical_block_id])
                hot_score = float(selected_score_map.get(logical_block_id, 0.0))
                slot_id, hit = self.cache.get(layer_idx, cpu_block_id, hot_score)
                new_block_table_cpu[req_idx, next_block_pos] = slot_id
                next_block_pos += 1
                if not hit:
                    swap_in_pairs.append([cpu_block_id, slot_id])

            cursor = query_start

            # Newly completed full blocks are always selected.
            for logical_block_id in range(existing_full_blocks, full_blocks):
                cpu_block_id = int(original_block_table_cpu[req_idx, logical_block_id])
                slot_id, hit = self.cache.get(layer_idx, cpu_block_id, hot_score=1.0)
                self.cache.unpin_block(layer_idx, cpu_block_id)
                new_block_table_cpu[req_idx, next_block_pos] = slot_id
                next_block_pos += 1

                block_start = logical_block_id * self.block_size
                block_end = block_start + self.block_size
                existing_tokens_in_block = max(
                    0, min(kv_len, block_end) - block_start
                )
                if existing_tokens_in_block > 0 and not hit:
                    swap_in_pairs.append([cpu_block_id, slot_id])

                query_tokens_in_block = max(
                    0, min(seq_len, block_end) - max(kv_len, block_start)
                )
                if query_tokens_in_block > 0:
                    new_slot_mapping_cpu[cursor : cursor + query_tokens_in_block] += (
                        slot_id * self.block_size
                    )
                    cursor += query_tokens_in_block

                swap_out_pairs.append([slot_id, cpu_block_id])

            # Keep partial block selected and pinned.
            if has_partial_block:
                logical_block_id = full_blocks
                cpu_block_id = int(original_block_table_cpu[req_idx, logical_block_id])
                slot_id, hit = self.cache.get(layer_idx, cpu_block_id, hot_score=0.0)
                self.cache.pin_block(layer_idx, cpu_block_id)
                new_block_table_cpu[req_idx, next_block_pos] = slot_id
                next_block_pos += 1

                block_start = logical_block_id * self.block_size
                block_end = block_start + self.block_size
                existing_tokens_in_block = max(
                    0, min(kv_len, block_end) - block_start
                )
                if existing_tokens_in_block > 0 and not hit:
                    swap_in_pairs.append([cpu_block_id, slot_id])

                query_tokens_in_block = max(
                    0, min(seq_len, block_end) - max(kv_len, block_start)
                )
                if query_tokens_in_block > 0:
                    new_slot_mapping_cpu[cursor : cursor + query_tokens_in_block] += (
                        slot_id * self.block_size
                    )
                    cursor += query_tokens_in_block

            if cursor != query_end:
                raise RuntimeError(
                    f"Sparse hot-cache remap mismatch for layer {layer_name}: "
                    f"cursor={cursor}, query_end={query_end}."
                )

            dropped_existing_blocks = existing_full_blocks - len(selected_existing)
            new_seq_lens_cpu[req_idx] = seq_len - dropped_existing_blocks * self.block_size
            max_seq_len = max(max_seq_len, int(new_seq_lens_cpu[req_idx]))

        swap_in_mapping = self._to_mapping_tensor(swap_in_pairs)
        swap_out_mapping = self._to_mapping_tensor(swap_out_pairs)

        swap_in_event: torch.cuda.Event | None = None
        if swap_in_mapping.numel() > 0:
            if self._async_copy_enabled and self.kv_swap_in_stream is not None:
                swap_in_event = torch.cuda.Event()
                with torch.cuda.stream(self.kv_swap_in_stream):
                    last_swap_out_event = self.layer_swap_out_events.get(layer_name)
                    if last_swap_out_event is not None:
                        self.kv_swap_in_stream.wait_event(last_swap_out_event)
                    self._swap_blocks(
                        self.cpu_kv_caches[layer_name],
                        self.gpu_kv_caches[layer_name],
                        swap_in_mapping,
                    )
                    swap_in_event.record(self.kv_swap_in_stream)
            else:
                self._swap_blocks(
                    self.cpu_kv_caches[layer_name],
                    self.gpu_kv_caches[layer_name],
                    swap_in_mapping,
                )

        new_attn_metadata = copy.copy(attn_metadata)
        new_attn_metadata.block_table = new_block_table_cpu.to(
            device=attn_metadata.block_table.device,
            non_blocking=self._async_copy_enabled,
        )
        new_attn_metadata.slot_mapping = new_slot_mapping_cpu.to(
            device=slot_mapping.device,
            non_blocking=self._async_copy_enabled,
        )
        new_attn_metadata.seq_lens = new_seq_lens_cpu.to(
            device=attn_metadata.seq_lens.device,
            non_blocking=self._async_copy_enabled,
        )
        new_attn_metadata.max_seq_len = max_seq_len

        prepared = PreparedLayerState(
            attn_metadata=new_attn_metadata,
            slot_mapping=new_attn_metadata.slot_mapping,
            swap_out_mapping=swap_out_mapping,
            layer_idx=layer_idx,
            swap_in_event=swap_in_event,
            kv_update_event=(
                torch.cuda.Event() if self._async_copy_enabled else None
            ),
            attn_forward_event=(
                torch.cuda.Event() if self._async_copy_enabled else None
            ),
        )
        self.prepared_layers[layer_name] = prepared
        return prepared.attn_metadata, prepared.slot_mapping

    def get_prepared_attn_metadata(self, layer_name: str, attn_metadata: Any) -> Any:
        state = self.prepared_layers.get(layer_name)
        if state is None:
            return attn_metadata
        return state.attn_metadata

    def wait_for_swap_in(self, layer_name: str) -> None:
        state = self.prepared_layers.get(layer_name)
        if state is None or state.swap_in_event is None:
            return
        if self._device is None:
            return
        torch.cuda.current_stream(device=self._device).wait_event(state.swap_in_event)

    def mark_kv_update_done(self, layer_name: str) -> None:
        state = self.prepared_layers.get(layer_name)
        if state is None or state.kv_update_event is None or self._device is None:
            return
        state.kv_update_event.record(torch.cuda.current_stream(device=self._device))

    def mark_attn_forward_done(self, layer_name: str) -> None:
        state = self.prepared_layers.get(layer_name)
        if state is None or state.attn_forward_event is None or self._device is None:
            return
        state.attn_forward_event.record(torch.cuda.current_stream(device=self._device))

    def finish_layer(self, layer_name: str, skip_swap_out: bool = False) -> None:
        state = self.prepared_layers.pop(layer_name, None)
        if state is None:
            return
        if skip_swap_out:
            return
        if state.swap_out_mapping.numel() == 0:
            return

        if self._async_copy_enabled and self.kv_swap_out_stream is not None:
            with torch.cuda.stream(self.kv_swap_out_stream):
                if state.kv_update_event is not None:
                    self.kv_swap_out_stream.wait_event(state.kv_update_event)
                if state.attn_forward_event is not None:
                    self.kv_swap_out_stream.wait_event(state.attn_forward_event)
                self._swap_blocks(
                    self.gpu_kv_caches[layer_name],
                    self.cpu_kv_caches[layer_name],
                    state.swap_out_mapping,
                )
                self._update_block_repr(
                    layer_name, state.layer_idx, state.swap_out_mapping
                )
                done_event = torch.cuda.Event()
                done_event.record(self.kv_swap_out_stream)
                self.layer_swap_out_events[layer_name] = done_event
        else:
            self._swap_blocks(
                self.gpu_kv_caches[layer_name],
                self.cpu_kv_caches[layer_name],
                state.swap_out_mapping,
            )
            self._update_block_repr(layer_name, state.layer_idx, state.swap_out_mapping)

    def _select_existing_blocks(
        self,
        layer_idx: int,
        existing_full_blocks_by_req: list[int],
        attn_metadata: FlashAttentionMetadata,
        query: torch.Tensor | None,
    ) -> tuple[list[list[int]], list[dict[int, float]]]:
        local_selected = [
            self._select_existing_full_blocks_local(existing_full_blocks)
            for existing_full_blocks in existing_full_blocks_by_req
        ]
        local_scores = [{idx: 0.0 for idx in selected} for selected in local_selected]

        if self.cache_config.sparse_topk is None:
            return local_selected, local_scores

        topk_blocks = self.cache_config.sparse_topk // self.block_size
        if topk_blocks <= 0:
            raise ValueError("sparse_topk must be at least one KV block.")
        max_existing_full_blocks = max(existing_full_blocks_by_req, default=0)
        if max_existing_full_blocks <= topk_blocks:
            return local_selected, local_scores

        if (
            not self._triton_sparse_available
            or query is None
            or self.block_repr is None
            or self.scores is None
        ):
            self._warn_once(
                "sparse_select_fallback_local",
                "Sparse Triton kernels are unavailable; falling back to local "
                "head-tail selection.",
            )
            return local_selected, local_scores

        try:
            query_for_select = query.narrow(0, 0, int(attn_metadata.num_actual_tokens))
            if (
                not query_for_select.is_cuda
                or not attn_metadata.block_table.is_cuda
                or not attn_metadata.seq_lens.is_cuda
                or not attn_metadata.query_start_loc.is_cuda
            ):
                self._warn_once(
                    "sparse_select_device_fallback_local",
                    "Sparse Triton kernels require CUDA tensors; using local "
                    "head-tail selection.",
                )
                return local_selected, local_scores

            scores_view = self.scores[: self.num_reqs, :max_existing_full_blocks]
            scores_view.fill_(float("-inf"))

            topk_choices, topk_scores = sparse_select_ops.sparse_kv_selection(
                block_table=attn_metadata.block_table,
                batch_size=self.num_reqs,
                block_size=self.block_size,
                max_num_blocks_this_batch=max_existing_full_blocks,
                seq_lens=attn_metadata.seq_lens,
                k_repr=self.block_repr[layer_idx],
                query=query_for_select,
                query_start_loc=attn_metadata.query_start_loc,
                top_k=topk_blocks,
                scores=scores_view,
            )

            selected_by_req: list[list[int]] = [[] for _ in range(self.num_reqs)]
            score_by_req: list[dict[int, float]] = [{} for _ in range(self.num_reqs)]
            topk_choices_cpu = topk_choices.detach().cpu()
            topk_scores_cpu = topk_scores.detach().cpu().to(torch.float32)

            for req_idx in range(self.num_reqs):
                existing_full_blocks = existing_full_blocks_by_req[req_idx]
                if existing_full_blocks <= 0:
                    continue
                score_map: dict[int, float] = {}
                for j in range(topk_choices_cpu.shape[1]):
                    logical_block_id = int(topk_choices_cpu[req_idx, j])
                    if logical_block_id < 0 or logical_block_id >= existing_full_blocks:
                        continue
                    score = float(topk_scores_cpu[req_idx, j])
                    prev = score_map.get(logical_block_id)
                    if prev is None or score > prev:
                        score_map[logical_block_id] = score
                if not score_map:
                    selected = self._select_existing_full_blocks_local(existing_full_blocks)
                    score_map = {idx: 0.0 for idx in selected}
                selected_by_req[req_idx] = sorted(score_map.keys())
                score_by_req[req_idx] = score_map

            return selected_by_req, score_by_req
        except Exception as exc:
            self._warn_once(
                "sparse_select_exception_fallback_local",
                f"Sparse Triton kernels failed ({exc}); falling back to local "
                "head-tail selection.",
            )
            return local_selected, local_scores

    def _select_existing_full_blocks_local(self, existing_full_blocks: int) -> list[int]:
        if existing_full_blocks <= 0:
            return []
        if self.cache_config.sparse_topk is None:
            return list(range(existing_full_blocks))
        topk_blocks = self.cache_config.sparse_topk // self.block_size
        if topk_blocks <= 0:
            raise ValueError("sparse_topk must be at least one KV block.")
        if topk_blocks >= existing_full_blocks:
            return list(range(existing_full_blocks))
        head_blocks = [0]
        tail_count = max(topk_blocks - len(head_blocks), 0)
        tail_start = max(existing_full_blocks - tail_count, 0)
        selected = set(head_blocks)
        selected.update(range(tail_start, existing_full_blocks))
        return sorted(selected)

    @staticmethod
    def _to_mapping_tensor(mapping: list[list[int]]) -> torch.Tensor:
        if not mapping:
            return torch.empty((0, 2), dtype=torch.int64, device="cpu")
        return torch.tensor(mapping, dtype=torch.int64, device="cpu")

    @staticmethod
    def _kv_planes(
        src_kv_cache: torch.Tensor, dst_kv_cache: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if (
            src_kv_cache.dim() >= 2
            and dst_kv_cache.dim() >= 2
            and src_kv_cache.shape[0] == 2
            and dst_kv_cache.shape[0] == 2
        ):
            return ((src_kv_cache[0], dst_kv_cache[0]), (src_kv_cache[1], dst_kv_cache[1]))
        return ((src_kv_cache, dst_kv_cache),)

    def _swap_blocks(
        self,
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        block_mapping: torch.Tensor,
    ) -> None:
        if block_mapping.numel() == 0:
            return

        copy_method = self.cache_config.copy_method
        if copy_method == CopyMethod.TORCH:
            self._swap_blocks_torch(src_kv_cache, dst_kv_cache, block_mapping)
            return
        if copy_method == CopyMethod.NON_MERGED:
            self._swap_blocks_non_merged(src_kv_cache, dst_kv_cache, block_mapping)
            return
        if copy_method == CopyMethod.MERGED:
            self._swap_blocks_merged(src_kv_cache, dst_kv_cache, block_mapping)
            return
        raise NotImplementedError(
            f"copy_method={self.cache_config.copy_method.value} is not supported."
        )

    def _swap_blocks_torch(
        self,
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        block_mapping: torch.Tensor,
    ) -> None:
        mappings = block_mapping.tolist()
        for src_plane, dst_plane in self._kv_planes(src_kv_cache, dst_kv_cache):
            for src_idx, dst_idx in mappings:
                dst_plane[dst_idx].copy_(
                    src_plane[src_idx], non_blocking=self._async_copy_enabled
                )

    def _swap_blocks_non_merged(
        self,
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        block_mapping: torch.Tensor,
    ) -> None:
        for src_plane, dst_plane in self._kv_planes(src_kv_cache, dst_kv_cache):
            block_size_in_bytes = int(src_plane.stride(0) * src_plane.element_size())
            ops.swap_blocks(src_plane, dst_plane, block_size_in_bytes, block_mapping)

    def _swap_blocks_merged(
        self,
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        block_mapping: torch.Tensor,
    ) -> None:
        # Preferred path: use built-in merged op if available.
        # Fallback path: pure python contiguous-range merged copies, so users can
        # patch only python code without rebuilding C++ extensions.
        if hasattr(ops, "swap_blocks_merged"):
            try:
                for src_plane, dst_plane in self._kv_planes(src_kv_cache, dst_kv_cache):
                    ops.swap_blocks_merged(src_plane, dst_plane, block_mapping)
                return
            except Exception as exc:
                self._warn_once(
                    "merged_op_fallback_python",
                    f"swap_blocks_merged failed ({exc}); using python merged copy.",
                )
        else:
            self._warn_once(
                "merged_op_missing_python",
                "swap_blocks_merged is unavailable; using python merged copy.",
            )

        self._swap_blocks_merged_python(src_kv_cache, dst_kv_cache, block_mapping)

    def _swap_blocks_merged_python(
        self,
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        block_mapping: torch.Tensor,
    ) -> None:
        if block_mapping.numel() == 0:
            return

        if block_mapping.device.type != "cpu":
            mapping_cpu = block_mapping.cpu()
        else:
            mapping_cpu = block_mapping

        pairs = sorted(
            ((int(src), int(dst)) for src, dst in mapping_cpu.tolist()),
            key=lambda x: (x[0], x[1]),
        )
        if not pairs:
            return

        runs: list[tuple[int, int, int]] = []
        run_src, run_dst = pairs[0]
        run_len = 1
        for src, dst in pairs[1:]:
            if src == run_src + run_len and dst == run_dst + run_len:
                run_len += 1
                continue
            runs.append((run_src, run_dst, run_len))
            run_src, run_dst, run_len = src, dst, 1
        runs.append((run_src, run_dst, run_len))

        for src_plane, dst_plane in self._kv_planes(src_kv_cache, dst_kv_cache):
            for src_start, dst_start, run_length in runs:
                dst_plane.narrow(0, dst_start, run_length).copy_(
                    src_plane.narrow(0, src_start, run_length),
                    non_blocking=self._async_copy_enabled,
                )

    def _update_block_repr(
        self, layer_name: str, layer_idx: int, swap_out_mapping: torch.Tensor
    ) -> None:
        if (
            swap_out_mapping.numel() == 0
            or self.block_repr is None
            or layer_name not in self.gpu_kv_caches
        ):
            return

        gpu_kv_cache = self.gpu_kv_caches[layer_name]
        mapping = swap_out_mapping
        if mapping.device.type == "cpu" and gpu_kv_cache.is_cuda:
            mapping = mapping.to(
                device=gpu_kv_cache.device, non_blocking=self._async_copy_enabled
            )
        try:
            sparse_select_ops.kv_repr_gen(
                kv_cache=gpu_kv_cache,
                block_repr=self.block_repr[layer_idx],
                mapping=mapping,
                num_mappings=swap_out_mapping.shape[0],
                block_size=self.block_size,
                num_kv_heads=int(self.block_repr.shape[-2]),
                head_dim=int(self.block_repr.shape[-1]),
            )
        except Exception as exc:
            self._warn_once(
                "repr_update_fallback",
                f"kv_repr_gen failed ({exc}); skipping block representation update.",
            )

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_keys:
            return
        warnings.warn(message, stacklevel=2)
        self._warned_keys.add(key)
