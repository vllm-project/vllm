# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from LongSpark at commit 9ab6526e481bbea67489f3a73ebc782218d56f71:
# https://github.com/Hao-Yuan-He/LongSpark/tree/9ab6526e481bbea67489f3a73ebc782218d56f71/vendor/sglang/python/sglang/srt/speculative
#
# MIT License
# Copyright (c) 2026 The LongSpark Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Native in-process LongSpark speculator for Model Runner V2.

The target owns the only persistent K/V cache. This speculator reads selected
target layers through their paged FlashAttention cache and maintains the
checkpoint-configured FP32 global-memory state per request slot.
"""

from typing import Any

import numpy as np
import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.longspark.cudagraph import (
    LongSparkCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.longspark.kernels import (
    fused_add_argmax,
    fused_markov_argmax,
    fused_markov_refined_argmax,
    update_longspark_global_delta,
    update_longspark_global_delta_batched,
)
from vllm.v1.worker.gpu.spec_decode.longspark.state import (
    merge_normalized_attention,
    normalized_gqa_attention,
    normalized_gqa_attention_batched,
)
from vllm.v1.worker.gpu.spec_decode.longspark.utils import (
    load_longspark_model,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class LongSparkSpeculator(DraftModelSpeculator):
    """Shape-parameterized DraftFreeKV global/raw/local proposer."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        self._validate_runtime_config(vllm_config)

        self.target_layer_ids = tuple(int(i) for i in config.target_layer_ids)
        self.num_selected_layers = len(self.target_layer_ids)
        self.num_attention_heads = int(config.qwen_num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.global_slots = int(config.global_memory_slots)
        self.raw_rows = int(config.raw_target_rows)
        self.block_size = int(config.block_size)
        self.max_accepted_tokens = int(config.max_accepted_tokens)
        self.num_query_per_req = self.block_size
        self._global_scan_chunk_size = 2048
        self.padding_state_slot = self.max_num_reqs

        state_shape = (
            self.max_num_reqs + 1,
            self.num_selected_layers,
            self.num_attention_heads,
            self.global_slots,
            self.head_dim,
        )
        self.global_output = torch.zeros(
            state_shape, dtype=torch.float32, device=device
        )
        self.global_lse = torch.zeros(
            state_shape[:-1], dtype=torch.float32, device=device
        )
        self._state_owner: list[str | None] = [None] * self.max_num_reqs
        self._state_committed_len = np.full(self.max_num_reqs, -1, dtype=np.int64)
        # Keep the authoritative decode length on device.  Reading rejected
        # draft counts back to the host here serializes every target/draft
        # iteration and is particularly expensive for small decode batches.
        self._state_committed_lens_gpu = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=device
        )
        self._batch_indices = torch.arange(
            self.max_num_reqs, dtype=torch.long, device=device
        )
        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )
        markov_chunks = (self.vocab_size + 8191) // 8192
        self._markov_partial_values = torch.empty(
            self.max_num_reqs,
            markov_chunks,
            dtype=torch.float32,
            device=device,
        )
        self._markov_partial_ids = torch.empty(
            self.max_num_reqs,
            markov_chunks,
            dtype=torch.int32,
            device=device,
        )
        markov_projection_chunks = (self.vocab_size + 31) // 32
        self._markov_projection_partial_values = torch.empty(
            self.max_num_reqs,
            markov_projection_chunks,
            dtype=torch.float32,
            device=device,
        )
        self._markov_projection_partial_ids = torch.empty(
            self.max_num_reqs,
            markov_projection_chunks,
            dtype=torch.int32,
            device=device,
        )
        self._markov_candidate_top_k = 32
        markov_candidate_tiles = (markov_projection_chunks + 511) // 512
        self._markov_candidate_tile_values = torch.empty(
            self.max_num_reqs,
            markov_candidate_tiles,
            self._markov_candidate_top_k,
            dtype=torch.float32,
            device=device,
        )
        self._markov_candidate_tile_ids = torch.empty(
            self.max_num_reqs,
            markov_candidate_tiles,
            self._markov_candidate_top_k,
            dtype=torch.int32,
            device=device,
        )
        self._markov_candidate_ids = torch.empty(
            self.max_num_reqs,
            self._markov_candidate_top_k,
            dtype=torch.int32,
            device=device,
        )
        self._fused_markov_weights: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
            | None
        ) = None

        self.target_attn_layers: list[AttentionLayerBase] = []
        self.target_attn_layer_names: list[str] = []
        self.target_block_tables: tuple[torch.Tensor, ...] = ()
        self._target_cache_addresses: tuple[int, ...] = ()
        self._target_block_table_addresses: tuple[int, ...] = ()
        self._target_cache_ptrs: torch.Tensor | None = None
        self._target_block_table_ptrs: torch.Tensor | None = None
        self.num_full_scans = 0
        self.num_delta_updates = 0
        self.num_delta_rows = 0
        self.longspark_cudagraph_manager: LongSparkCudaGraphManager | None = None
        self._global_update_stream = torch.cuda.Stream(device=device)

    def _validate_runtime_config(self, vllm_config: VllmConfig) -> None:
        config = self.draft_model_config.hf_config
        parallel = vllm_config.parallel_config
        unsupported_parallel = {
            "tensor_parallel_size": parallel.tensor_parallel_size,
            "pipeline_parallel_size": parallel.pipeline_parallel_size,
            "decode_context_parallel_size": parallel.decode_context_parallel_size,
            "prefill_context_parallel_size": parallel.prefill_context_parallel_size,
        }
        invalid = [name for name, value in unsupported_parallel.items() if value != 1]
        if invalid:
            raise ValueError(
                "LongSpark MRV2 currently requires TP=PP=DCP=PCP=1; got "
                + ", ".join(f"{name}={unsupported_parallel[name]}" for name in invalid)
            )
        if (
            device_type := getattr(self.device, "type", None)
        ) and device_type != "cuda":
            raise ValueError("LongSpark MRV2 currently requires CUDA")
        if self.dtype != torch.bfloat16:
            raise ValueError("LongSpark MRV2 currently requires bfloat16")
        if vllm_config.model_config.quantization is not None:
            raise ValueError("LongSpark MRV2 does not support a quantized target")
        if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("LongSpark MRV2 requires an unquantized BF16 KV cache")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError(
                "LongSpark MRV2 does not yet support prefix caching; set "
                "enable_prefix_caching=False"
            )
        if vllm_config.lora_config is not None:
            raise ValueError("LongSpark MRV2 does not yet support target LoRA")
        if vllm_config.model_config.supports_multimodal_inputs:
            raise ValueError("LongSpark MRV2 currently supports text-only targets")
        if not 1 <= self.num_speculative_steps <= int(config.block_size):
            raise ValueError(
                "LongSpark requires 1 <= num_speculative_tokens <= checkpoint "
                f"block_size ({config.block_size}); got {self.num_speculative_steps}"
            )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        self._all_target_attn_layer_names = target_attn_layer_names
        model = load_longspark_model(target_model, self.vllm_config)
        self._fused_markov_weights = model.fused_markov_weights()
        return model

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        # LongSpark has no draft K/V cache or draft attention metadata.  Keep
        # only the target references needed for paged logical gathers.
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups

        target_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
            self._all_target_attn_layer_names,
        )
        selected_names: list[str] = []
        selected_layers: list[AttentionLayerBase] = []
        for layer_id in self.target_layer_ids:
            pattern = re.compile(rf"(?:^|\.)layers\.{layer_id}\.self_attn\.attn$")
            matches = [
                (name, layer)
                for name, layer in target_layers.items()
                if pattern.search(name)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "LongSpark expected exactly one target attention layer for "
                    f"layer {layer_id}, found {[name for name, _ in matches]}"
                )
            name, layer = matches[0]
            backend_name = layer.get_attn_backend().get_name()
            if backend_name != "FLASH_ATTN":
                raise ValueError(
                    "LongSpark MRV2 currently requires FLASH_ATTN target KV "
                    f"layout, got {backend_name} for {name}"
                )
            if (
                getattr(layer, "num_kv_heads", None) != self.num_kv_heads
                or getattr(layer, "head_size", None) != self.head_dim
            ):
                raise ValueError(
                    f"LongSpark target KV shape does not match checkpoint at {name}"
                )
            selected_names.append(name)
            selected_layers.append(layer)
        self.target_attn_layer_names = selected_names
        self.target_attn_layers = selected_layers
        group_by_layer = {
            layer_name: group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            for layer_name in group.layer_names
        }
        try:
            self.target_block_tables = tuple(
                block_tables.input_block_tables[group_by_layer[name]]
                for name in selected_names
            )
        except KeyError as error:
            raise RuntimeError(
                f"LongSpark target layer has no KV-cache group: {error.args[0]}"
            ) from error
        # KV caches are allocated after set_attn() and are rebound once more
        # between CUDA-graph memory profiling and the real initialization.
        # Build the device pointer tables lazily from the live tensors.
        self._target_cache_addresses = ()
        self._target_block_table_addresses = ()
        self._target_cache_ptrs = None
        self._target_block_table_ptrs = None

    def _target_pointer_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        cache_addresses = tuple(
            layer.kv_cache.data_ptr() for layer in self.target_attn_layers
        )
        table_addresses = tuple(table.data_ptr() for table in self.target_block_tables)
        if (
            self._target_cache_ptrs is None
            or cache_addresses != self._target_cache_addresses
        ):
            self._target_cache_ptrs = torch.tensor(
                cache_addresses,
                dtype=torch.uint64,
                device=self.device,
            )
            self._target_cache_addresses = cache_addresses
        if (
            self._target_block_table_ptrs is None
            or table_addresses != self._target_block_table_addresses
        ):
            self._target_block_table_ptrs = torch.tensor(
                table_addresses,
                dtype=torch.uint64,
                device=self.device,
            )
            self._target_block_table_addresses = table_addresses
        return self._target_cache_ptrs, self._target_block_table_ptrs

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        supports_graph = (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and self.draft_logits is None
            and self.draft_watermarker is None
        )
        self.longspark_cudagraph_manager = (
            LongSparkCudaGraphManager(
                self.vllm_config,
                self.device,
                dtype=self.dtype,
                selected_layers=self.num_selected_layers,
                hidden_size=self.hidden_size,
                query_heads=self.num_attention_heads,
                global_slots=self.global_slots,
                head_dim=self.head_dim,
                block_size=self.block_size,
            )
            if supports_graph
            else None
        )
        if not supports_graph:
            logger.info_once(
                "LongSpark draft CUDA graphs require target FULL decode graphs "
                "and greedy, unwatermarked draft sampling; using eager draft."
            )

    def capture(self) -> None:
        if self.longspark_cudagraph_manager is None:
            return
        self.longspark_cudagraph_manager.capture(self._generate_draft)

    @staticmethod
    def _gather_cache_rows(
        layer: AttentionLayerBase,
        block_table_row: torch.Tensor,
        start: int,
        end: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = layer.kv_cache
        if cache.ndim != 4 or cache.shape[-1] != 2 * head_dim:
            raise RuntimeError(
                "LongSpark expected FlashAttention KV cache layout "
                "[blocks, kv_heads, block_size, 2 * head_dim]"
            )
        if cache.dtype != torch.bfloat16:
            raise RuntimeError(
                f"LongSpark expected BF16 target KV cache, got {cache.dtype}"
            )
        block_size = cache.shape[2]
        positions = torch.arange(start, end, dtype=torch.long, device=cache.device)
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        offsets = positions.remainder(block_size)
        physical_blocks = block_table_row.index_select(0, logical_blocks).long()
        pages = cache.index_select(0, physical_blocks)
        row_indices = torch.arange(end - start, device=cache.device)
        rows = pages.permute(0, 2, 1, 3)[row_indices, offsets]
        keys, values = rows.split(head_dim, dim=-1)
        return keys.transpose(0, 1), values.transpose(0, 1)

    @staticmethod
    def _gather_cache_positions(
        layer: AttentionLayerBase,
        block_table_rows: torch.Tensor,
        positions: torch.Tensor,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather a dense batch of logical positions without duplicating pages."""
        cache = layer.kv_cache
        if cache.ndim != 4 or cache.shape[-1] != 2 * head_dim:
            raise RuntimeError(
                "LongSpark expected FlashAttention KV cache layout "
                "[blocks, kv_heads, block_size, 2 * head_dim]"
            )
        block_size = cache.shape[2]
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        offsets = positions.remainder(block_size)
        physical_blocks = torch.gather(
            block_table_rows.long(), 1, logical_blocks.long()
        )
        rows = cache[physical_blocks, :, offsets]
        keys, values = rows.split(head_dim, dim=-1)
        return keys.transpose(1, 2), values.transpose(1, 2)

    def _scan_global_state(
        self,
        request_slot: int,
        batch_index: int,
        start: int,
        end: int,
        attn_metadata: dict[str, Any],
    ) -> None:
        if start >= end:
            return
        for layer_index, (name, layer) in enumerate(
            zip(self.target_attn_layer_names, self.target_attn_layers)
        ):
            metadata = attn_metadata[name]
            block_table_row = metadata.block_table[batch_index]
            output = lse = None
            if start > 0:
                output = self.global_output[request_slot, layer_index]
                lse = self.global_lse[request_slot, layer_index]
            query = self.model.global_queries[layer_index]
            for chunk_start in range(start, end, self._global_scan_chunk_size):
                chunk_end = min(chunk_start + self._global_scan_chunk_size, end)
                keys, values = self._gather_cache_rows(
                    layer,
                    block_table_row,
                    chunk_start,
                    chunk_end,
                    self.head_dim,
                )
                chunk_output, chunk_lse = normalized_gqa_attention(query, keys, values)
                if output is None:
                    output, lse = chunk_output, chunk_lse
                else:
                    assert lse is not None
                    output, lse = merge_normalized_attention(
                        output, lse, chunk_output, chunk_lse
                    )
            assert output is not None and lse is not None
            self.global_output[request_slot, layer_index].copy_(output)
            self.global_lse[request_slot, layer_index].copy_(lse)

    def _update_global_states(
        self,
        input_batch: InputBatch,
        committed_lens: torch.Tensor,
        attn_metadata: dict[str, Any],
        *,
        defer_decode: bool = False,
    ) -> torch.Tensor | None:
        rebuild_rows: list[tuple[int, int, int]] = []
        prefill_delta_rows: list[tuple[int, int, int, int]] = []
        decode_batch_indices: list[int] = []
        decode_request_slots: list[int] = []
        for batch_index, request_id in enumerate(input_batch.req_ids):
            request_slot = int(input_batch.idx_mapping_np[batch_index])
            rebuild = self._state_owner[request_slot] != request_id
            if rebuild:
                # A new request has no prior speculative tokens, so the CPU
                # upper bound is exact.  Full scans are intentionally a host-
                # planned, infrequent prefill operation.
                end = int(input_batch.seq_lens_cpu_upper_bound[batch_index])
                rebuild_rows.append((batch_index, request_slot, end))
                self._state_committed_len[request_slot] = end
                self.num_full_scans += 1
            elif input_batch.is_prefilling_np[batch_index]:
                # Chunked prefill can add more rows than the fixed-size decode
                # delta kernel supports.  Prefill lengths are host-known and
                # exact because no draft tokens are being verified.
                start = int(self._state_committed_len[request_slot])
                end = int(input_batch.seq_lens_cpu_upper_bound[batch_index])
                if start < end:
                    prefill_delta_rows.append((batch_index, request_slot, start, end))
                    self.num_delta_updates += 1
                    self.num_delta_rows += end - start
                self._state_committed_len[request_slot] = end
            else:
                decode_batch_indices.append(batch_index)
                decode_request_slots.append(request_slot)
                self.num_delta_updates += 1
            self._state_owner[request_slot] = request_id

        if rebuild_rows:
            self._scan_global_batch(rebuild_rows, attn_metadata)
        if prefill_delta_rows:
            self._update_global_delta_batch(prefill_delta_rows, attn_metadata)
        deferred_update = None
        if decode_batch_indices:
            if len(decode_batch_indices) == input_batch.num_reqs:
                batch_indices = self._batch_indices[: input_batch.num_reqs]
                request_slots = input_batch.idx_mapping[: input_batch.num_reqs]
            else:
                batch_indices = torch.tensor(
                    decode_batch_indices, dtype=torch.long, device=self.device
                )
                request_slots = torch.tensor(
                    decode_request_slots, dtype=torch.long, device=self.device
                )
            if defer_decode and len(decode_batch_indices) == input_batch.num_reqs:
                deferred_update = request_slots
            else:
                starts = self._state_committed_lens_gpu.index_select(0, request_slots)
                ends = committed_lens.index_select(0, batch_indices)
                self._update_global_decode_batch(
                    batch_indices, request_slots, starts, ends, attn_metadata
                )

        # Publish the exact, rejection-adjusted lengths for the next decode
        # step without making the target stream wait on a D2H transfer. Steady
        # CUDA-graph decode folds this publish into graph input preparation,
        # avoiding two index-selects plus a separate index-copy launch.
        if deferred_update is None:
            self._state_committed_lens_gpu.index_copy_(
                0,
                input_batch.idx_mapping[: input_batch.num_reqs],
                committed_lens,
            )
        return deferred_update

    def _scan_global_batch(
        self,
        rows: list[tuple[int, int, int]],
        attn_metadata: dict[str, Any],
    ) -> None:
        batch_indices = torch.tensor(
            [row[0] for row in rows], dtype=torch.long, device=self.device
        )
        request_slots = torch.tensor(
            [row[1] for row in rows], dtype=torch.long, device=self.device
        )
        ends = torch.tensor(
            [row[2] for row in rows], dtype=torch.long, device=self.device
        )
        max_end = max(row[2] for row in rows)
        for chunk_start in range(0, max_end, self._global_scan_chunk_size):
            active = ends > chunk_start
            active_indices = active.nonzero().flatten()
            active_batch = batch_indices.index_select(0, active_indices)
            active_slots = request_slots.index_select(0, active_indices)
            active_ends = ends.index_select(0, active_indices)
            width = min(self._global_scan_chunk_size, max_end - chunk_start)
            positions = (
                chunk_start
                + torch.arange(width, dtype=torch.long, device=self.device)[None]
            )
            positions = positions.expand(active_indices.numel(), -1)
            valid_mask = positions < active_ends[:, None]
            safe_positions = torch.minimum(positions, active_ends[:, None] - 1)
            for layer_index, (name, layer) in enumerate(
                zip(self.target_attn_layer_names, self.target_attn_layers)
            ):
                block_tables = attn_metadata[name].block_table.index_select(
                    0, active_batch
                )
                keys, values = self._gather_cache_positions(
                    layer, block_tables, safe_positions, self.head_dim
                )
                output, lse = normalized_gqa_attention_batched(
                    self.model.global_queries[layer_index],
                    keys,
                    values,
                    valid_mask,
                )
                if chunk_start:
                    output, lse = merge_normalized_attention(
                        self.global_output[active_slots, layer_index],
                        self.global_lse[active_slots, layer_index],
                        output,
                        lse,
                    )
                self.global_output[:, layer_index].index_copy_(0, active_slots, output)
                self.global_lse[:, layer_index].index_copy_(0, active_slots, lse)

    def _update_global_delta_batch(
        self,
        rows: list[tuple[int, int, int, int]],
        attn_metadata: dict[str, Any],
    ) -> None:
        batch_indices = torch.tensor(
            [row[0] for row in rows], dtype=torch.long, device=self.device
        )
        request_slots = torch.tensor(
            [row[1] for row in rows], dtype=torch.long, device=self.device
        )
        starts = torch.tensor(
            [row[2] for row in rows], dtype=torch.long, device=self.device
        )
        ends = torch.tensor(
            [row[3] for row in rows], dtype=torch.long, device=self.device
        )
        width = max(row[3] - row[2] for row in rows)
        use_kernel = width <= self.max_accepted_tokens
        if not use_kernel:
            positions = (
                starts[:, None]
                + torch.arange(width, dtype=torch.long, device=self.device)[None]
            )
            valid_mask = positions < ends[:, None]
            safe_positions = torch.minimum(positions, ends[:, None] - 1)
        for layer_index, (name, layer) in enumerate(
            zip(self.target_attn_layer_names, self.target_attn_layers)
        ):
            if use_kernel:
                update_longspark_global_delta(
                    query=self.model.global_queries[layer_index],
                    cache=layer.kv_cache,
                    block_table=attn_metadata[name].block_table,
                    batch_indices=batch_indices,
                    request_slots=request_slots,
                    starts=starts,
                    ends=ends,
                    state_output=self.global_output[:, layer_index],
                    state_lse=self.global_lse[:, layer_index],
                    max_delta_rows=width,
                )
            else:
                block_tables = attn_metadata[name].block_table.index_select(
                    0, batch_indices
                )
                keys, values = self._gather_cache_positions(
                    layer, block_tables, safe_positions, self.head_dim
                )
                delta_output, delta_lse = normalized_gqa_attention_batched(
                    self.model.global_queries[layer_index],
                    keys,
                    values,
                    valid_mask,
                )
                output, lse = merge_normalized_attention(
                    self.global_output[request_slots, layer_index],
                    self.global_lse[request_slots, layer_index],
                    delta_output,
                    delta_lse,
                )
                self.global_output[:, layer_index].index_copy_(0, request_slots, output)
                self.global_lse[:, layer_index].index_copy_(0, request_slots, lse)

    def _update_global_decode_batch(
        self,
        batch_indices: torch.Tensor,
        request_slots: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        attn_metadata: dict[str, Any],
    ) -> None:
        """Update global memory for a decode step without host synchronization."""
        for layer_index, (name, layer) in enumerate(
            zip(self.target_attn_layer_names, self.target_attn_layers)
        ):
            update_longspark_global_delta(
                query=self.model.global_queries[layer_index],
                cache=layer.kv_cache,
                block_table=attn_metadata[name].block_table,
                batch_indices=batch_indices,
                request_slots=request_slots,
                starts=starts,
                ends=ends,
                state_output=self.global_output[:, layer_index],
                state_lse=self.global_lse[:, layer_index],
                max_delta_rows=self.max_accepted_tokens,
            )

    def _gather_raw_window(
        self,
        committed_lens: torch.Tensor,
        attn_metadata: dict[str, Any],
        raw_keys: torch.Tensor | None = None,
        raw_values: torch.Tensor | None = None,
        raw_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = committed_lens.shape[0]
        width = self.raw_rows
        shape = (
            batch_size,
            self.num_selected_layers,
            self.num_kv_heads,
            width,
            self.head_dim,
        )
        if raw_keys is None:
            raw_keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        if raw_values is None:
            raw_values = torch.empty(shape, dtype=self.dtype, device=self.device)
        ends = committed_lens.long()
        counts = ends.clamp(max=self.raw_rows)
        starts = ends - counts
        positions = (
            starts[:, None]
            + torch.arange(width, dtype=torch.long, device=self.device)[None]
        )
        if raw_mask is None:
            raw_mask = torch.empty(
                (batch_size, width), dtype=torch.bool, device=self.device
            )
        torch.lt(positions, ends[:, None], out=raw_mask)
        safe_positions = torch.minimum(positions, ends[:, None] - 1)
        for layer_index, (name, layer) in enumerate(
            zip(self.target_attn_layer_names, self.target_attn_layers)
        ):
            keys, values = self._gather_cache_positions(
                layer,
                attn_metadata[name].block_table[:batch_size],
                safe_positions,
                self.head_dim,
            )
            raw_keys[:, layer_index].copy_(keys)
            raw_values[:, layer_index].copy_(values)
        return raw_keys, raw_values, raw_mask

    def _select_target_hidden(
        self,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        num_rejected: torch.Tensor,
    ) -> torch.Tensor:
        if aux_hidden_states is None:
            raise RuntimeError("LongSpark requires selected target hidden states")
        if len(aux_hidden_states) != self.num_selected_layers:
            raise RuntimeError(
                "LongSpark expected "
                f"{self.num_selected_layers} target hidden-state tensors, got "
                f"{len(aux_hidden_states)}"
            )
        index_tensor = (
            input_batch.query_start_loc[1 : input_batch.num_reqs + 1]
            - num_rejected[: input_batch.num_reqs]
            - 1
        ).long()
        return torch.stack(
            [states.index_select(0, index_tensor) for states in aux_hidden_states],
            dim=1,
        )

    def _sample_block(
        self,
        block_hidden: torch.Tensor,
        block_residual: torch.Tensor | None,
        anchors: torch.Tensor,
        current_positions: torch.Tensor,
        num_reqs: int,
    ) -> None:
        idx_mapping = self.idx_mapping[:num_reqs]
        previous = anchors.long()
        base_logits = self.model.compute_draft_logits(block_hidden, block_residual)
        use_fused_projection = (
            self.draft_logits is None
            and not self.use_acceptance_estimator
            and self._fused_markov_weights is not None
        )
        for step in range(self.num_speculative_steps):
            logits = base_logits[:, step]
            if use_fused_projection:
                assert self._fused_markov_weights is not None
                (
                    embedding_weight,
                    projection_weight,
                    candidate_embedding,
                    candidate_projection,
                ) = self._fused_markov_weights
                if candidate_embedding is not None and candidate_projection is not None:
                    sampled = fused_markov_refined_argmax(
                        logits,
                        previous,
                        candidate_embedding,
                        candidate_projection,
                        embedding_weight,
                        projection_weight,
                        reverse=bool(step & 1),
                        partial_values=self._markov_projection_partial_values,
                        partial_ids=self._markov_projection_partial_ids,
                        tile_values=self._markov_candidate_tile_values,
                        tile_ids=self._markov_candidate_tile_ids,
                        candidate_ids=self._markov_candidate_ids,
                        output=self.draft_tokens[:num_reqs, step],
                        top_k=self._markov_candidate_top_k,
                    )
                else:
                    sampled = fused_markov_argmax(
                        logits,
                        previous,
                        embedding_weight,
                        projection_weight,
                        reverse=bool(step & 1),
                        partial_values=self._markov_projection_partial_values,
                        partial_ids=self._markov_projection_partial_ids,
                        output=self.draft_tokens[:num_reqs, step],
                    )
            else:
                markov_embed = self.model.markov_embed(previous)
                markov_bias = self.model.markov_bias(markov_embed).to(logits.dtype)
                if self.draft_logits is None and not self.use_acceptance_estimator:
                    sampled = fused_add_argmax(
                        logits,
                        markov_bias,
                        partial_values=self._markov_partial_values,
                        partial_ids=self._markov_partial_ids,
                        output=self.draft_tokens[:num_reqs, step],
                    )
                else:
                    logits.add_(markov_bias)
                    self._maybe_predict_acceptance(
                        logits, idx_mapping, self._step_cols[step]
                    )
                    sampler = (
                        gumbel_sample
                        if self.draft_watermarker is None
                        else self.draft_watermarker.sample
                    )
                    sampled = sampler(
                        logits,
                        idx_mapping,
                        self.temperature,
                        self.seeds,
                        current_positions + step,
                        apply_temperature=True,
                        is_drafting=True,
                        logits_cache=self.draft_logits,
                        logits_cache_col=self._step_cols[step],
                        use_fp64=self.use_fp64_gumbel,
                    )
                    self.draft_tokens[:num_reqs, step] = sampled
            previous = sampled

    def _generate_draft(
        self,
        selected_hidden: torch.Tensor,
        anchors: torch.Tensor,
        position_ids: torch.Tensor,
        global_output: torch.Tensor,
        raw_keys: torch.Tensor,
        raw_values: torch.Tensor,
        raw_mask: torch.Tensor,
        current_positions: torch.Tensor,
        num_reqs: int,
        global_request_slots: torch.Tensor | None = None,
        global_starts: torch.Tensor | None = None,
        global_ends: torch.Tensor | None = None,
    ) -> None:
        update_stream = None
        if (
            global_request_slots is not None
            and global_starts is not None
            and global_ends is not None
        ):
            batch_indices = self._batch_indices[:num_reqs]
            target_cache_ptrs, target_block_table_ptrs = self._target_pointer_tables()
            current_stream = torch.cuda.current_stream(self.device)
            update_stream = self._global_update_stream
            update_stream.wait_stream(current_stream)
            with torch.cuda.stream(update_stream):
                update_longspark_global_delta_batched(
                    queries=self.model.global_queries,
                    caches=tuple(layer.kv_cache for layer in self.target_attn_layers),
                    cache_ptrs=target_cache_ptrs,
                    block_tables=self.target_block_tables,
                    block_table_ptrs=target_block_table_ptrs,
                    batch_indices=batch_indices,
                    request_slots=global_request_slots,
                    starts=global_starts,
                    ends=global_ends,
                    state_output=self.global_output,
                    state_lse=self.global_lse,
                    batch_output=global_output,
                    max_delta_rows=self.max_accepted_tokens,
                )
        use_paged_raw = (
            len(self.target_attn_layers) == self.num_selected_layers
            and len(self.target_block_tables) == self.num_selected_layers
        )
        hidden_states = self.model.prepare_hidden(selected_hidden, anchors)
        if update_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(update_stream)
        block_output = self.model.forward_layers(
            hidden_states,
            anchors,
            position_ids,
            global_output,
            raw_keys,
            raw_values,
            raw_mask,
            raw_caches=(
                tuple(layer.kv_cache for layer in self.target_attn_layers)
                if use_paged_raw
                else None
            ),
            raw_block_tables=self.target_block_tables if use_paged_raw else None,
            raw_seq_lens=current_positions if use_paged_raw else None,
            return_residual=True,
        )
        assert isinstance(block_output, tuple)
        block_hidden, block_residual = block_output
        self._sample_block(
            block_hidden,
            block_residual,
            anchors,
            current_positions,
            num_reqs,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        del (
            slot_mappings,
            last_hidden_states,
            dp_sync_state,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile,
            num_speculative_tokens,
        )
        num_reqs = input_batch.num_reqs
        committed_lens = input_batch.seq_lens[:num_reqs] - num_rejected[:num_reqs]

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
            dummy_run=dummy_run,
        )
        if aux_hidden_states is None:
            raise RuntimeError("LongSpark requires selected target hidden states")
        if len(aux_hidden_states) != self.num_selected_layers:
            raise RuntimeError(
                "LongSpark expected "
                f"{self.num_selected_layers} target hidden-state tensors, got "
                f"{len(aux_hidden_states)}"
            )
        request_slots = input_batch.idx_mapping[:num_reqs]
        graph_manager = self.longspark_cudagraph_manager
        graph_desc = (
            None
            if dummy_run or graph_manager is None
            else graph_manager.select(num_reqs)
        )
        graph_inputs = None
        if graph_desc is not None:
            assert graph_manager is not None
            graph_inputs = graph_manager.get_inputs(graph_desc)
        if dummy_run:
            raw_shape = (
                num_reqs,
                self.num_selected_layers,
                self.num_kv_heads,
                self.raw_rows,
                self.head_dim,
            )
            raw_keys = torch.zeros(raw_shape, dtype=self.dtype, device=self.device)
            raw_values = torch.zeros_like(raw_keys)
            raw_mask = torch.zeros(
                (num_reqs, self.raw_rows), dtype=torch.bool, device=self.device
            )
            global_output = torch.zeros(
                (
                    num_reqs,
                    self.num_selected_layers,
                    self.num_attention_heads,
                    self.global_slots,
                    self.head_dim,
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            deferred_update = self._update_global_states(
                input_batch,
                committed_lens,
                attn_metadata,
                defer_decode=graph_desc is not None,
            )
            if graph_desc is not None and deferred_update is None:
                # Slot rebuilds and chunked-prefill deltas are intentionally
                # handled eagerly.  Resume graph replay on the next steady
                # decode iteration.
                graph_desc = None
                graph_inputs = None
            if graph_inputs is None:
                # Paged raw-window attention reads target K/V directly. These
                # placeholders are used only by the dense fallback for
                # unsupported checkpoint shapes.
                raw_shape = (
                    num_reqs,
                    self.num_selected_layers,
                    1,
                    1,
                    1,
                )
                raw_keys = torch.empty(raw_shape, dtype=self.dtype, device=self.device)
                raw_values = torch.empty_like(raw_keys)
                raw_mask = torch.empty(
                    (num_reqs, self.raw_rows),
                    dtype=torch.bool,
                    device=self.device,
                )
                global_output = self.global_output.index_select(0, request_slots)
            else:
                assert deferred_update is not None
                raw_keys = graph_inputs.raw_keys
                raw_values = graph_inputs.raw_values
                raw_mask = graph_inputs.raw_mask
                global_output = graph_inputs.global_output

        current_positions = (
            committed_lens if graph_desc is not None else committed_lens.long()
        )
        if graph_desc is None:
            selected_hidden = self._select_target_hidden(
                input_batch, aux_hidden_states, num_rejected
            )
            sampled_mask = num_sampled[:num_reqs] > 0
            anchors = torch.where(
                sampled_mask,
                last_sampled.index_select(0, request_slots).squeeze(-1),
                next_prefill_tokens[0].index_select(0, request_slots),
            )
            position_ids = (
                current_positions[:, None]
                + torch.arange(self.block_size, dtype=torch.long, device=self.device)[
                    None
                ]
            )
            self._generate_draft(
                selected_hidden,
                anchors,
                position_ids,
                global_output,
                raw_keys,
                raw_values,
                raw_mask,
                current_positions,
                num_reqs,
            )
        else:
            assert graph_manager is not None
            assert deferred_update is not None
            global_request_slots = deferred_update
            captured_num_reqs = graph_desc.num_reqs
            assert captured_num_reqs is not None
            if captured_num_reqs > num_reqs:
                for block_table in self.target_block_tables:
                    block_table[num_reqs:captured_num_reqs].copy_(
                        block_table[:1].expand(captured_num_reqs - num_reqs, -1)
                    )
            graph_manager.replay(
                graph_desc,
                num_reqs=num_reqs,
                selected_hidden_states=aux_hidden_states,
                query_start_loc=input_batch.query_start_loc,
                num_rejected=num_rejected,
                num_sampled=num_sampled,
                last_sampled=last_sampled,
                next_prefill_tokens=next_prefill_tokens[0],
                global_output=global_output,
                raw_keys=raw_keys,
                raw_values=raw_values,
                raw_mask=raw_mask,
                current_positions=current_positions,
                global_request_slots=global_request_slots,
                state_committed_lens=self._state_committed_lens_gpu,
                padding_request_slot=self.padding_state_slot,
            )
        return self.draft_tokens[:num_reqs]
