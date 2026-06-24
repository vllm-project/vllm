# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    prepare_pos_seq_lens_with_start_pos,
    prepare_prefill_inputs_with_start_pos,
)
from vllm.v1.worker.gpu.states import RequestState


class PCPManager:
    """Stateful MRV2 PCP virtual-batch manager.

    The model runner keeps physical request state. This manager rewrites only
    the per-step InputBatch into virtual DualChunkSwap rows and keeps the
    physical view privately for sampling/postprocess restore.
    """

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        dcp_world_size: int = 1,
    ) -> None:
        self.pcp_world_size = pcp_world_size
        self.pcp_rank = pcp_rank
        self.device = device
        self.dcp_world_size = dcp_world_size
        self.dcp_passthrough = False

        self._physical_batch: InputBatch | None = None
        self._physical_slot_mappings_by_layer: dict[str, torch.Tensor] | None = None
        self._physical_attn_metadata_by_layer: dict[str, object] | None = None
        self._hidden_restore_idx: torch.Tensor | None = None
        self._local_virtual_to_physical_idx: torch.Tensor | None = None
        self._per_rank_num_tokens: list[int] | None = None

    def _pad_num_reqs(self, num_reqs: int) -> int:
        return (
            (num_reqs + self.pcp_world_size - 1)
            // self.pcp_world_size
            * self.pcp_world_size
        )

    def _local_num_tokens(self) -> int:
        assert self._per_rank_num_tokens is not None
        return self._per_rank_num_tokens[self.pcp_rank]

    @staticmethod
    def _copy_physical_batch(input_batch: InputBatch) -> InputBatch:
        return replace(
            input_batch,
            req_ids=list(input_batch.req_ids),
            idx_mapping_np=input_batch.idx_mapping_np.copy(),
            num_scheduled_tokens=input_batch.num_scheduled_tokens.copy(),
            query_start_loc=input_batch.query_start_loc.clone(),
            query_start_loc_np=input_batch.query_start_loc_np.copy(),
            seq_lens=input_batch.seq_lens.clone(),
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound.clone(),
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens.clone()
            if input_batch.dcp_local_seq_lens is not None
            else None,
            num_computed_tokens_np=input_batch.num_computed_tokens_np.copy(),
            prefill_len_np=input_batch.prefill_len_np.copy(),
            num_computed_prefill_tokens_np=(
                input_batch.num_computed_prefill_tokens_np.copy()
            ),
            is_prefilling_np=input_batch.is_prefilling_np.copy(),
            max_seq_len_np=input_batch.max_seq_len_np.copy()
            if input_batch.max_seq_len_np is not None
            else None,
            input_ids=input_batch.input_ids.clone(),
            positions=input_batch.positions.clone(),
            logits_indices=input_batch.logits_indices.clone(),
            cu_num_logits=input_batch.cu_num_logits.clone(),
            cu_num_logits_np=input_batch.cu_num_logits_np.copy(),
        )

    def _get_rank_segments(
        self,
        rank: int,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        """Return (physical_req_idx, start_pos, num_tokens) for one PCP rank."""
        segments: list[tuple[int, int, int]] = []
        num_chunks = 2 * self.pcp_world_size
        for req_idx, num_tokens in enumerate(num_scheduled_tokens):
            base_pos = int(num_computed_tokens[req_idx])
            query_len = int(num_tokens)
            if query_len == 0:
                continue
            if not bool(is_prefilling[req_idx]):
                segments.append((req_idx, base_pos, query_len))
                continue

            chunk_base = query_len // num_chunks
            remainder = query_len % num_chunks
            chunk_sizes = np.full(num_chunks, chunk_base, dtype=np.int32)
            chunk_sizes[:remainder] += 1
            chunk_starts = np.zeros(num_chunks + 1, dtype=np.int32)
            np.cumsum(chunk_sizes, out=chunk_starts[1:])

            for chunk_idx in (rank, num_chunks - 1 - rank):
                chunk_len = int(chunk_sizes[chunk_idx])
                if chunk_len == 0:
                    continue
                start_pos = base_pos + int(chunk_starts[chunk_idx])
                segments.append((req_idx, start_pos, chunk_len))
        return sorted(
            segments,
            key=lambda segment: (
                bool(is_prefilling[segment[0]]),
                int(segment[1]) == 0,
                segment[0],
                segment[1],
            ),
        )

    def _build_hidden_restore_idx(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> None:
        all_rank_segments = [
            self._get_rank_segments(
                rank, num_scheduled_tokens, num_computed_tokens, is_prefilling
            )
            for rank in range(self.pcp_world_size)
        ]
        self._per_rank_num_tokens = [
            sum(num_tokens for _, _, num_tokens in segments)
            for segments in all_rank_segments
        ]

        rank_base_offsets = np.zeros(self.pcp_world_size, dtype=np.int64)
        for rank in range(1, self.pcp_world_size):
            rank_base_offsets[rank] = (
                rank_base_offsets[rank - 1] + self._per_rank_num_tokens[rank - 1]
            )

        restore_pairs: list[tuple[int, int]] = []
        local_virtual_to_physical: list[int] = []
        for rank, segments in enumerate(all_rank_segments):
            rank_offset = 0
            for req_idx, start_pos, num_tokens in segments:
                base_pos = int(num_computed_tokens[req_idx])
                is_decode = not bool(is_prefilling[req_idx])
                for local_idx in range(num_tokens):
                    gathered_idx = int(rank_base_offsets[rank] + rank_offset)
                    rank_offset += 1
                    physical_offset = start_pos - base_pos + local_idx
                    output_idx = int(query_start_loc_np[req_idx]) + physical_offset
                    if rank == self.pcp_rank:
                        local_virtual_to_physical.append(output_idx)
                    if is_decode and rank != 0:
                        continue
                    restore_pairs.append((output_idx, gathered_idx))

        restore_pairs.sort(key=lambda item: item[0])
        restore_idx = np.array(
            [gathered_idx for _, gathered_idx in restore_pairs], dtype=np.int64
        )
        self._hidden_restore_idx = async_copy_to_gpu(
            restore_idx,
            device=self.device,
        )
        local_virtual_to_physical_idx = np.asarray(
            local_virtual_to_physical, dtype=np.int64
        )
        self._local_virtual_to_physical_idx = async_copy_to_gpu(
            local_virtual_to_physical_idx,
            device=self.device,
        )

    def _allgather_and_restore_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._hidden_restore_idx is None or self._per_rank_num_tokens is None:
            return tensor
        tensor = tensor[: self._local_num_tokens()]
        gathered = get_pcp_group().all_gatherv(
            tensor.contiguous(),
            dim=0,
            sizes=self._per_rank_num_tokens,
        )
        return gathered[self._hidden_restore_idx]

    def gather_and_restore_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._allgather_and_restore_tokens(tensor)

    def select_local_from_physical(
        self,
        tensor: torch.Tensor,
        local_token_offset: int = 0,
        local_num_tokens: int | None = None,
    ) -> torch.Tensor:
        if self._local_virtual_to_physical_idx is None:
            return tensor
        idx = self._local_virtual_to_physical_idx
        if local_token_offset or local_num_tokens is not None:
            end = None
            if local_num_tokens is not None:
                end = local_token_offset + local_num_tokens
            idx = idx[local_token_offset:end]
        return tensor.index_select(0, idx)

    def set_physical_slot_mappings(
        self,
        slot_mappings_by_layer: dict[str, torch.Tensor],
    ) -> None:
        self._physical_slot_mappings_by_layer = slot_mappings_by_layer

    def set_physical_attn_metadata(
        self,
        attn_metadata_by_layer: dict[str, object],
    ) -> None:
        self._physical_attn_metadata_by_layer = attn_metadata_by_layer

    def get_physical_attn_metadata(self, layer_name: str | None) -> object | None:
        if layer_name is None or self._physical_attn_metadata_by_layer is None:
            return None
        return self._physical_attn_metadata_by_layer.get(layer_name)

    def _get_physical_slot_mapping(self, layer_name: str | None) -> torch.Tensor | None:
        if layer_name is None or self._physical_slot_mappings_by_layer is None:
            return None
        return self._physical_slot_mappings_by_layer.get(layer_name)

    def gather_and_restore_mla_latent_cache_inputs(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather local virtual MLA latent cache inputs into physical order.

        MLA cache entries are the compressed latent KV vector plus RoPE key
        embedding. Sparse MLA PCP relies on this narrow gather: ranks exchange
        only the latent cache update payload, not expanded K/V tensors.
        """
        num_tokens = kv_c_normed.shape[0]
        slot_mapping = slot_mapping[:num_tokens]
        kv_combined = torch.cat(
            (kv_c_normed, k_pe.reshape(num_tokens, -1)),
            dim=-1,
        )
        restored_kv = self._allgather_and_restore_tokens(kv_combined)
        restored_slots = self._get_physical_slot_mapping(layer_name)
        if restored_slots is None:
            restored_slots = self._allgather_and_restore_tokens(slot_mapping)
        else:
            restored_slots = restored_slots[: restored_kv.shape[0]]

        kv_lora_rank = kv_c_normed.shape[-1]
        restored_kv_c = restored_kv[..., :kv_lora_rank]
        restored_k_pe = restored_kv[..., kv_lora_rank:].view(
            -1,
            *k_pe.shape[1:],
        )
        return restored_kv_c, restored_k_pe, restored_slots

    def gather_and_restore_mla_cache_inputs(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.gather_and_restore_mla_latent_cache_inputs(
            kv_c_normed,
            k_pe,
            slot_mapping,
            layer_name,
        )

    def gather_and_restore_indexer_cache_inputs(
        self,
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather local virtual sparse-indexer K cache inputs.

        Sparse MLA has a separate indexer K cache. PCP should exchange only
        that latent K payload before cache insertion; indexer Q and row-local
        top-k work remain local to the virtual batch.
        """
        num_tokens = k.shape[0]
        restored_k = self._allgather_and_restore_tokens(k)
        restored_slots = self._get_physical_slot_mapping(layer_name)
        if restored_slots is None:
            restored_slots = self._allgather_and_restore_tokens(
                slot_mapping[:num_tokens]
            )
        else:
            restored_slots = restored_slots[: restored_k.shape[0]]
        return restored_k, restored_slots

    def partition_input_batch(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        input_buffers: InputBuffers,
    ) -> InputBatch:
        if input_batch.num_draft_tokens > 0:
            raise NotImplementedError("MRV2 PCP does not support spec decode yet.")
        if input_batch.num_reqs_after_padding != input_batch.num_reqs:
            raise NotImplementedError(
                "MRV2 PCP does not support request-padded CUDA graphs yet."
            )

        physical_batch = self._copy_physical_batch(input_batch)
        self._physical_batch = physical_batch

        num_scheduled_tokens = physical_batch.num_scheduled_tokens
        num_computed_tokens = physical_batch.num_computed_tokens_np
        is_prefilling = physical_batch.is_prefilling_np

        self._build_hidden_restore_idx(
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
            physical_batch.query_start_loc_np,
        )

        local_segments = self._get_rank_segments(
            self.pcp_rank, num_scheduled_tokens, num_computed_tokens, is_prefilling
        )
        if not local_segments:
            local_segments = [(0, int(num_computed_tokens[0]), 0)]

        num_virtual_reqs = len(local_segments)
        num_virtual_reqs_padded = self._pad_num_reqs(num_virtual_reqs)
        if num_virtual_reqs_padded > input_buffers.max_num_reqs:
            raise RuntimeError(
                "PCP virtual request count exceeds the MRV2 input buffer size: "
                f"{num_virtual_reqs_padded} > {input_buffers.max_num_reqs}."
            )

        virtual_to_physical_np = np.fromiter(
            (req_idx for req_idx, _, _ in local_segments),
            dtype=np.int32,
            count=num_virtual_reqs,
        )
        virtual_start_pos_np = np.fromiter(
            (start_pos for _, start_pos, _ in local_segments),
            dtype=np.int32,
            count=num_virtual_reqs,
        )
        virtual_num_scheduled = np.fromiter(
            (num_tokens for _, _, num_tokens in local_segments),
            dtype=np.int32,
            count=num_virtual_reqs,
        )
        virtual_idx_mapping_np = physical_batch.idx_mapping_np[virtual_to_physical_np]
        virtual_req_ids = [
            physical_batch.req_ids[physical_idx]
            for physical_idx in virtual_to_physical_np
        ]

        virtual_num_tokens = int(virtual_num_scheduled.sum())
        assert self._per_rank_num_tokens is not None
        virtual_num_tokens_padded = max(self._per_rank_num_tokens)
        if virtual_num_tokens_padded > input_buffers.max_num_tokens:
            raise RuntimeError(
                "PCP virtual token count exceeds the MRV2 input buffer size: "
                f"{virtual_num_tokens_padded} > {input_buffers.max_num_tokens}."
            )

        query_start_loc_np = np.empty(input_buffers.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        query_start_loc_out = query_start_loc_np[1 : num_virtual_reqs + 1]
        np.cumsum(virtual_num_scheduled, out=query_start_loc_out)
        query_start_loc_np[num_virtual_reqs + 1 :] = virtual_num_tokens
        async_copy_to_gpu(query_start_loc_np, out=input_buffers.query_start_loc)
        query_start_loc = input_buffers.query_start_loc[: num_virtual_reqs_padded + 1]

        idx_mapping = async_copy_to_gpu(virtual_idx_mapping_np, device=self.device)
        virtual_start_pos = async_copy_to_gpu(virtual_start_pos_np, device=self.device)

        if np.any(is_prefilling[virtual_to_physical_np]):
            prepare_prefill_inputs_with_start_pos(
                input_buffers.input_ids,
                req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                req_states.all_token_ids.gpu,
                req_states.prefill_len.gpu,
                virtual_start_pos,
            )

        prepare_pos_seq_lens_with_start_pos(
            query_start_loc,
            virtual_start_pos,
            input_buffers.positions,
            input_buffers.seq_lens[:num_virtual_reqs_padded],
        )
        seq_lens = input_buffers.seq_lens[:num_virtual_reqs_padded]
        if virtual_num_tokens_padded > virtual_num_tokens:
            input_buffers.input_ids[
                virtual_num_tokens:virtual_num_tokens_padded
            ].zero_()
            input_buffers.positions[
                virtual_num_tokens:virtual_num_tokens_padded
            ].zero_()

        total_num_logits = num_virtual_reqs if virtual_num_tokens > 0 else 0
        if total_num_logits > 0:
            cu_num_logits_np = np.arange(num_virtual_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_virtual_reqs + 1, device=self.device, dtype=torch.int32
            )
        else:
            cu_num_logits_np = np.zeros(num_virtual_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.zeros(
                num_virtual_reqs + 1, device=self.device, dtype=torch.int32
            )
        logits_indices = combine_sampled_and_draft_tokens(
            input_buffers.input_ids,
            idx_mapping,
            req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            req_states.prefill_len.gpu,
            req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            1,
        )

        virtual_prefill_len_np = physical_batch.prefill_len_np[virtual_to_physical_np]
        virtual_num_computed_prefill_tokens_np = np.minimum(
            virtual_start_pos_np, virtual_prefill_len_np
        )
        virtual_is_prefilling_np = (
            virtual_num_computed_prefill_tokens_np < virtual_prefill_len_np
        )
        seq_lens_cpu_upper_bound_np = np.zeros(num_virtual_reqs_padded, dtype=np.int32)
        seq_lens_cpu_upper_bound_np[:num_virtual_reqs] = (
            virtual_start_pos_np + virtual_num_scheduled
        )

        return replace(
            input_batch,
            req_ids=virtual_req_ids,
            num_reqs=num_virtual_reqs,
            num_reqs_after_padding=num_virtual_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=virtual_idx_mapping_np,
            expanded_idx_mapping=idx_mapping,
            expanded_local_pos=torch.zeros(
                num_virtual_reqs, dtype=torch.int32, device=self.device
            ),
            num_scheduled_tokens=virtual_num_scheduled,
            num_tokens=virtual_num_tokens,
            num_tokens_after_padding=virtual_num_tokens_padded,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np[: num_virtual_reqs_padded + 1],
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_cpu_upper_bound_np),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=virtual_start_pos_np,
            prefill_len_np=virtual_prefill_len_np,
            num_computed_prefill_tokens_np=virtual_num_computed_prefill_tokens_np,
            is_prefilling_np=virtual_is_prefilling_np,
            max_seq_len_np=physical_batch.max_seq_len_np[virtual_to_physical_np]
            if physical_batch.max_seq_len_np is not None
            else None,
            input_ids=input_buffers.input_ids[:virtual_num_tokens_padded],
            positions=input_buffers.positions[:virtual_num_tokens_padded],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
        )

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._hidden_restore_idx is None or self._per_rank_num_tokens is None:
            return hidden_states
        if self.pcp_world_size == 1:
            return hidden_states
        return self._allgather_and_restore_tokens(hidden_states)

    def restore_for_sampling(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
    ) -> tuple[torch.Tensor, InputBatch]:
        del input_batch
        assert self._physical_batch is not None
        physical_batch = self._physical_batch
        self._physical_slot_mappings_by_layer = None
        self._physical_attn_metadata_by_layer = None
        return self.restore_hidden_states(hidden_states), physical_batch
