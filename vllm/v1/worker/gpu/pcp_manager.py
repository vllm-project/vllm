# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, replace

import numpy as np
import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    prepare_pos_seq_lens,
)
from vllm.v1.worker.gpu.states import RequestState


@dataclass
class PCPBatchLayout:
    hidden_restore_idx: torch.Tensor
    per_rank_num_tokens: list[int]
    rank_segments: list[list[tuple[int, int, int]]]


def allgather_tokens(
    tensor: torch.Tensor,
    per_rank_num_tokens: list[int],
) -> torch.Tensor:
    pcp_group = get_pcp_group()
    padded_num_tokens = max(per_rank_num_tokens)
    assert tensor.shape[0] == padded_num_tokens
    gathered = pcp_group.all_gather(tensor, dim=0)
    if all(num_tokens == padded_num_tokens for num_tokens in per_rank_num_tokens):
        return gathered
    return torch.cat(
        [
            rank_tensor[:num_tokens]
            for rank_tensor, num_tokens in zip(
                gathered.split(padded_num_tokens, dim=0),
                per_rank_num_tokens,
            )
        ],
        dim=0,
    )


@triton.jit
def _prepare_prefill_inputs_with_start_pos_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prefill_lens_ptr,
    virtual_start_pos_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    start_pos = tl.load(virtual_start_pos_ptr + batch_idx)
    if start_pos >= prefill_len:
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    request_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(request_ptr + start_pos + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    next_pos = start_pos + query_len
    if next_pos < prefill_len:
        next_token = tl.load(request_ptr + next_pos)
        tl.store(next_prefill_tokens_ptr + req_state_idx, next_token)


def _prepare_prefill_inputs_with_start_pos(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    virtual_start_pos: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    _prepare_prefill_inputs_with_start_pos_kernel[(num_reqs,)](
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        prefill_len,
        virtual_start_pos,
        BLOCK_SIZE=1024,
    )


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
        req_states: RequestState | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        block_tables: BlockTables | None = None,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        cp_interleave: int = 1,
    ) -> None:
        self.pcp_world_size = pcp_world_size
        self.pcp_rank = pcp_rank
        self.device = device
        self.dcp_world_size = dcp_world_size
        self.dcp_rank = dcp_rank
        self.cp_interleave = cp_interleave

        self._physical_batch: InputBatch | None = None
        self._req_states = req_states
        self._block_tables = block_tables
        self._compute_slot_mappings = (
            block_tables.compute_slot_mappings if block_tables is not None else None
        )
        self._batch_layout: PCPBatchLayout | None = None

        max_num_virtual_reqs = (
            self._pad_num_reqs(2 * max_num_reqs) if max_num_reqs is not None else None
        )
        self._input_buffers = (
            InputBuffers(max_num_virtual_reqs, max_num_tokens, device)
            if max_num_virtual_reqs is not None and max_num_tokens is not None
            else None
        )
        self._virtual_idx_mapping = (
            torch.arange(max_num_virtual_reqs, dtype=torch.int32, device=device)
            if max_num_virtual_reqs is not None
            else None
        )
        self._virtual_block_tables: tuple[torch.Tensor, ...] | None
        self._virtual_block_table_ptrs: torch.Tensor | None
        if block_tables is not None and max_num_virtual_reqs is not None:
            (
                self._virtual_block_tables,
                self._virtual_block_table_ptrs,
            ) = block_tables.make_input_block_tables(max_num_virtual_reqs)
        else:
            self._virtual_block_tables = None
            self._virtual_block_table_ptrs = None
        num_kv_cache_groups = (
            block_tables.num_kv_cache_groups if block_tables is not None else 0
        )
        self._local_slot_mappings = (
            torch.empty(
                num_kv_cache_groups,
                max_num_tokens,
                dtype=torch.int64,
                device=device,
            )
            if max_num_tokens is not None and num_kv_cache_groups > 0
            else None
        )
        self._cache_slot_mappings = (
            torch.empty(
                num_kv_cache_groups,
                max_num_tokens * pcp_world_size,
                dtype=torch.int64,
                device=device,
            )
            if max_num_tokens is not None and num_kv_cache_groups > 0
            else None
        )

    @staticmethod
    def validate_config(
        vllm_config: VllmConfig,
        supports_mm_inputs: bool,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        pcp_size = parallel_config.prefill_context_parallel_size
        if pcp_size <= 1:
            return

        if not model_config.use_mla:
            raise NotImplementedError("MRV2 PCP currently supports MLA models only.")
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("MRV2 PCP does not support PP yet.")
        if model_config.is_encoder_decoder:
            raise NotImplementedError(
                "MRV2 PCP does not support encoder-decoder models yet."
            )
        if supports_mm_inputs:
            raise NotImplementedError("MRV2 PCP does not support MM inputs yet.")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("MRV2 PCP does not support LoRA yet.")
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "MRV2 PCP does not support speculative decoding yet."
            )
        is_sparse_mla = hasattr(model_config.hf_text_config, "index_topk")
        if (
            is_sparse_mla
            and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            raise NotImplementedError(
                "MRV2 sparse MLA PCP does not support CUDA graphs yet. "
                "Set -cc.cudagraph_mode=NONE."
            )
        if vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            raise NotImplementedError("MRV2 PCP does not support full CUDA graphs yet.")

    def _pad_num_reqs(self, num_reqs: int) -> int:
        return (
            (num_reqs + self.pcp_world_size - 1)
            // self.pcp_world_size
            * self.pcp_world_size
        )

    @staticmethod
    def _set_padding_mask(
        is_padding: torch.Tensor,
        num_tokens: int,
        num_tokens_after_padding: int,
    ) -> torch.Tensor:
        is_padding[:num_tokens].fill_(False)
        is_padding[num_tokens:num_tokens_after_padding].fill_(True)
        return is_padding[:num_tokens_after_padding]

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

            chunk_base, remainder = divmod(query_len, num_chunks)

            for chunk_idx in (rank, num_chunks - 1 - rank):
                chunk_len = chunk_base + int(chunk_idx < remainder)
                if chunk_len == 0:
                    continue
                chunk_offset = chunk_idx * chunk_base + min(chunk_idx, remainder)
                start_pos = base_pos + chunk_offset
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

    def _build_batch_layout(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> PCPBatchLayout:
        all_rank_segments = [
            self._get_rank_segments(
                rank, num_scheduled_tokens, num_computed_tokens, is_prefilling
            )
            for rank in range(self.pcp_world_size)
        ]
        per_rank_num_tokens = [
            sum(num_tokens for _, _, num_tokens in segments)
            for segments in all_rank_segments
        ]

        rank_base_offsets = np.zeros(self.pcp_world_size, dtype=np.int64)
        for rank in range(1, self.pcp_world_size):
            rank_base_offsets[rank] = (
                rank_base_offsets[rank - 1] + per_rank_num_tokens[rank - 1]
            )

        restore_idx = np.empty(int(query_start_loc_np[-1]), dtype=np.int64)
        for rank, segments in enumerate(all_rank_segments):
            rank_offset = 0
            for req_idx, start_pos, num_tokens in segments:
                base_pos = int(num_computed_tokens[req_idx])
                gathered_start = int(rank_base_offsets[rank] + rank_offset)
                rank_offset += num_tokens
                if not bool(is_prefilling[req_idx]) and rank != 0:
                    continue
                output_start = int(query_start_loc_np[req_idx]) + start_pos - base_pos
                restore_idx[output_start : output_start + num_tokens] = np.arange(
                    gathered_start,
                    gathered_start + num_tokens,
                    dtype=np.int64,
                )

        return PCPBatchLayout(
            hidden_restore_idx=async_copy_to_gpu(restore_idx, device=self.device),
            per_rank_num_tokens=per_rank_num_tokens,
            rank_segments=all_rank_segments,
        )

    def partition_batch(self, input_batch: InputBatch) -> InputBatch:
        assert self._req_states is not None
        assert self._input_buffers is not None
        req_states = self._req_states
        input_buffers = self._input_buffers
        if input_batch.num_draft_tokens > 0:
            raise NotImplementedError("MRV2 PCP does not support spec decode yet.")
        if input_batch.num_reqs_after_padding != input_batch.num_reqs:
            raise NotImplementedError(
                "MRV2 PCP does not support request-padded CUDA graphs yet."
            )

        physical_batch = input_batch
        self._physical_batch = physical_batch

        num_scheduled_tokens = physical_batch.num_scheduled_tokens
        num_computed_tokens = physical_batch.num_computed_tokens_np
        is_prefilling = physical_batch.is_prefilling_np

        batch_layout = self._build_batch_layout(
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
            physical_batch.query_start_loc_np,
        )
        self._batch_layout = batch_layout

        local_segments = batch_layout.rank_segments[self.pcp_rank]
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
        virtual_num_tokens_padded = max(batch_layout.per_rank_num_tokens)
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
            _prepare_prefill_inputs_with_start_pos(
                input_buffers.input_ids,
                req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                req_states.all_token_ids.gpu,
                req_states.prefill_len.gpu,
                virtual_start_pos,
            )

        assert self._virtual_idx_mapping is not None
        prepare_pos_seq_lens(
            self._virtual_idx_mapping[:num_virtual_reqs],
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
        is_padding = self._set_padding_mask(
            input_buffers.is_padding,
            virtual_num_tokens,
            virtual_num_tokens_padded,
        )

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

        dcp_local_seq_lens = None
        if self.dcp_world_size > 1:
            prepare_dcp_local_seq_lens(
                input_buffers.dcp_local_seq_lens,
                seq_lens,
                num_virtual_reqs,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[
                :num_virtual_reqs_padded
            ]

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
            dcp_local_seq_lens=dcp_local_seq_lens,
            num_computed_tokens_np=virtual_start_pos_np,
            prefill_len_np=virtual_prefill_len_np,
            num_computed_prefill_tokens_np=virtual_num_computed_prefill_tokens_np,
            is_prefilling_np=virtual_is_prefilling_np,
            max_seq_len_np=physical_batch.max_seq_len_np[virtual_to_physical_np]
            if physical_batch.max_seq_len_np is not None
            else None,
            input_ids=input_buffers.input_ids[:virtual_num_tokens_padded],
            positions=input_buffers.positions[:virtual_num_tokens_padded],
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
        )

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        assert self._block_tables is not None
        assert self._virtual_block_tables is not None
        assert self._virtual_block_table_ptrs is not None
        block_tables = self._block_tables.gather_block_tables(
            input_batch.idx_mapping,
            input_batch.num_reqs_after_padding,
            out=self._virtual_block_tables,
            out_ptrs=self._virtual_block_table_ptrs,
        )
        slot_mappings, cache_slot_mappings = self.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            input_batch.num_tokens_after_padding,
        )
        return block_tables, slot_mappings, cache_slot_mappings

    def prepare_dummy_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        assert self._virtual_block_tables is not None
        assert self._local_slot_mappings is not None
        block_tables = tuple(
            block_table[: input_batch.num_reqs]
            for block_table in self._virtual_block_tables
        )
        slot_mappings = self._local_slot_mappings[:, : input_batch.num_tokens]
        cache_slot_mappings = self.dummy_cache_slot_mappings(slot_mappings)
        return block_tables, cache_slot_mappings, cache_slot_mappings

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._compute_slot_mappings is not None
        assert self._local_slot_mappings is not None
        local_slot_mappings = self._compute_slot_mappings(
            idx_mapping,
            query_start_loc,
            positions,
            num_tokens_padded,
            self._local_slot_mappings,
        )
        cache_slot_mappings = self._compute_cache_slot_mappings(local_slot_mappings)
        return local_slot_mappings, cache_slot_mappings

    def _compute_cache_slot_mappings(
        self,
        local_slot_mappings: torch.Tensor,
    ) -> torch.Tensor:
        assert self._compute_slot_mappings is not None
        assert self._physical_batch is not None
        assert self._batch_layout is not None

        padded_num_tokens = max(self._batch_layout.per_rank_num_tokens)
        full_num_tokens = padded_num_tokens * self.pcp_world_size
        physical_batch = self._physical_batch
        idx_mapping_entries: list[int] = []
        query_start_locs = [0]
        positions_np = np.zeros(full_num_tokens, dtype=np.int64)
        is_padding_np = np.zeros(full_num_tokens, dtype=np.bool_)

        for rank, segments in enumerate(self._batch_layout.rank_segments):
            rank_start = rank * padded_num_tokens
            cursor = rank_start
            for req_idx, start_pos, num_tokens in segments:
                idx_mapping_entries.append(int(physical_batch.idx_mapping_np[req_idx]))
                positions_np[cursor : cursor + num_tokens] = np.arange(
                    start_pos,
                    start_pos + num_tokens,
                    dtype=np.int64,
                )
                cursor += num_tokens
                query_start_locs.append(cursor)

            rank_end = rank_start + padded_num_tokens
            if cursor < rank_end:
                idx_mapping_entries.append(0)
                is_padding_np[cursor:rank_end] = True
                cursor = rank_end
                query_start_locs.append(cursor)

        idx_mapping_np = np.array(idx_mapping_entries, dtype=np.int32)
        query_start_loc_np = np.array(query_start_locs, dtype=np.int32)
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)
        query_start_loc = async_copy_to_gpu(query_start_loc_np, device=self.device)
        positions = async_copy_to_gpu(positions_np, device=self.device)

        if self._cache_slot_mappings is None:
            self._cache_slot_mappings = local_slot_mappings.new_empty(
                local_slot_mappings.shape[0], full_num_tokens
            )
        cache_slot_mappings = self._cache_slot_mappings[:, :full_num_tokens]
        cache_slot_mappings = self._compute_slot_mappings(
            idx_mapping,
            query_start_loc,
            positions,
            full_num_tokens,
            cache_slot_mappings,
        )
        if is_padding_np.any():
            is_padding = async_copy_to_gpu(is_padding_np, device=self.device)
            cache_slot_mappings.masked_fill_(is_padding.unsqueeze(0), PAD_SLOT_ID)
        return cache_slot_mappings

    def dummy_cache_slot_mappings(
        self,
        slot_mappings: torch.Tensor,
    ) -> torch.Tensor:
        full_num_tokens = slot_mappings.shape[1] * self.pcp_world_size
        if self._cache_slot_mappings is None:
            self._cache_slot_mappings = slot_mappings.new_empty(
                slot_mappings.shape[0], full_num_tokens
            )
        cache_slot_mappings = self._cache_slot_mappings[:, :full_num_tokens]
        cache_slot_mappings.fill_(PAD_SLOT_ID)
        return cache_slot_mappings

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._batch_layout is None:
            return hidden_states
        hidden_states = allgather_tokens(
            hidden_states, self._batch_layout.per_rank_num_tokens
        )
        return hidden_states[self._batch_layout.hidden_restore_idx]

    def restore_for_sampling(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, InputBatch]:
        assert self._physical_batch is not None
        physical_batch = self._physical_batch
        return self.restore_hidden_states(hidden_states), physical_batch


def maybe_build_pcp_manager(
    vllm_config: VllmConfig,
    device: torch.device,
    supports_mm_inputs: bool,
    req_states: RequestState,
    block_tables: BlockTables,
) -> PCPManager | None:
    parallel_config = vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    if pcp_size <= 1:
        return None

    PCPManager.validate_config(vllm_config, supports_mm_inputs)

    pcp_rank = get_pcp_group().rank_in_group
    dcp_size = parallel_config.decode_context_parallel_size
    dcp_rank = get_dcp_group().rank_in_group if dcp_size > 1 else 0

    return PCPManager(
        pcp_world_size=pcp_size,
        pcp_rank=pcp_rank,
        device=device,
        req_states=req_states,
        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
        max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        block_tables=block_tables,
        dcp_world_size=dcp_size,
        dcp_rank=dcp_rank,
        cp_interleave=parallel_config.cp_kv_cache_interleave_size,
    )
