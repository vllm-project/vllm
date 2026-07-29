# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, replace

import numpy as np
import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
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

logger = init_logger(__name__)


@dataclass(frozen=True)
class RankSegment:
    global_batch_req_idx: int
    global_batch_slice: slice
    rank_local_batch_slice: slice

    @property
    def num_tokens(self) -> int:
        return self.global_batch_slice.stop - self.global_batch_slice.start


class PCPManager:
    """MRV2 PC batch manager.

    The model runner keeps the global scheduled batch. This manager rewrites only
    the per-step InputBatch into rank-local DualChunkSwap rows and keeps the
    global-batch view private to restore to the global batch shape before
    sampling/postprocess.
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

        self._global_batch: InputBatch | None = None
        self._req_states = req_states
        self._block_tables = block_tables
        self._hidden_restore_idx: torch.Tensor | None = None
        self._padded_gather_idx: torch.Tensor | None = None
        self._gathered_kv_write_mask: torch.Tensor | None = None
        self._pad_slot_id = torch.tensor(PAD_SLOT_ID, dtype=torch.int64, device=device)

        max_num_local_reqs = 2 * max_num_reqs if max_num_reqs is not None else None
        self._input_buffers = (
            InputBuffers(max_num_local_reqs, max_num_tokens, device)
            if max_num_local_reqs is not None and max_num_tokens is not None
            else None
        )
        self._local_req_idx = (
            torch.arange(max_num_local_reqs, dtype=torch.int32, device=device)
            if max_num_local_reqs is not None
            else None
        )
        self._local_block_tables: tuple[torch.Tensor, ...] | None
        self._local_block_table_ptrs: torch.Tensor | None
        if block_tables is not None and max_num_local_reqs is not None:
            self._local_block_tables = tuple(
                table.new_zeros((max_num_local_reqs, table.shape[1]))
                for table in block_tables.input_block_tables
            )
            self._local_block_table_ptrs = torch.tensor(
                [table.data_ptr() for table in self._local_block_tables],
                dtype=torch.uint64,
                device=device,
            )
        else:
            self._local_block_tables = None
            self._local_block_table_ptrs = None
        num_kv_cache_groups = (
            block_tables.num_kv_cache_groups if block_tables is not None else 0
        )
        self._global_batch_slot_mappings = (
            torch.empty(
                num_kv_cache_groups,
                max_num_tokens,
                dtype=torch.int64,
                device=device,
            )
            if max_num_tokens is not None and num_kv_cache_groups > 0
            else None
        )
        self._gathered_kv_slot_mappings = (
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
            raise NotImplementedError("MRV2 PCP supports PIECEWISE CUDA graphs only.")

    @staticmethod
    def _reorder_segments(
        segments: list[RankSegment],
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[RankSegment]:
        """Move pure prefills last to match the batch ordering expected by
        attention backends like MLA and sparse MLA.
        """

        def is_pure_prefill(segment: RankSegment) -> bool:
            req_idx = segment.global_batch_req_idx
            start_pos = (
                num_computed_tokens[req_idx]
                + segment.global_batch_slice.start
                - query_start_loc_np[req_idx]
            )
            return is_prefilling[req_idx] and start_pos == 0

        segments.sort(key=is_pure_prefill)
        rank_offset = 0
        for index, segment in enumerate(segments):
            segments[index] = replace(
                segment,
                rank_local_batch_slice=slice(
                    rank_offset, rank_offset + segment.num_tokens
                ),
            )
            rank_offset += segment.num_tokens
        return segments

    def _get_rank_segments(
        self,
        rank: int,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> list[RankSegment]:
        """Build one rank's attention-compatible DualChunkSwap rows.

        PCP=4 partitions each prefill into eight chunks:

            full:  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
            rank 0:  0                           7
            rank 1:      1                   6
            rank 2:          2           5
            rank 3:              3   4
        """
        rank_segments = []
        rank_offset = 0
        num_chunks = 2 * self.pcp_world_size
        for global_batch_req_idx, num_tokens in enumerate(num_scheduled_tokens):
            query_len = int(num_tokens)
            if query_len == 0:
                continue
            global_batch_start = int(query_start_loc_np[global_batch_req_idx])
            chunk_indices: tuple[int, ...]
            if bool(is_prefilling[global_batch_req_idx]):
                chunk_size = (query_len + num_chunks - 1) // num_chunks
                chunk_indices = (rank, num_chunks - 1 - rank)
            else:  # decodes are replicated
                chunk_size = query_len
                chunk_indices = (0,)

            for chunk_idx in chunk_indices:
                chunk_offset = chunk_idx * chunk_size
                chunk_len = min(chunk_size, query_len - chunk_offset)
                if chunk_len <= 0:
                    continue
                chunk_start = global_batch_start + chunk_offset
                rank_segments.append(
                    RankSegment(
                        global_batch_req_idx=global_batch_req_idx,
                        global_batch_slice=slice(chunk_start, chunk_start + chunk_len),
                        rank_local_batch_slice=slice(
                            rank_offset, rank_offset + chunk_len
                        ),
                    )
                )
                rank_offset += chunk_len
        return self._reorder_segments(
            rank_segments,
            num_computed_tokens,
            is_prefilling,
            query_start_loc_np,
        )

    def _build_batch_layout(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        is_prefilling: np.ndarray,
        query_start_loc_np: np.ndarray,
    ) -> tuple[list[list[RankSegment]], list[int]]:
        segments_by_rank = []
        per_rank_num_tokens = []
        for rank in range(self.pcp_world_size):
            segments = self._get_rank_segments(
                rank,
                num_scheduled_tokens,
                num_computed_tokens,
                is_prefilling,
                query_start_loc_np,
            )
            num_rank_tokens = sum(segment.num_tokens for segment in segments)
            segments_by_rank.append(segments)
            per_rank_num_tokens.append(num_rank_tokens)

        # PCP=2 example:
        #   global batch:       [A B C D E F G]
        #   rank 0 / rank 1:    [A B G] / [C D E F]
        #   padded gathered:    [A B G _ | C D E F]
        #   hidden_restore_idx: [0, 1, 4, 5, 6, 7, 2]
        #   padded_gather_idx:  [0, 1, 6, 0, 2, 3, 4, 5]
        # Therefore global = gathered[hidden_restore_idx] and
        # padded_gathered = global[padded_gather_idx].
        hidden_restore_idx = np.empty(int(query_start_loc_np[-1]), dtype=np.int64)
        padded_num_tokens = max(per_rank_num_tokens)
        num_expanded_tokens = padded_num_tokens * self.pcp_world_size
        padded_gather_idx = np.zeros(num_expanded_tokens, dtype=np.int64)
        gathered_kv_write_mask = np.zeros(num_expanded_tokens, dtype=np.bool_)
        for rank, segments in enumerate(segments_by_rank):
            expanded_rank_offset = rank * padded_num_tokens
            for segment in segments:
                padded_gathered_slice = slice(
                    expanded_rank_offset + segment.rank_local_batch_slice.start,
                    expanded_rank_offset + segment.rank_local_batch_slice.stop,
                )
                padded_gather_idx[padded_gathered_slice] = np.arange(
                    segment.global_batch_slice.start,
                    segment.global_batch_slice.stop,
                    dtype=np.int64,
                )
                # Cache insertion pairs one slot entry with each rank's local decode.
                if not bool(is_prefilling[segment.global_batch_req_idx]) and rank != 0:
                    continue
                gathered_kv_write_mask[padded_gathered_slice] = True
                hidden_restore_idx[segment.global_batch_slice] = np.arange(
                    padded_gathered_slice.start,
                    padded_gathered_slice.stop,
                    dtype=np.int64,
                )

        self._hidden_restore_idx = async_copy_to_gpu(
            hidden_restore_idx, device=self.device
        )
        self._padded_gather_idx = async_copy_to_gpu(
            padded_gather_idx, device=self.device
        )
        self._gathered_kv_write_mask = async_copy_to_gpu(
            gathered_kv_write_mask, device=self.device
        )
        return segments_by_rank, per_rank_num_tokens

    def partition_batch(self, input_batch: InputBatch) -> InputBatch:
        assert self._req_states is not None
        assert self._input_buffers is not None
        req_states = self._req_states
        input_buffers = self._input_buffers
        if input_batch.num_draft_tokens > 0:
            raise NotImplementedError("MRV2 PCP does not support spec decode yet.")

        global_batch = input_batch
        self._global_batch = global_batch

        num_scheduled_tokens = global_batch.num_scheduled_tokens
        num_computed_tokens = global_batch.num_computed_tokens_np
        is_prefilling = global_batch.is_prefilling_np

        segments_by_rank, per_rank_num_tokens = self._build_batch_layout(
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
            global_batch.query_start_loc_np,
        )

        local_segments = segments_by_rank[self.pcp_rank]
        if not local_segments:
            local_segments = [
                RankSegment(
                    global_batch_req_idx=0,
                    global_batch_slice=slice(0, 0),
                    rank_local_batch_slice=slice(0, 0),
                )
            ]

        num_local_reqs = len(local_segments)
        if num_local_reqs > input_buffers.max_num_reqs:
            raise RuntimeError(
                "PCP local request count exceeds the MRV2 input buffer size: "
                f"{num_local_reqs} > {input_buffers.max_num_reqs}."
            )

        local_to_global_batch_req_idx_np = np.fromiter(
            (segment.global_batch_req_idx for segment in local_segments),
            dtype=np.int32,
            count=num_local_reqs,
        )
        local_start_pos_np = np.fromiter(
            (
                num_computed_tokens[segment.global_batch_req_idx]
                + segment.global_batch_slice.start
                - global_batch.query_start_loc_np[segment.global_batch_req_idx]
                for segment in local_segments
            ),
            dtype=np.int32,
            count=num_local_reqs,
        )
        local_num_scheduled_tokens = np.fromiter(
            (segment.num_tokens for segment in local_segments),
            dtype=np.int32,
            count=num_local_reqs,
        )
        local_to_global_req_idx_np = global_batch.idx_mapping_np[
            local_to_global_batch_req_idx_np
        ]
        local_req_ids = [
            global_batch.req_ids[global_batch_req_idx]
            for global_batch_req_idx in local_to_global_batch_req_idx_np
        ]

        num_local_tokens = int(local_num_scheduled_tokens.sum())
        num_local_tokens_padded = max(per_rank_num_tokens)
        fresh_prefills = int(
            np.count_nonzero(is_prefilling & (num_computed_tokens == 0))
        )
        continued_prefills = int(
            np.count_nonzero(is_prefilling & (num_computed_tokens > 0))
        )
        logger.debug(
            "PCP batch: rank=%d global_batch_reqs=%d fresh_prefills=%d "
            "continued_prefills=%d decodes=%d local_reqs=%d "
            "local_tokens=%d per_rank_tokens=%s",
            self.pcp_rank,
            global_batch.num_reqs,
            fresh_prefills,
            continued_prefills,
            global_batch.num_reqs - fresh_prefills - continued_prefills,
            num_local_reqs,
            num_local_tokens,
            per_rank_num_tokens,
        )
        if num_local_tokens_padded > input_buffers.max_num_tokens:
            raise RuntimeError(
                "PCP local token count exceeds the MRV2 input buffer size: "
                f"{num_local_tokens_padded} > {input_buffers.max_num_tokens}."
            )
        rank_token_start = self.pcp_rank * num_local_tokens_padded
        assert self._padded_gather_idx is not None
        local_gather_idx = self._padded_gather_idx[
            rank_token_start : rank_token_start + num_local_tokens_padded
        ]
        torch.index_select(
            global_batch.input_ids,
            0,
            local_gather_idx,
            out=input_buffers.input_ids[:num_local_tokens_padded],
        )

        local_query_start_loc_np = np.empty(
            input_buffers.max_num_reqs + 1, dtype=np.int32
        )
        local_query_start_loc_np[0] = 0
        local_query_start_loc_out = local_query_start_loc_np[1 : num_local_reqs + 1]
        np.cumsum(local_num_scheduled_tokens, out=local_query_start_loc_out)
        local_query_start_loc_np[num_local_reqs + 1 :] = num_local_tokens
        async_copy_to_gpu(local_query_start_loc_np, out=input_buffers.query_start_loc)
        local_query_start_loc = input_buffers.query_start_loc[: num_local_reqs + 1]

        local_to_global_req_idx = async_copy_to_gpu(
            local_to_global_req_idx_np, device=self.device
        )
        local_start_pos = async_copy_to_gpu(local_start_pos_np, device=self.device)

        assert self._local_req_idx is not None
        prepare_pos_seq_lens(
            self._local_req_idx[:num_local_reqs],
            local_query_start_loc,
            local_start_pos,
            input_buffers.positions,
            input_buffers.seq_lens[:num_local_reqs],
        )
        seq_lens = input_buffers.seq_lens[:num_local_reqs]
        is_padding = input_buffers.is_padding[:num_local_tokens_padded]
        is_padding[:num_local_tokens].fill_(False)
        is_padding[num_local_tokens:].fill_(True)
        if num_local_tokens_padded > num_local_tokens:
            input_buffers.input_ids[:num_local_tokens_padded].masked_fill_(
                is_padding, 0
            )
            input_buffers.positions[:num_local_tokens_padded].masked_fill_(
                is_padding, 0
            )

        total_num_logits = num_local_reqs if num_local_tokens > 0 else 0
        if total_num_logits > 0:
            cu_num_logits_np = np.arange(num_local_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_local_reqs + 1, device=self.device, dtype=torch.int32
            )
        else:
            cu_num_logits_np = np.zeros(num_local_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.zeros(
                num_local_reqs + 1, device=self.device, dtype=torch.int32
            )
        logits_indices = combine_sampled_and_draft_tokens(
            input_buffers.input_ids,
            local_to_global_req_idx,
            req_states.last_sampled_tokens,
            local_query_start_loc,
            seq_lens,
            req_states.prefill_len.gpu,
            req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            1,
        )

        local_prefill_len_np = global_batch.prefill_len_np[
            local_to_global_batch_req_idx_np
        ]
        local_num_computed_prefill_tokens_np = np.minimum(
            local_start_pos_np, local_prefill_len_np
        )
        local_is_prefilling_np = (
            local_num_computed_prefill_tokens_np < local_prefill_len_np
        )
        seq_lens_cpu_upper_bound_np = np.zeros(num_local_reqs, dtype=np.int32)
        seq_lens_cpu_upper_bound_np[:] = local_start_pos_np + local_num_scheduled_tokens

        dcp_local_seq_lens = None
        if self.dcp_world_size > 1:
            prepare_dcp_local_seq_lens(
                input_buffers.dcp_local_seq_lens,
                seq_lens,
                num_local_reqs,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_local_reqs]

        return replace(
            input_batch,
            req_ids=local_req_ids,
            num_reqs=num_local_reqs,
            num_reqs_after_padding=num_local_reqs,
            idx_mapping=local_to_global_req_idx,
            idx_mapping_np=local_to_global_req_idx_np,
            expanded_idx_mapping=local_to_global_req_idx,
            expanded_local_pos=torch.zeros(
                num_local_reqs, dtype=torch.int32, device=self.device
            ),
            num_scheduled_tokens=local_num_scheduled_tokens,
            num_tokens=num_local_tokens,
            num_tokens_after_padding=num_local_tokens_padded,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=local_query_start_loc,
            query_start_loc_np=local_query_start_loc_np[: num_local_reqs + 1],
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_cpu_upper_bound_np),
            dcp_local_seq_lens=dcp_local_seq_lens,
            num_computed_tokens_np=local_start_pos_np,
            prefill_len_np=local_prefill_len_np,
            num_computed_prefill_tokens_np=local_num_computed_prefill_tokens_np,
            is_prefilling_np=local_is_prefilling_np,
            max_seq_len_np=global_batch.max_seq_len_np[local_to_global_batch_req_idx_np]
            if global_batch.max_seq_len_np is not None
            else None,
            input_ids=input_buffers.input_ids[:num_local_tokens_padded],
            positions=input_buffers.positions[:num_local_tokens_padded],
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            prompt_lens=None,
        )

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        assert self._block_tables is not None
        assert self._local_block_tables is not None
        assert self._local_block_table_ptrs is not None
        block_tables = self._block_tables.gather_block_tables(
            input_batch.idx_mapping,
            input_batch.num_reqs_after_padding,
            out=self._local_block_tables,
            out_ptrs=self._local_block_table_ptrs,
        )
        slot_mappings = self.prepare_slot_mappings()
        return block_tables, slot_mappings

    def prepare_slot_mappings(self) -> torch.Tensor:
        assert self._block_tables is not None
        assert self._global_batch_slot_mappings is not None
        assert self._global_batch is not None
        global_batch = self._global_batch
        global_batch_slot_mappings = self._block_tables.compute_slot_mappings(
            global_batch.idx_mapping,
            global_batch.query_start_loc,
            global_batch.positions,
            global_batch.num_tokens,
            out=self._global_batch_slot_mappings,
        )
        return self._convert_to_gathered_slot_mappings(global_batch_slot_mappings)

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        assert self._gathered_kv_slot_mappings is not None
        self._gathered_kv_slot_mappings.fill_(PAD_SLOT_ID)
        return self._gathered_kv_slot_mappings[:, : num_tokens * self.pcp_world_size]

    def _convert_to_gathered_slot_mappings(
        self,
        global_batch_slot_mappings: torch.Tensor,
    ) -> torch.Tensor:
        assert self._padded_gather_idx is not None
        assert self._gathered_kv_write_mask is not None
        padded_gather_idx = self._padded_gather_idx
        num_expanded_tokens = padded_gather_idx.shape[0]
        if self._gathered_kv_slot_mappings is None:
            self._gathered_kv_slot_mappings = global_batch_slot_mappings.new_empty(
                global_batch_slot_mappings.shape[0], num_expanded_tokens
            )
        gathered_kv_slot_mappings = self._gathered_kv_slot_mappings[
            :, :num_expanded_tokens
        ]
        torch.index_select(
            global_batch_slot_mappings,
            1,
            padded_gather_idx,
            out=gathered_kv_slot_mappings,
        )
        torch.where(
            self._gathered_kv_write_mask.unsqueeze(0),
            gathered_kv_slot_mappings,
            self._pad_slot_id,
            out=gathered_kv_slot_mappings,
        )
        return gathered_kv_slot_mappings

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._hidden_restore_idx is None:
            return hidden_states
        gathered = get_pcp_group().all_gather(hidden_states, dim=0)
        return gathered[self._hidden_restore_idx]

    def restore_for_sampling(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, InputBatch]:
        assert self._global_batch is not None
        return self.restore_hidden_states(hidden_states), self._global_batch


def maybe_partition_pcp_batch(
    manager: PCPManager | None,
    input_batch: InputBatch,
) -> InputBatch:
    if manager is None:
        return input_batch
    return manager.partition_batch(input_batch)


def maybe_get_pcp_dummy_slot_mappings(
    manager: PCPManager | None,
    block_tables: BlockTables,
    num_tokens: int,
) -> torch.Tensor:
    if manager is None:
        return block_tables.get_dummy_slot_mappings(num_tokens)
    return manager.get_dummy_slot_mappings(num_tokens)


def maybe_restore_pcp_for_sampling(
    manager: PCPManager | None,
    hidden_states: torch.Tensor | None,
    input_batch: InputBatch,
) -> tuple[torch.Tensor, InputBatch]:
    assert hidden_states is not None
    if manager is None:
        return hidden_states, input_batch
    return manager.restore_for_sampling(hidden_states)


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
