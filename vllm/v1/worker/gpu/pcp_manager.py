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
from vllm.v1.worker.gpu.pcp_hidden_restore import (
    PCPMulticastHiddenStateRestorer,
    PCPMulticastUnavailableError,
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
        hidden_state_restorer: PCPMulticastHiddenStateRestorer | None = None,
        max_concurrent_batches: int = 1,
    ) -> None:
        self.pcp_world_size = pcp_world_size
        self.pcp_rank = pcp_rank
        self.device = device
        self.dcp_world_size = dcp_world_size
        self.dcp_rank = dcp_rank
        self.cp_interleave = cp_interleave
        self._hidden_state_restorer = hidden_state_restorer

        self._global_batch: InputBatch | None = None
        self._req_states = req_states
        self._block_tables = block_tables
        self._hidden_restore_idx: torch.Tensor | None = None
        self._hidden_restore_idx_cpu: np.ndarray | None = None
        self._segments_by_rank: list[list[RankSegment]] | None = None
        self._padded_num_tokens = 0
        self._hidden_states_are_replicated = False
        self._sample_rows_are_identity = False
        self._sample_local_row_idx: torch.Tensor | None = None
        self._sample_restore_idx: torch.Tensor | None = None
        self._sample_index_cpu: tuple[torch.Tensor, ...] = ()
        self._sample_index_cpu_np: tuple[np.ndarray, ...] = ()
        self._sample_index_buffers: tuple[torch.Tensor, ...] = ()
        self._next_sample_index_buffer = 0
        if max_num_reqs is not None:
            num_sample_index_buffers = max(2, max_concurrent_batches)
            self._sample_index_cpu = tuple(
                torch.empty(
                    2 * max_num_reqs,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=device.type == "cuda",
                )
                for _ in range(num_sample_index_buffers)
            )
            self._sample_index_cpu_np = tuple(
                buffer.numpy() for buffer in self._sample_index_cpu
            )
            self._sample_index_buffers = tuple(
                torch.empty(
                    2 * max_num_reqs,
                    dtype=torch.int64,
                    device=device,
                )
                for _ in range(num_sample_index_buffers)
            )
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
        # Pure decode rows are replicated on every PCP rank and retain global
        # request order. They need no communication or reorder before sampling.
        self._hidden_states_are_replicated = not np.any(is_prefilling)
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
        #   padded_gather_idx:  [0, 1, 6, 0, 2, 3, 4, 5]
        # Therefore padded_gathered = global[padded_gather_idx]. The inverse
        # dense map is materialized only if prompt logprobs need every row.
        query_lens = np.diff(query_start_loc_np)
        if np.any(query_lens <= 0):
            raise RuntimeError("PCP sampling requires one or more rows per request.")
        self._sample_rows_are_identity = bool(np.all(query_lens == 1))
        padded_num_tokens = max(per_rank_num_tokens)
        num_expanded_tokens = padded_num_tokens * self.pcp_world_size
        padded_gather_idx = np.zeros(num_expanded_tokens, dtype=np.int64)
        gathered_kv_write_mask = np.zeros(num_expanded_tokens, dtype=np.bool_)
        sample_owner = (
            None
            if self._hidden_states_are_replicated
            else np.full(query_lens.shape[0], -1, dtype=np.int64)
        )
        sample_local_row = (
            None
            if self._hidden_states_are_replicated
            else np.empty(query_lens.shape[0], dtype=np.int64)
        )
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
                req_idx = segment.global_batch_req_idx
                if (
                    sample_owner is not None
                    and sample_local_row is not None
                    and segment.global_batch_slice.stop
                    == query_start_loc_np[req_idx + 1]
                ):
                    sample_owner[req_idx] = rank
                    sample_local_row[req_idx] = segment.rank_local_batch_slice.stop - 1

        self._hidden_restore_idx = None
        self._hidden_restore_idx_cpu = None
        self._segments_by_rank = segments_by_rank
        self._padded_num_tokens = padded_num_tokens
        self._sample_local_row_idx = None
        self._sample_restore_idx = None
        if not self._hidden_states_are_replicated:
            assert sample_owner is not None
            assert sample_local_row is not None
            if np.any(sample_owner < 0) or np.any(sample_owner >= self.pcp_world_size):
                raise RuntimeError("PCP sampled-row ownership is out of range.")

            owner_counts = np.bincount(
                sample_owner, minlength=self.pcp_world_size
            ).astype(np.int64, copy=False)
            padded_sample_rows = int(owner_counts.max())
            sample_local_rows = np.zeros(
                (self.pcp_world_size, padded_sample_rows),
                dtype=np.int64,
            )
            sample_restore_idx = np.empty(sample_owner.shape[0], dtype=np.int64)
            owner_offsets = np.zeros(self.pcp_world_size, dtype=np.int64)
            for output_row, (owner, local_row) in enumerate(
                zip(sample_owner, sample_local_row, strict=True)
            ):
                owner_slot = owner_offsets[owner]
                sample_local_rows[owner, owner_slot] = local_row
                sample_restore_idx[output_row] = owner * padded_sample_rows + owner_slot
                owner_offsets[owner] += 1

            num_sample_indices = padded_sample_rows + sample_restore_idx.shape[0]
            if not self._sample_index_cpu_np:
                sample_index_cpu_np = np.empty(num_sample_indices, dtype=np.int64)
                sample_index_cpu = None
                sample_index_buffer = None
            else:
                buffer_index = self._next_sample_index_buffer
                sample_index_cpu = self._sample_index_cpu[buffer_index]
                sample_index_buffer = self._sample_index_buffers[buffer_index]
                self._next_sample_index_buffer = (buffer_index + 1) % len(
                    self._sample_index_buffers
                )
                if num_sample_indices > sample_index_cpu.shape[0]:
                    raise RuntimeError("PCP sampled-row metadata exceeds capacity.")
                sample_index_cpu_np = self._sample_index_cpu_np[buffer_index][
                    :num_sample_indices
                ]
            sample_index_cpu_np[:padded_sample_rows] = sample_local_rows[self.pcp_rank]
            sample_index_cpu_np[padded_sample_rows:] = sample_restore_idx
            if sample_index_buffer is None or sample_index_cpu is None:
                sample_index = async_copy_to_gpu(
                    sample_index_cpu_np,
                    device=self.device,
                )
            else:
                sample_index = sample_index_buffer[:num_sample_indices].copy_(
                    sample_index_cpu[:num_sample_indices],
                    non_blocking=True,
                )
            self._sample_local_row_idx = sample_index[:padded_sample_rows]
            self._sample_restore_idx = sample_index[padded_sample_rows:]

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

    def restore_sample_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert self._global_batch is not None
        if self._hidden_states_are_replicated:
            # PCP rejects speculative decode, so a pure-decode batch contains
            # globally ordered rows. The common one-row case is already the
            # exact sampling view; resumed decode may contain a token backlog.
            if self._sample_rows_are_identity:
                return hidden_states[: self._global_batch.num_reqs]
            return hidden_states[self._global_batch.logits_indices]
        if self._sample_local_row_idx is None or self._sample_restore_idx is None:
            raise RuntimeError("PCP sampled-row restore map is not initialized.")
        # A one-row local prefill has no dense rows to eliminate. Keep the
        # existing collective in this degenerate case; it avoids the extra
        # pack launch needed by multicast. Pure decode bypasses communication
        # above regardless of batch size.
        if hidden_states.shape[0] == self._sample_local_row_idx.shape[0] == 1:
            compact_global_rows = get_pcp_group().all_gather(hidden_states, dim=0)
            return compact_global_rows[self._sample_restore_idx]
        if isinstance(
            self._hidden_state_restorer,
            PCPMulticastHiddenStateRestorer,
        ):
            return self._hidden_state_restorer.restore_selected(
                hidden_states,
                self._sample_local_row_idx,
                self._sample_restore_idx,
                num_selected_rows=self._sample_restore_idx.shape[0],
            )
        compact_local_rows = hidden_states[self._sample_local_row_idx]
        compact_global_rows = get_pcp_group().all_gather(compact_local_rows, dim=0)
        return compact_global_rows[self._sample_restore_idx]

    def restore_full_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Restore dense global rows for prompt-logprob computation."""
        if self._hidden_states_are_replicated:
            return hidden_states
        if self._hidden_restore_idx is None:
            if self._hidden_restore_idx_cpu is None:
                if self._segments_by_rank is None or self._global_batch is None:
                    raise RuntimeError("PCP dense restore layout is not initialized.")
                hidden_restore_idx = np.empty(
                    self._global_batch.num_tokens,
                    dtype=np.int64,
                )
                for rank, segments in enumerate(self._segments_by_rank):
                    expanded_rank_offset = rank * self._padded_num_tokens
                    for segment in segments:
                        req_idx = segment.global_batch_req_idx
                        if (
                            not bool(self._global_batch.is_prefilling_np[req_idx])
                            and rank != 0
                        ):
                            continue
                        padded_start = (
                            expanded_rank_offset + segment.rank_local_batch_slice.start
                        )
                        hidden_restore_idx[segment.global_batch_slice] = np.arange(
                            padded_start,
                            padded_start + segment.num_tokens,
                            dtype=np.int64,
                        )
                self._hidden_restore_idx_cpu = hidden_restore_idx
            self._hidden_restore_idx = async_copy_to_gpu(
                self._hidden_restore_idx_cpu,
                device=self.device,
            )
        gathered = get_pcp_group().all_gather(hidden_states, dim=0)
        return gathered[self._hidden_restore_idx]

    def restore_for_sampling(
        self,
        hidden_states: torch.Tensor,
        *,
        needs_prompt_hidden_states: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, InputBatch]:
        assert self._global_batch is not None
        if needs_prompt_hidden_states:
            hidden_states = self.restore_full_hidden_states(hidden_states)
            sample_hidden_states = hidden_states[self._global_batch.logits_indices]
        else:
            sample_hidden_states = self.restore_sample_hidden_states(hidden_states)
        return hidden_states, sample_hidden_states, self._global_batch

    def close(self) -> None:
        if self._hidden_state_restorer is not None:
            self._hidden_state_restorer.close()
            self._hidden_state_restorer = None


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
    *,
    needs_prompt_hidden_states: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, InputBatch]:
    assert hidden_states is not None
    if manager is None:
        return hidden_states, None, input_batch
    return manager.restore_for_sampling(
        hidden_states,
        needs_prompt_hidden_states=needs_prompt_hidden_states,
    )


def maybe_get_pcp_global_batch(
    manager: PCPManager | None,
    input_batch: InputBatch,
) -> InputBatch:
    if manager is None:
        return input_batch
    if manager._global_batch is None:
        raise RuntimeError("PCP global batch is not initialized.")
    return manager._global_batch


def maybe_create_pcp_hidden_state_restorer(
    vllm_config: VllmConfig,
    device: torch.device,
    supports_mm_inputs: bool,
) -> PCPMulticastHiddenStateRestorer | None:
    """Allocate the fastest available compact-row exchange before profiling.

    The allocation is persistent non-KV memory. Creating it with the model
    runner ensures vLLM's normal memory profiler subtracts it before sizing the
    KV cache. Systems without CUDA multicast retain the compact-row algorithm
    through the existing PCP collective.
    """
    if vllm_config.parallel_config.prefill_context_parallel_size <= 1:
        return None
    PCPManager.validate_config(vllm_config, supports_mm_inputs)
    try:
        return PCPMulticastHiddenStateRestorer(
            group=get_pcp_group().cpu_group,
            device=device,
            max_num_tokens=vllm_config.scheduler_config.max_num_seqs,
            hidden_size=vllm_config.model_config.get_hidden_size(),
            dtype=vllm_config.model_config.dtype,
        )
    except PCPMulticastUnavailableError as error:
        logger.warning_once(
            "CUDA multicast is unavailable for compact PCP final-row exchange; "
            "using the PCP collective backend instead: %s",
            error,
        )
        return None


def maybe_build_pcp_manager(
    vllm_config: VllmConfig,
    device: torch.device,
    supports_mm_inputs: bool,
    req_states: RequestState,
    block_tables: BlockTables,
    hidden_state_restorer: PCPMulticastHiddenStateRestorer | None = None,
) -> PCPManager | None:
    parallel_config = vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    if pcp_size <= 1:
        if hidden_state_restorer is not None:
            raise RuntimeError(
                "PCP hidden-state restorer was allocated with PCP disabled."
            )
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
        hidden_state_restorer=hidden_state_restorer,
        max_concurrent_batches=vllm_config.max_concurrent_batches,
    )
