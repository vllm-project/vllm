# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    prepare_pos_seq_lens_with_start_pos,
)
from vllm.v1.worker.gpu.states import RequestState

SlotMappingsComputer = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor
]


def allgather_token_tensors(
    tensors: tuple[torch.Tensor, ...],
    per_rank_num_tokens: list[int] | None,
) -> tuple[torch.Tensor, ...]:
    if per_rank_num_tokens is None:
        return tensors

    pcp_group = get_pcp_group()
    padded_num_tokens = max(per_rank_num_tokens)
    if all(tensor.shape[0] == padded_num_tokens for tensor in tensors):
        gathered_tensors = tuple(
            pcp_group.all_gather(tensor, dim=0) for tensor in tensors
        )
        if all(num_tokens == padded_num_tokens for num_tokens in per_rank_num_tokens):
            return gathered_tensors
        return tuple(
            torch.cat(
                [
                    rank_tensor[:num_tokens]
                    for rank_tensor, num_tokens in zip(
                        gathered_tensor.split(padded_num_tokens, dim=0),
                        per_rank_num_tokens,
                    )
                ],
                dim=0,
            )
            for gathered_tensor in gathered_tensors
        )

    local_num_tokens = per_rank_num_tokens[pcp_group.rank_in_group]
    local_tensors = [tensor[:local_num_tokens] for tensor in tensors]
    return tuple(
        pcp_group.all_gatherv(
            local_tensors,
            dim=0,
            sizes=per_rank_num_tokens,
        )
    )


def allgather_and_restore_tokens(
    tensor: torch.Tensor,
    hidden_restore_idx: torch.Tensor | None,
    per_rank_num_tokens: list[int] | None,
) -> torch.Tensor:
    if hidden_restore_idx is None or per_rank_num_tokens is None:
        return tensor
    (gathered_tensor,) = allgather_token_tensors((tensor,), per_rank_num_tokens)
    return gathered_tensor[hidden_restore_idx]


def gather_mla_latent_cache_inputs(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor,
    per_rank_num_tokens: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = kv_c_normed.shape[0]
    k_pe_flat = k_pe.reshape(num_tokens, -1)
    gathered_kv_c, gathered_k_pe_flat, gathered_slots = allgather_token_tensors(
        (kv_c_normed, k_pe_flat, slot_mapping[:num_tokens]),
        per_rank_num_tokens,
    )
    gathered_k_pe = gathered_k_pe_flat.view(-1, *k_pe.shape[1:])
    return gathered_kv_c, gathered_k_pe, gathered_slots


def gather_indexer_cache_inputs(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    per_rank_num_tokens: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = k.shape[0]
    return allgather_token_tensors((k, slot_mapping[:num_tokens]), per_rank_num_tokens)


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
        input_buffers: InputBuffers | None = None,
        compute_slot_mappings: SlotMappingsComputer | None = None,
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
        self._input_buffers = input_buffers
        self._compute_slot_mappings = compute_slot_mappings
        self._hidden_restore_idx: torch.Tensor | None = None
        self._per_rank_num_tokens: list[int] | None = None

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
        dcp_size = parallel_config.decode_context_parallel_size
        if dcp_size > 1:
            if dcp_size != pcp_size:
                raise NotImplementedError(
                    "MRV2 MLA PCP+DCP currently implements only the replicated-Q "
                    "layout where decode_context_parallel_size equals "
                    "prefill_context_parallel_size. Full TP x PCP DCP needs the "
                    "gathered-Q/reduce-scatter path."
                )
            if parallel_config.dcp_comm_backend != "ag_rs":
                raise NotImplementedError(
                    "MRV2 MLA PCP+DCP currently supports only the ag_rs DCP "
                    "communication backend."
                )
        if vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            raise NotImplementedError("MRV2 PCP does not support full CUDA graphs yet.")

    def _pad_num_reqs(self, num_reqs: int) -> int:
        return (
            (num_reqs + self.pcp_world_size - 1)
            // self.pcp_world_size
            * self.pcp_world_size
        )

    def forward_context_kwargs(self) -> dict[str, Any]:
        return {
            "pcp_per_rank_num_tokens": self._per_rank_num_tokens,
        }

    @staticmethod
    def _set_padding_mask(
        is_padding: torch.Tensor,
        num_tokens: int,
        num_tokens_after_padding: int,
    ) -> torch.Tensor:
        is_padding[:num_tokens].fill_(False)
        is_padding[num_tokens:num_tokens_after_padding].fill_(True)
        return is_padding[:num_tokens_after_padding]

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
            _prepare_prefill_inputs_with_start_pos(
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

    def partition_batch(self, input_batch: InputBatch) -> InputBatch:
        assert self._req_states is not None
        assert self._input_buffers is not None
        return self.partition_input_batch(
            input_batch,
            self._req_states,
            self._input_buffers,
        )

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor:
        assert self._compute_slot_mappings is not None
        return self._compute_slot_mappings(
            idx_mapping,
            query_start_loc,
            positions,
            num_tokens_padded,
        )

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._hidden_restore_idx is None or self._per_rank_num_tokens is None:
            return hidden_states
        if self.pcp_world_size == 1:
            return hidden_states
        return allgather_and_restore_tokens(
            hidden_states,
            self._hidden_restore_idx,
            self._per_rank_num_tokens,
        )

    def restore_for_sampling(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, InputBatch]:
        assert self._physical_batch is not None
        physical_batch = self._physical_batch
        return self.restore_hidden_states(hidden_states), physical_batch


def get_pcp_forward_context_kwargs(
    manager: PCPManager | None,
    dummy_run: bool,
) -> dict[str, Any] | None:
    if manager is None or dummy_run:
        return None
    return manager.forward_context_kwargs()


def get_pcp_max_num_input_reqs(
    vllm_config: VllmConfig,
    supports_mm_inputs: bool,
) -> int:
    parallel_config = vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    max_num_reqs = vllm_config.scheduler_config.max_num_seqs
    if pcp_size <= 1:
        return max_num_reqs

    PCPManager.validate_config(vllm_config, supports_mm_inputs)
    return max_num_reqs * 2 + pcp_size - 1


def maybe_build_pcp_manager(
    vllm_config: VllmConfig,
    device: torch.device,
    supports_mm_inputs: bool,
    req_states: RequestState,
    input_buffers: InputBuffers,
    compute_slot_mappings: SlotMappingsComputer,
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
        input_buffers=input_buffers,
        compute_slot_mappings=compute_slot_mappings,
        dcp_world_size=dcp_size,
        dcp_rank=dcp_rank,
        cp_interleave=parallel_config.cp_kv_cache_interleave_size,
    )
