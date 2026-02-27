# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.mrope_utils import MRopeState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


@dataclass
class AttnMetadataInputs:
    num_reqs: int
    num_tokens: int
    query_start_loc_gpu: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: tuple[torch.Tensor, ...]
    slot_mappings: torch.Tensor
    dcp_local_seq_lens: torch.Tensor | None


class ModelState:
    def __init__(self, vllm_config: VllmConfig, model: nn.Module, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.device = device

        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens

        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = self.compilation_config.cudagraph_mode
        self.uniform_decode_query_len = 1
        spec_config = vllm_config.speculative_config
        if spec_config is not None:
            self.uniform_decode_query_len += spec_config.num_speculative_tokens

        self.uses_mrope = self.model_config.uses_mrope
        if self.uses_mrope:
            self.mrope_state = MRopeState(
                max_num_reqs=self.max_num_reqs,
                max_num_tokens=self.max_num_tokens,
                max_model_len=self.max_model_len,
                device=self.device,
            )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if self.uses_mrope:
            # Pre-compute M-RoPE positions for prefill.
            assert new_req_data.prefill_token_ids is not None
            self.mrope_state.init_prefill_mrope_positions(
                req_index,
                self.model,  # type: ignore
                new_req_data.prefill_token_ids,
                mm_features=new_req_data.mm_features,
            )

    def apply_staged_writes(self) -> None:
        if self.uses_mrope:
            self.mrope_state.apply_staged_writes()

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        if not self.uses_mrope:
            # Common case (1D positions).
            return {}

        # Prepare M-RoPE positions.
        self.mrope_state.prepare_mrope_positions(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            req_states.prefill_len.gpu,
            req_states.num_computed_tokens.gpu,
        )
        mrope_positions = self.mrope_state.mrope_positions[
            :, : input_batch.num_tokens_after_padding
        ]
        return {"positions": mrope_positions}

    def prepare_dummy_inputs(
        self, num_reqs: int, num_tokens: int
    ) -> dict[str, torch.Tensor | None]:
        if not self.uses_mrope:
            return {}
        mrope_positions = self.mrope_state.mrope_positions[:, :num_tokens]
        return {"positions": mrope_positions}

    def _prepare_attn_metadata_inputs(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
    ) -> AttnMetadataInputs:
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)

        if input_batch.num_tokens_after_padding <= num_tokens:
            return AttnMetadataInputs(
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                query_start_loc_gpu=input_batch.query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=input_batch.seq_lens,
                block_tables=block_tables,
                slot_mappings=slot_mappings[:, :num_tokens],
                dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            )

        query_lens_np = (
            input_batch.query_start_loc_np[1 : num_reqs + 1]
            - input_batch.query_start_loc_np[:num_reqs]
        )
        # for separate-routine FULL decode, uniform query lengths must use
        # num_reqs = ceil(num_tokens_after_padding / uniform_decode_query_len)
        # to match the capture-time graph shape (host-side check, no GPU sync).
        is_uniform_full_decode = (
            self.cudagraph_mode.separate_routine()
            and self.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and num_reqs > 0
            and bool((query_lens_np == self.uniform_decode_query_len).all())
        )
        if is_uniform_full_decode:
            attn_num_reqs = min(
                cdiv(
                    input_batch.num_tokens_after_padding, self.uniform_decode_query_len
                ),
                self.max_num_reqs,
            )
        else:
            attn_num_reqs = min(input_batch.num_tokens_after_padding, self.max_num_reqs)
        attn_num_tokens = input_batch.num_tokens_after_padding

        attn_query_start_loc_cpu = torch.empty(attn_num_reqs + 1, dtype=torch.int32)
        attn_query_start_loc_cpu[: num_reqs + 1] = query_start_loc_cpu
        attn_query_start_loc_cpu[num_reqs + 1 :] = num_tokens

        attn_query_start_loc = torch.empty(
            attn_num_reqs + 1,
            dtype=input_batch.query_start_loc.dtype,
            device=input_batch.query_start_loc.device,
        )
        attn_query_start_loc[: num_reqs + 1] = input_batch.query_start_loc
        attn_query_start_loc[num_reqs + 1 :] = num_tokens

        attn_seq_lens = torch.zeros(
            attn_num_reqs,
            dtype=input_batch.seq_lens.dtype,
            device=input_batch.seq_lens.device,
        )
        attn_seq_lens[:num_reqs] = input_batch.seq_lens

        attn_block_tables = tuple(
            torch.cat(
                [
                    block_table,
                    torch.zeros(
                        (attn_num_reqs - num_reqs, block_table.shape[1]),
                        dtype=block_table.dtype,
                        device=block_table.device,
                    ),
                ],
                dim=0,
            )
            for block_table in block_tables
        )

        attn_slot_mappings = torch.full(
            (slot_mappings.shape[0], attn_num_tokens),
            PAD_SLOT_ID,
            dtype=slot_mappings.dtype,
            device=slot_mappings.device,
        )
        attn_slot_mappings[:, :num_tokens] = slot_mappings[:, :num_tokens]

        attn_dcp_local_seq_lens = None
        if input_batch.dcp_local_seq_lens is not None:
            attn_dcp_local_seq_lens = torch.zeros(
                attn_num_reqs,
                dtype=input_batch.dcp_local_seq_lens.dtype,
                device=input_batch.dcp_local_seq_lens.device,
            )
            attn_dcp_local_seq_lens[:num_reqs] = input_batch.dcp_local_seq_lens

        return AttnMetadataInputs(
            num_reqs=attn_num_reqs,
            num_tokens=attn_num_tokens,
            query_start_loc_gpu=attn_query_start_loc,
            query_start_loc_cpu=attn_query_start_loc_cpu,
            seq_lens=attn_seq_lens,
            block_tables=attn_block_tables,
            slot_mappings=attn_slot_mappings,
            dcp_local_seq_lens=attn_dcp_local_seq_lens,
        )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, Any]:
        attn_inputs = self._prepare_attn_metadata_inputs(
            input_batch, block_tables, slot_mappings
        )
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=attn_inputs.num_reqs,
            num_tokens=attn_inputs.num_tokens,
            query_start_loc_gpu=attn_inputs.query_start_loc_gpu,
            query_start_loc_cpu=attn_inputs.query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=attn_inputs.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=attn_inputs.block_tables,
            slot_mappings=attn_inputs.slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=attn_inputs.dcp_local_seq_lens,
        )
        return attn_metadata
