# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.mm.mrope_utils import MRopeState
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class DefaultModelState(ModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.device = device

        self.supports_mm_inputs = encoder_cache is not None
        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()
        self.dtype = self.model_config.dtype

        if self.supports_mm_inputs:
            assert encoder_cache is not None
            self.encoder_cache = encoder_cache
            self.encoder_runner = EncoderRunner(
                model=self.model,
                max_num_tokens=self.max_num_tokens,
                hidden_size=self.inputs_embeds_size,
                encoder_cache=encoder_cache,
                dtype=self.dtype,
                device=self.device,
            )

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

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
            scheduled_encoder_inputs
        )
        if mm_kwargs:
            # Execute the multimodal encoder.
            encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
            self.encoder_cache.encoder_outputs.update(zip(mm_hashes, encoder_outputs))

        mm_embeds, is_mm_embed = self.encoder_runner.gather_mm_embeddings(
            input_batch.req_ids,
            input_batch.num_tokens,
            input_batch.num_scheduled_tokens,
            input_batch.query_start_loc_np,
            req_states.prefill_len.np[input_batch.idx_mapping_np],
            req_states.num_computed_prefill_tokens[input_batch.idx_mapping_np],
        )
        inputs_embeds = self.encoder_runner.get_inputs_embeds(
            input_batch.input_ids, mm_embeds, is_mm_embed
        )
        return inputs_embeds[: input_batch.num_tokens_after_padding]

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
        model_inputs = {}
        if self.supports_mm_inputs:
            inputs_embeds = self.encoder_runner.inputs_embeds[:num_tokens]
            model_inputs["inputs_embeds"] = inputs_embeds
        if self.uses_mrope:
            mrope_positions = self.mrope_state.mrope_positions[:, :num_tokens]
            model_inputs["positions"] = mrope_positions
        return model_inputs

    def prepare_attn(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, Any]:
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=input_batch.num_reqs,
            num_tokens=input_batch.num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
        )
        return attn_metadata
