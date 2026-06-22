# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.interfaces import supports_multimodal_pruning
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.mm.rope import get_rope_state
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
        self.req_states: RequestState | None = None

        self.supports_mm_inputs = encoder_cache is not None
        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()
        self.dtype = self.model_config.dtype
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(model)
            and self.model_config.multimodal_config is not None
            and self.model_config.multimodal_config.is_multimodal_pruning_enabled()
        )

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

        self.rope_state = get_rope_state(
            self.model_config,
            model,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if self.rope_state is not None:
            assert new_req_data.prefill_token_ids is not None
            self.rope_state.init_prefill_positions(
                req_index,
                self.model,
                new_req_data.prefill_token_ids,
                mm_features=new_req_data.mm_features,
            )

    def apply_staged_writes(self) -> None:
        if self.rope_state is not None:
            self.rope_state.apply_staged_writes()

    def _recompute_mrope_positions(
        self,
        mm_embeds: list[torch.Tensor],
        input_batch: InputBatch,
    ) -> list[torch.Tensor]:
        assert self.rope_state is not None
        assert self.req_states is not None
        req_states = self.req_states

        mm_embeds_out: list[torch.Tensor] = []
        mm_embed_idx = 0
        for batch_idx, req_id in enumerate(input_batch.req_ids):
            req_idx = req_states.req_id_to_index[req_id]
            num_computed_tokens = int(req_states.num_computed_tokens_np[req_idx])

            mm_features = self.encoder_cache.mm_features[req_id]
            query_start = num_computed_tokens
            query_end = query_start + int(input_batch.num_scheduled_tokens[batch_idx])

            num_req_mm_embeds = 0
            lo, hi = get_mm_features_in_window(
                mm_features,
                start=query_start,
                end=query_end,
            )
            # iterate and get the mm_embeds num in current window
            for mm_feature in mm_features[lo:hi]:
                start_pos = mm_feature.mm_position.offset
                num_encoder_tokens = mm_feature.mm_position.length
                start_idx = max(query_start - start_pos, 0)
                end_idx = min(query_end - start_pos, num_encoder_tokens)
                curr_embeds_start, curr_embeds_end = (
                    mm_feature.mm_position.get_embeds_indices_in_range(
                        start_idx, end_idx
                    )
                )
                if curr_embeds_start != curr_embeds_end:
                    num_req_mm_embeds += 1

            if num_req_mm_embeds == 0:
                continue

            req_mm_embeds = mm_embeds[mm_embed_idx : mm_embed_idx + num_req_mm_embeds]
            mm_embed_idx += num_req_mm_embeds

            # get prompttoken ids
            prompt_len = int(req_states.prompt_len.np[req_idx])
            prompt_token_ids = req_states.all_token_ids._uva_buf.np[
                req_idx, :prompt_len
            ].tolist()
            # get mrope positions
            start = req_idx * self.rope_state.num_dims
            end = start + self.rope_state.num_dims
            mrope_positions = torch.tensor(
                self.rope_state.prefill_positions._uva_buf.np[start:end, :prompt_len],
                dtype=torch.long,
            )
            req_mm_embeds, new_positions, new_delta = (
                self.model.recompute_mrope_positions(
                    input_ids=prompt_token_ids,
                    multimodal_embeddings=tuple(req_mm_embeds),
                    mrope_positions=mrope_positions,
                    num_computed_tokens=num_computed_tokens,
                )
            )
            new_positions_cpu = new_positions.to(device="cpu", dtype=torch.int32)
            self.rope_state.prefill_positions._uva_buf.cpu[
                start:end, : new_positions_cpu.shape[1]
            ].copy_(new_positions_cpu)
            self.rope_state.prefill_delta.np[req_idx] = new_delta
            self.rope_state.prefill_delta.copy_to_uva()
            mm_embeds_out.extend(req_mm_embeds)

        assert mm_embed_idx == len(mm_embeds)
        return mm_embeds_out

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
    ) -> torch.Tensor:
        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
            scheduled_encoder_inputs
        )
        if mm_kwargs:
            # Execute the multimodal encoder.
            encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
            # Cache the encoder outputs by mm_hash
            self.encoder_cache.encoder_outputs.update(zip(mm_hashes, encoder_outputs))

        mm_embeds, is_mm_embed = self.encoder_runner.gather_mm_embeddings(
            input_batch.req_ids,
            input_batch.num_tokens,
            input_batch.num_scheduled_tokens,
            input_batch.query_start_loc_np,
            input_batch.prefill_len_np,
            input_batch.num_computed_prefill_tokens_np,
        )
        if (
            mm_embeds
            and self.is_multimodal_pruning_enabled
            and self.rope_state is not None
            and self.rope_state.has_delta
        ):
            mm_embeds = self._recompute_mrope_positions(mm_embeds, input_batch)
        # Use unpadded input_ids to match is_mm_embed size (num_tokens).
        # input_batch.input_ids may be padded for CUDA graphs.
        input_ids_unpadded = input_batch.input_ids[: input_batch.num_tokens]
        inputs_embeds = self.encoder_runner.get_inputs_embeds(
            input_ids_unpadded, mm_embeds, is_mm_embed
        )
        return inputs_embeds[: input_batch.num_tokens_after_padding]

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        if self.rope_state is None:
            return {}  # Common case (1D positions).

        self.rope_state.prepare_positions(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            req_states.prefill_len.gpu,
            req_states.num_computed_tokens.gpu,
        )
        positions = self.rope_state.get_positions(input_batch.num_tokens_after_padding)
        return {"positions": positions}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = {}
        if self.supports_mm_inputs:
            inputs_embeds = self.encoder_runner.inputs_embeds[:num_tokens]
            model_inputs["inputs_embeds"] = inputs_embeds
        if self.rope_state is not None:
            model_inputs["positions"] = self.rope_state.get_positions(num_tokens)
        return model_inputs

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            # Use padded sizes - padding is handled by model_runner.prepare_attn.
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            # For piecewise cudagraphs and eager, use unpadded sizes.
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            for_cudagraph_capture=for_capture,
        )
        return attn_metadata
