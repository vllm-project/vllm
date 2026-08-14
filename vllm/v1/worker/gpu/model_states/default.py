# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    compute_mm_prefix_ranges,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.rope import get_rope_state
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.model_states.mm_pruning import maybe_create_mm_pruner
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
        super().__init__(vllm_config, model, encoder_cache, device)

        self.supports_prompt_embeds = self.model_config.enable_prompt_embeds
        if self.supports_prompt_embeds:
            if not self.supports_mm_inputs:
                self.inputs_embeds = torch.zeros(
                    self.max_num_tokens,
                    self.inputs_embeds_size,
                    dtype=self.dtype,
                    device=self.device,
                )
            self.prompt_embeds: dict[str, torch.Tensor] = {}
            self.prompt_is_token_ids: dict[str, torch.Tensor | None] = {}

        self.rope_state = get_rope_state(
            self.model_config,
            model,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
        )

        # Pruner is used for multimodal embedding pruning (EVS).
        self.mm_pruner = maybe_create_mm_pruner(
            self.model_config, model, self.rope_state, encoder_cache
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
        if self.supports_prompt_embeds:
            req_id = new_req_data.req_id
            if new_req_data.prompt_embeds is None:
                self.prompt_embeds.pop(req_id, None)
                self.prompt_is_token_ids.pop(req_id, None)
            else:
                self.prompt_embeds[req_id] = new_req_data.prompt_embeds
                prompt_is_token_ids = new_req_data.prompt_is_token_ids
                self.prompt_is_token_ids[req_id] = (
                    None
                    if prompt_is_token_ids is None
                    else torch.tensor(prompt_is_token_ids, dtype=torch.bool)
                )

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        if self.supports_prompt_embeds:
            self.prompt_embeds.pop(req_id, None)
            self.prompt_is_token_ids.pop(req_id, None)

    def apply_staged_writes(self) -> None:
        if self.rope_state is not None:
            self.rope_state.apply_staged_writes()

    def dummy_inputs_embeds(self, num_tokens: int) -> torch.Tensor:
        """Pre-allocated inputs_embeds buffer for dummy runs (contents unused)."""
        if self.supports_mm_inputs:
            return self.encoder_runner.inputs_embeds[:num_tokens]
        return self.inputs_embeds[:num_tokens]

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        # Use unpadded input_ids to match is_mm_embed size (num_tokens).
        # input_batch.input_ids may be padded for CUDA graphs.
        input_ids_unpadded = input_batch.input_ids[: input_batch.num_tokens]

        if self.supports_mm_inputs:
            self.execute_mm_encoder(scheduled_encoder_inputs)

            mm_embeds, is_mm_embed = super().gather_mm_embeddings(input_batch)
            if self.mm_pruner is not None and mm_embeds:
                # EVS: recompute mrope positions for pruned media.
                mm_embeds = self.mm_pruner.recompute(mm_embeds, input_batch, req_states)
                # We must flush the staged rope updates for prepare_inputs() to
                # pick up.
                self.apply_staged_writes()

            inputs_embeds = self.encoder_runner.get_inputs_embeds(
                input_ids_unpadded, mm_embeds, is_mm_embed
            )
        else:
            input_embeddings = self.model.embed_input_ids(input_ids_unpadded)
            self.inputs_embeds[: input_embeddings.shape[0]] = input_embeddings
            inputs_embeds = self.inputs_embeds

        if self.supports_prompt_embeds:
            self._apply_prompt_embeds(input_batch, req_states, inputs_embeds)

        return inputs_embeds[: input_batch.num_tokens_after_padding]

    def gather_mm_embeddings(
        self, input_batch: InputBatch, draft_lookahead: int = 0
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        mm_embeds, is_mm_embed = super().gather_mm_embeddings(
            input_batch, draft_lookahead
        )
        if self.mm_pruner is not None:
            # EVS: strip the appended mrope-position channels.
            mm_embeds = self.mm_pruner.strip(mm_embeds)
        return mm_embeds, is_mm_embed

    def _apply_prompt_embeds(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        inputs_embeds: torch.Tensor,
    ) -> None:
        prefill_lens = req_states.prefill_len.np[input_batch.idx_mapping_np]
        computed_lens = req_states.num_computed_prefill_tokens[
            input_batch.idx_mapping_np
        ]

        for batch_idx, req_id in enumerate(input_batch.req_ids):
            prompt_embeds = self.prompt_embeds.get(req_id)
            if prompt_embeds is None:
                continue

            query_start = int(computed_lens[batch_idx])
            query_end = min(
                query_start + int(input_batch.num_scheduled_tokens[batch_idx]),
                int(prefill_lens[batch_idx]),
                prompt_embeds.shape[0],
            )
            if query_start >= query_end:
                continue

            out_start = int(input_batch.query_start_loc_np[batch_idx])
            out_end = out_start + query_end - query_start
            src = prompt_embeds[query_start:query_end].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )

            prompt_is_token_ids = self.prompt_is_token_ids.get(req_id)
            if prompt_is_token_ids is None:
                inputs_embeds[out_start:out_end].copy_(src)
                continue

            token_mask = prompt_is_token_ids[query_start:query_end]
            embed_positions = torch.nonzero(~token_mask, as_tuple=False).flatten()
            if embed_positions.numel() == 0:
                continue
            embed_positions = embed_positions.to(device=self.device, non_blocking=True)
            inputs_embeds[out_start:out_end].index_copy_(
                0,
                embed_positions,
                src.index_select(0, embed_positions),
            )

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
        if self.supports_mm_inputs or self.supports_prompt_embeds:
            model_inputs["inputs_embeds"] = self.dummy_inputs_embeds(num_tokens)
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
        query_start_loc_cpu = torch.from_numpy(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        query_start_loc_gpu = input_batch.query_start_loc[: num_reqs + 1]
        max_query_len = input_batch.max_query_len
        if max_query_len is None:
            max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
        req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
        if (
            self.supports_mm_inputs
            and self.encoder_cache is not None
            and self.model_config.is_mm_prefix_lm
        ):
            req_doc_ranges = compute_mm_prefix_ranges(
                req_ids=input_batch.req_ids,
                mm_features=self.encoder_cache.mm_features,
                sliding_window=self.model_config.get_sliding_window(),
            )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
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
            is_prefilling=torch.from_numpy(input_batch.is_prefilling_np),
            mm_req_doc_ranges=req_doc_ranges,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
        return attn_metadata
