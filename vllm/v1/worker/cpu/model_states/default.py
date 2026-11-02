# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import compute_mm_prefix_ranges
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.cpu.attn_utils import build_attn_metadata
from vllm.v1.worker.cpu.encoder_cache import EncoderCache
from vllm.v1.worker.cpu.input_batch import InputBatch
from vllm.v1.worker.cpu.mm_pruning import maybe_create_mm_pruner
from vllm.v1.worker.cpu.model_states.interface import (
    ModelSpecificAttnMetadata,
    ModelState,
)
from vllm.v1.worker.cpu.rope import get_rope_state
from vllm.v1.worker.cpu.states import CPURequestState
from vllm.v1.worker.utils import AttentionGroup


class DefaultModelState(ModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)

        self.rope_state = get_rope_state(
            self.model_config,
            model,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
        )
        self.mm_pruner = maybe_create_mm_pruner(
            self.model_config, model, self.rope_state, encoder_cache
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if self.rope_state is not None:
            assert new_req_data.prefill_token_ids is not None
            # `prompt_embeds` is a passthrough modality with no grid info, but
            # M-RoPE assumes per-feature grids. Filter it out.
            mm_features = [
                f for f in new_req_data.mm_features if f.modality != "prompt_embeds"
            ]
            self.rope_state.init_prefill_positions(
                req_index,
                self.model,
                new_req_data.prefill_token_ids,
                mm_features=mm_features,
            )

    def dummy_inputs_embeds(self, num_tokens: int) -> torch.Tensor:
        """Pre-allocated inputs_embeds buffer for dummy runs (contents unused)."""
        return self.encoder_runner.inputs_embeds[:num_tokens]

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: CPURequestState,
    ) -> torch.Tensor:
        # Use unpadded input_ids to match is_mm_embed size (num_tokens).
        input_ids_unpadded = input_batch.input_ids[: input_batch.num_tokens]

        self.execute_mm_encoder(scheduled_encoder_inputs)

        mm_embeds, is_mm_embed = super().gather_mm_embeddings(input_batch)
        if self.mm_pruner is not None and mm_embeds:
            # EVS: recompute mrope positions for pruned media.
            mm_embeds = self.mm_pruner.recompute(mm_embeds, input_batch, req_states)

        inputs_embeds = self.encoder_runner.get_inputs_embeds(
            input_ids_unpadded, mm_embeds, is_mm_embed
        )
        return inputs_embeds[: input_batch.num_tokens_after_padding]

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: CPURequestState
    ) -> dict[str, torch.Tensor | None]:
        if self.rope_state is None:
            return {}  # Common case (1D positions).

        self.rope_state.prepare_positions(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            req_states.prefill_len,
            req_states.num_computed_tokens,
        )
        positions = self.rope_state.get_positions(input_batch.num_tokens_after_padding)
        return {"positions": positions}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = {}
        if self.supports_mm_inputs:
            model_inputs["inputs_embeds"] = self.dummy_inputs_embeds(num_tokens)
        if self.rope_state is not None:
            model_inputs["positions"] = self.rope_state.get_positions(num_tokens)
        return model_inputs

    def prepare_attn(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        query_start_loc = input_batch.query_start_loc[: num_reqs + 1]
        max_query_len = input_batch.max_query_len
        if max_query_len is None:
            max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()

        req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
        if (
            self.supports_mm_inputs
            and self.encoder_cache is not None
            and self.model_config.is_mm_prefix_lm
        ):
            mm_features = self.encoder_cache.mm_features
            hf_text_config = self.model_config.hf_text_config
            req_doc_ranges = compute_mm_prefix_ranges(
                [mm_features.get(req_id, ()) for req_id in input_batch.req_ids],
                sliding_window=self.model_config.get_sliding_window(),
                clamp_sliding_window=getattr(
                    self.model, "mm_prefix_clamp_sliding_window", False
                )
                or getattr(hf_text_config, "mm_prefix_clamp_sliding_window", False),
                span_pad=getattr(self.model, "mm_prefix_span_leading_pad_modulus", 0),
            )

        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc=query_start_loc,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            positions=input_batch.positions,
            is_prefilling=torch.from_numpy(input_batch.is_prefilling_np),
            mm_req_doc_ranges=req_doc_ranges,
            rswa_prefix_lens=input_batch.prompt_lens,
            req_idx=input_batch.idx_mapping_np,
            fast_prefill=input_batch.fast_prefill,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        return attn_metadata
