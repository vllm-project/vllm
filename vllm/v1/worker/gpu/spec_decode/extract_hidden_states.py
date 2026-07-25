# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class ExtractHiddenStatesSpeculator(DraftModelSpeculator):
    """Cache target hidden states while returning always-accepted draft tokens."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        if self.num_speculative_steps != 1:
            raise ValueError(
                "extract_hidden_states requires num_speculative_tokens to be 1"
            )
        if self.speculative_config.disable_padded_drafter_batch:
            raise ValueError(
                "disable_padded_drafter_batch is not supported with "
                "extract_hidden_states method"
            )

        self.supports_mm_inputs = False
        layer_ids = getattr(
            self.draft_model_config.hf_config,
            "eagle_aux_hidden_state_layer_ids",
            None,
        )
        if not layer_ids:
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must be set in the draft "
                "model config for extract_hidden_states method"
            )

        self.num_hidden_states = len(layer_ids)
        assert isinstance(self.dtype, torch.dtype)
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            self.num_hidden_states,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.dtype,
            device=device,
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        del target_model, target_attn_layer_names
        with set_model_tag("extract_hidden_states"):
            return get_model(
                vllm_config=self.vllm_config,
                model_config=self.draft_model_config,
            )

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if len(self.draft_attn_layer_names) != 1:
            raise ValueError(
                "ExtractHiddenStatesModel should have exactly one attention "
                f"layer, found {len(self.draft_attn_layer_names)}"
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode

    def capture(self) -> None:
        return None

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del (
            last_hidden_states,
            num_sampled,
            num_rejected,
            next_prefill_tokens,
            temperature,
            seeds,
            dummy_run,
            mm_inputs,
            is_profile,
        )

        draft_tokens = last_sampled[input_batch.idx_mapping, :1]
        if skip_attn_for_dummy_run:
            return draft_tokens
        if aux_hidden_states is None:
            raise ValueError(
                "aux_hidden_states are required when using extract_hidden_states"
            )
        if len(aux_hidden_states) != self.num_hidden_states:
            raise ValueError(
                f"Expected {self.num_hidden_states} auxiliary hidden states, "
                f"got {len(aux_hidden_states)}"
            )

        stacked_hidden_states = torch.stack(aux_hidden_states, dim=1)
        num_tokens = stacked_hidden_states.shape[0]
        self.hidden_states[:num_tokens].copy_(stacked_hidden_states)

        draft_attn_metadata = {
            name: attn_metadata[name] for name in self.draft_attn_layer_names
        }
        draft_slot_mappings = {
            name: slot_mappings[name][:num_tokens]
            for name in self.draft_attn_layer_names
        }
        with set_forward_context(
            draft_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=draft_slot_mappings,
            is_padding=input_batch.is_padding[:num_tokens],
        ):
            self.model(hidden_states=self.hidden_states[:num_tokens])

        return draft_tokens
