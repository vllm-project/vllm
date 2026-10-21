# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


class MambaAttnHybridSpeculator(DSparkSpeculator):
    _speculator_name = "MambaAttnHybrid"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        hf_config = self.draft_model_config.hf_config
        width = len(hf_config.eagle_aux_hidden_state_layer_ids) * (
            vllm_config.model_config.get_hidden_size()
        )
        self.hidden_states = torch.zeros(
            self.max_num_tokens, width, dtype=self.dtype, device=device
        )
        self.latent_seed = torch.zeros(
            self.max_num_reqs, width, dtype=self.dtype, device=device
        )
        self._num_rejected = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        attention_layers = [
            layer for layer in model.model.layers if hasattr(layer, "self_attn")
        ]
        for layer, target_layer_id in zip(
            attention_layers, model.config.attn_kv_layer_ids, strict=True
        ):
            layer.self_attn.attn.kv_sharing_target_layer_name = (
                f"model.layers.{target_layer_id}.self_attn.attn"
            )
        return model

    def prepare_context_anchor(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor | None,
    ) -> None:
        assert num_rejected is not None
        last_accepted = query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs] - 1
        self.latent_seed[:num_reqs].copy_(
            self.hidden_states[last_accepted.clamp_min(0).long()]
        )

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
            is_padding=self.input_buffers.is_padding[:num_tokens],
        ):
            return self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
                latent_seed=self.latent_seed,
            )
