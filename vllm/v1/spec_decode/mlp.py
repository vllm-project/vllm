# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import torch
import torch.nn as nn

from vllm.config import (ModelConfig, ModelDType, ParallelConfig,
                         SpeculativeConfig, VllmConfig,
                         set_current_vllm_config)
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.mlp_speculator import SQRT2

logger = init_logger(__name__)


@dataclass
class MlpProposer:
    vllm_config: VllmConfig
    device: torch.device

    model_config: ModelConfig = field(init=False, repr=False)
    speculative_config: SpeculativeConfig = field(init=False, repr=False)
    draft_model_config: ModelConfig = field(init=False, repr=False)
    parallel_config: ParallelConfig = field(init=False, repr=False)
    model: nn.Module = field(init=False, repr=False)
    max_num_tokens: int = field(init=False, repr=False)
    hidden_size: int = field(init=False, repr=False)
    dtype: Union[ModelDType, torch.dtype] = field(init=False, repr=False)

    def __post_init__(self):
        if self.vllm_config.speculative_config is None:
            raise ValueError(
                "'speculative_config' cannot be None when using 'mlp_speculator'"
            )

        self.model_config = self.vllm_config.model_config
        self.speculative_config = self.vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.max_num_tokens = \
            self.vllm_config.scheduler_config.max_num_batched_tokens
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.dtype = self.model_config.dtype

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int) -> None:
        hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        with set_forward_context(None, self.vllm_config,
                                 num_tokens=num_tokens):
            self.model(hidden_states)

    def propose(
        self,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        # [batch_size + 1] starting with 0
        cu_num_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(cu_num_tokens) - 1
        last_hidden_states = []
        for i in range(batch_size):
            start = cu_num_tokens[i].item()
            end = cu_num_tokens[i + 1].item()
            if end > start:
                # Take the last hidden state for this request
                last_hidden_states.append(target_hidden_states[end - 1])
            else:
                # Handle empty case (shouldn't happen in normal operation)
                last_hidden_states.append(
                    torch.zeros_like(target_hidden_states[0]))

        previous_hidden_states = torch.stack(
            last_hidden_states)  # [batch_size, hidden_size]

        num_predict_tokens = self.speculative_config.num_speculative_tokens
        draft_tokens = []

        # Start with the current hidden states
        current_hidden_states = previous_hidden_states
        last_tokens = next_token_ids  # [batch_size]

        if self.model.scale_input:
            current_hidden_states = self.model.ln0(
                current_hidden_states) / SQRT2

        for head_idx in range(num_predict_tokens):
            z = self.model.emb[head_idx](last_tokens.unsqueeze(1))  # b k d
            # [batch_size, 1, hidden_size]
            states = self.model.proj[head_idx](
                current_hidden_states.unsqueeze(1))

            # Weighted combination as in MLPSpeculator
            states.add_(
                z,
                alpha=self.model.emb_weight / self.model.state_weight,
            )
            # [batch_size, 1, hidden_size]
            states = self.model.activation(self.model.ln[head_idx](states))

            # Compute logits for this head
            # [batch_size, vocab_size]
            logits = self.model.head[head_idx](states.squeeze(1))

            # [batch_size]
            sampled_tokens = torch.argmax(logits, dim=-1)
            draft_tokens.append(sampled_tokens)

            # Update for next iteration
            current_hidden_states = states.squeeze(1)
            last_tokens = sampled_tokens

        # Stack draft tokens: [batch_size, num_predict_tokens]
        draft_token_ids = torch.stack(draft_tokens, dim=1)

        return draft_token_ids

    def load_model(self, target_model: nn.Module) -> None:
        loader = get_model_loader(self.vllm_config.load_config)

        # FIXME: This does not handle with distributed inference.
        target_device = self.vllm_config.device_config.device
        # We need to set the vllm_config here to register attention
        # layers in the forward context.
        with set_default_torch_dtype(self.draft_model_config.dtype), \
                set_current_vllm_config(self.vllm_config):
            draft_cls, arch = ModelRegistry.resolve_model_cls(
                self.draft_model_config.architectures)
            self.model = draft_cls(
                vllm_config=self.vllm_config).to(target_device)

        weights = loader.get_all_weights(self.draft_model_config, self.model)
        self.model.load_weights(weights)
        # TODO: support PP
