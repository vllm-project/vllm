# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/deepseek_mtp.py
# Copyright 2025 Xiaomi Corporation.
# Copyright 2023 The vLLM team.
# Copyright 2024 DeepSeek-AI team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MiMo-MTP model."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
from vllm.sequence import IntermediateTensors

from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix


class MiMoMultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        self.token_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.mtp_block = Qwen2DecoderLayer(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds[positions == 0] = 0
        inputs_embeds = self.token_layernorm(inputs_embeds)
        previous_hidden_states = self.hidden_layernorm(previous_hidden_states)

        hidden_states = self.input_proj(
            torch.cat([previous_hidden_states, inputs_embeds], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states = residual + hidden_states
        return self.final_layernorm(hidden_states)


class MiMoMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.mtp_layers = torch.nn.ModuleDict(
            {
                str(idx): MiMoMultiTokenPredictorLayer(
                    config,
                    f"{prefix}.layers.{idx}",
                    model_config=vllm_config.model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return self.mtp_layers[str(self.mtp_start_layer_idx + spec_step_idx)](
            inputs_embeds,
            positions,
            previous_hidden_states,
            spec_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        self.mtp_layers[str(self.mtp_start_layer_idx + spec_step_idx)]
        logits = self.logits_processor(lm_head, hidden_states)
        return logits


class MiMoMTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = MiMoMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        # Checkpoint stores MTP layers 0-indexed and without the `mtp_block`
        # wrapper around the transformer block; remap onto the offset index.
        start = self.config.num_hidden_layers
        self.hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_substr={
                ".self_attn.": ".mtp_block.self_attn.",
                ".mlp.": ".mtp_block.mlp.",
                ".input_layernorm.": ".mtp_block.input_layernorm.",
                ".post_attention_layernorm.": ".mtp_block.post_attention_layernorm.",
            },
            orig_to_new_stacked={
                ".q_proj": (".qkv_proj", "q"),
                ".k_proj": (".qkv_proj", "k"),
                ".v_proj": (".qkv_proj", "v"),
                ".gate_proj": (".gate_up_proj", 0),
                ".up_proj": (".gate_up_proj", 1),
            },
            orig_to_new_prefix={
                f"model.mtp_layers.{i}.": f"model.mtp_layers.{i + start}."
                for i in range(self.config.num_nextn_predict_layers)
            },
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        assert spec_step_idx == 0, "mimo_mtp only support predict one token now"
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, self.lm_head, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The checkpoint carries the full model; keep only the MTP layers and
        # the shared embedding/head.
        def mtp_weights():
            for name, weight in weights:
                if "mtp_layers" in name or "embed_tokens" in name or "lm_head" in name:
                    yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(mtp_weights(), mapper=self.hf_to_vllm_mapper)
