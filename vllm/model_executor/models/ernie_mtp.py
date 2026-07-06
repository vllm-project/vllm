# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The Baidu team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
"""Inference-only Ernie-MTP model."""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from .llama import LlamaDecoderLayer
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix


class ErnieMultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.mtp_emb_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_linear_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        # The MTP block's output is consumed by an explicit residual add below
        # (no finalize_norm), so it must reduce itself rather than defer.
        self.mtp_block = LlamaDecoderLayer(vllm_config, prefix, reduce_results=True)

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

        inputs_embeds = self.mtp_emb_norm(inputs_embeds)
        previous_hidden_states = self.mtp_hidden_norm(previous_hidden_states)

        hidden_states = self.mtp_linear_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states = residual + hidden_states

        return hidden_states


class ErnieMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): ErnieMultiTokenPredictorLayer(
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
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
        return self.layers[str(self.mtp_start_layer_idx + spec_step_idx)](
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
        self.layers[str(self.mtp_start_layer_idx + spec_step_idx)]
        logits = self.logits_processor(lm_head, hidden_states)
        return logits


class ErnieMTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        # MTP weights are stored under a flat `mtp_*.0.` block in the
        # checkpoint; rewrite them into `model.layers.{spec_layer}.*`.
        spec_layer = self.config.num_hidden_layers
        self.hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_substr={
                "model.mtp_emb_norm.0.": f"model.layers.{spec_layer}.mtp_emb_norm.",
                "model.mtp_hidden_norm.0.": (
                    f"model.layers.{spec_layer}.mtp_hidden_norm."
                ),
                "model.mtp_linear_proj.0.": (
                    f"model.layers.{spec_layer}.mtp_linear_proj."
                ),
                "model.mtp_block.0.": f"model.layers.{spec_layer}.mtp_block.",
            },
            orig_to_new_stacked={
                ".q_proj": (".qkv_proj", "q"),
                ".k_proj": (".qkv_proj", "k"),
                ".v_proj": (".qkv_proj", "v"),
                ".gate_proj": (".gate_up_proj", 0),
                ".up_proj": (".gate_up_proj", 1),
            },
        )
        self.model = ErnieMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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
        assert spec_step_idx == 0, "ernie_mtp only support predict one token"
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
        # Checkpoint bundles the full base model; only MTP, embed_tokens and
        # lm_head weights belong to this module.
        def _filter(
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if any(k in name for k in ("mtp", "embed_tokens", "lm_head")):
                    yield name, weight

        skip_prefixes = ["lm_head"] if self.config.tie_word_embeddings else []
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(_filter(weights), mapper=self.hf_to_vllm_mapper)
