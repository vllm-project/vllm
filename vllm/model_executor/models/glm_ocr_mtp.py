# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 The ZhipuAI Team.
# Copyright 2026 The vLLM team.
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
"""Inference-only GLM-OCR MTP model compatible with HuggingFace weights."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .glm4 import Glm4DecoderLayer, get_spec_layer_idx_from_weight_name
from .glm4_moe_lite_mtp import (
    Glm4MoeLiteMultiTokenPredictor,
    Glm4MoeLiteMultiTokenPredictorLayer,
)
from .interfaces import SupportsPP
from .utils import (
    is_pp_missing_parameter,
    maybe_prefix,
)


class SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class GlmOcrMultiTokenPredictorLayer(Glm4MoeLiteMultiTokenPredictorLayer):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.device = current_platform.device_type

        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        self.mtp_block = Glm4DecoderLayer(
            vllm_config=vllm_config, prefix=prefix, config=self.config
        )


class GlmOcrMultiTokenPredictor(Glm4MoeLiteMultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): GlmOcrMultiTokenPredictorLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )


class GlmOcrMTP(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = GlmOcrMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.expert_weights = []
        self.num_layers = self.config.num_nextn_predict_layers
        for layer in self.model.layers.values():
            assert isinstance(layer, GlmOcrMultiTokenPredictorLayer)
            layer = layer.mtp_block
            assert isinstance(layer, Glm4DecoderLayer)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            name = self._rewrite_spec_layer_name(spec_layer, name)

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name or "zero_point" in name:
                # Remapping the name of FP8 kv-scale or zero point.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        and rename shared layer weights to be top level.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        elif shared_weight:
            # treat shared weights as top level weights
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name
