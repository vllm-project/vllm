# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
# All rights reserved.
#
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

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.torchao import TorchAOConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama4 import (Llama4DecoderLayer,
                                               Llama4ForCausalLM)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.multimodal.inputs import NestedTensors

from .utils import AutoWeightsLoader, maybe_prefix, merge_multimodal_embeddings

logger = init_logger(__name__)


@support_torch_compile
class LlamaModel(nn.Module):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = (
            vllm_config.speculative_config.draft_model_config.hf_config)
        self.validate_and_update_config(start_layer_id, quant_config)
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList([
            Llama4DecoderLayer(
                self.config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
            ) for i in range(self.config.num_hidden_layers)
        ])
        self.fc = torch.nn.Linear(self.config.hidden_size * 2,
                                  self.config.hidden_size,
                                  bias=False)
        self.norm = RMSNorm(self.config.hidden_size,
                            eps=self.config.rms_norm_eps)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
        hidden_states = self.fc(
            torch.cat((inputs_embeds, hidden_states), dim=-1))
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
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
            name = name.removeprefix("model.")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # if PP disabled then draft will share embed with target
                if get_pp_group().world_size == 1 and \
                    "embed_tokens." in name:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        for name in params_dict:
            # if PP disabled then draft will share embed with target
            if get_pp_group().world_size == 1 and \
                "embed_tokens." in name:
                continue
            assert name in loaded_params, f"{name} is not loaded!"
        return loaded_params

    def validate_and_update_config(
            self,
            start_layer_id: int,
            quant_config: Optional[QuantizationConfig] = None) -> None:
        # yoco and moe is not supported by draft model yet
        assert self.config.yoco_global_kv_layer is None
        assert self.config.yoco_local_kv_layer is None
        assert len(self.config.moe_layers) == 0
        # draft model layer index is increased by start_layer_id,
        # so we need to pad relevant configs accordingly
        self.config.no_rope_layers = [
            0
        ] * start_layer_id + self.config.no_rope_layers
        # currently only TorchAO quantization is supported
        if isinstance(quant_config, TorchAOConfig):

            def pad_layer_name(layer: str) -> str:
                layer_index = extract_layer_index(layer)
                return layer.replace(str(layer_index),
                                     str(layer_index + start_layer_id))

            quant_config.torchao_config.module_fqn_to_config = {
                pad_layer_name(layer): quantization
                for layer, quantization in
                quant_config.torchao_config.module_fqn_to_config.items()
            }


class EagleLlama4ForCausalLM(Llama4ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = (
            vllm_config.speculative_config.draft_model_config.hf_config)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)
        # draft model quantization config may differ from target model
        quant_config = VllmConfig.get_quantization_config(
            vllm_config.speculative_config.draft_model_config,
            vllm_config.load_config)
        self.model = LlamaModel(vllm_config=vllm_config,
                                prefix="model",
                                start_layer_id=target_layer_num,
                                quant_config=quant_config)
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                scale=logit_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> None:
        loader = AutoWeightsLoader(
            self,
            # lm_head is tied with target model (Llama4ForCausalLM)
            skip_prefixes=(["lm_head."]),
        )

        model_weights = {}
        weights = [
            self.permute_qk_weight_for_rotary(name, loaded_weight)
            for name, loaded_weight in weights
        ]
        for name, loaded_weight in weights:
            if "lm_head" not in name:
                name = "model." + name
            model_weights[name] = loaded_weight

        loader.load_weights(model_weights.items())

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.image_token_index,
            )

        return inputs_embeds
