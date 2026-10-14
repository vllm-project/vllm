# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
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
"""Inference-only Qwen3 model compatible with HuggingFace weights."""

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer as Qwen3DecoderLayerBase
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM as Qwen3ForCausalLMBase
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import (
    is_sp_active,
    mark_sp_region,
    sp_all_gather,
    sp_shard,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class Qwen3DecoderLayer(Qwen3DecoderLayerBase):
    """Identical to the upstream layer -- this vendor fork exists so
    sequence-parallel and other NVIDIA-specific behavior can be layered on
    top without touching the upstream model."""

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        per_layer_sliding_window: int | None = None,
    ) -> None:
        super().__init__(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            per_layer_sliding_window=per_layer_sliding_window,
        )


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )
        self.parallel_config = vllm_config.parallel_config
        if not self.parallel_config.use_sequence_parallel:
            return
        assert get_pp_group().world_size == 1, (
            "Sequence parallelism is not supported with pipeline parallelism"
        )
        pc = self.parallel_config
        # Each layer has two SP regions: [prev down_proj -> qkv_proj] and
        # [o_proj -> gate_up_proj].
        prev_down_proj = None  # first region starts at the sharded embeddings
        for layer in self.layers[self.start_layer : self.end_layer]:
            mark_sp_region(prev_down_proj, layer.self_attn.qkv_proj, pc)
            mark_sp_region(layer.self_attn.o_proj, layer.mlp.gate_up_proj, pc)
            prev_down_proj = layer.mlp.down_proj
        # Last region ends at the gathered final norm.
        mark_sp_region(prev_down_proj, None, pc)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if not is_sp_active(self.parallel_config):
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        output = super().forward(
            input_ids, positions, intermediate_tensors, sp_shard(inputs_embeds)
        )
        num_tokens = positions.shape[-1]
        if isinstance(output, tuple):
            hidden_states, aux_hidden_states = output
            return sp_all_gather(hidden_states)[:num_tokens], [
                sp_all_gather(x)[:num_tokens] for x in aux_hidden_states
            ]
        return sp_all_gather(output)[:num_tokens]


class Qwen3ForCausalLM(Qwen3ForCausalLMBase):
    hf_to_vllm_mapper = Qwen3Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
