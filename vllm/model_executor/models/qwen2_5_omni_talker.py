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
"""Inference-only Qwen2-Audio model compatible with HuggingFace weights."""
from functools import cached_property
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTalkerConfig

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniConditionalGenerationMixin,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
    Qwen2_5OmniThinkerDummyInputsBuilder)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper,
                    init_vllm_registered_model,
                    maybe_prefix)

@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class Qwen2_5OmniTalkerForConditionalGeneration(nn.Module, SupportsMultiModal,
                                                SupportsPP,
                                                Qwen2_5OmniConditionalGenerationMixin):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen2_5OmniTalkerConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        if hasattr(config, "talker_config"):
            self.config = config.talker_config
        else:
            self.config = config

        self.thinker_to_talker_proj = ColumnParallelLinear(
            self.config.embedding_size,
            self.config.hidden_size,
            bias=True,
            gather_output=True,
            skip_bias_add=False,
            quant_config=quant_config,
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=getattr(self.config, 'text_config', self.config),
            architectures=["Qwen2ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        forward_context: ForwardContext = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            # for profile_run:
            inputs_embeds = self.get_input_embeddings(input_ids)
        else:
            # in the decoding stage, the "input_embeds" means "thinker_reply_part"
            if attn_metadata.prefill_metadata is None:
                inputs_embeds += self.get_input_embeddings(input_ids)

        input_ids = None

        # projection
        inputs_embeds, _ = self.thinker_to_talker_proj(inputs_embeds)

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "talker.codec_head.": "language_model.lm_head.",
                "talker.model.": "language_model.model.",
                "talker.": "",
            })

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=['thinker.', 'token2wav.'],
        )
        return loader.load_weights(weights, mapper=hf_to_vllm_mapper)
