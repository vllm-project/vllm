# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
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
"""Inference-only Mistral model compatible with HuggingFace weights."""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import MistralConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
import os

KVCache = Tuple[torch.Tensor, torch.Tensor]


class MistralForCausalLM(nn.Module):

    def __init__(
        self,
        config: MistralConfig,
        linear_method=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = None
        self.lm_head = None
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> SamplerOutput:
        # TODO: drop this line
        batch_size, n_active_tokens = input_ids.shape
        positions = positions # [:1, :]

        with torch.inference_mode():
            seq_ids = []
            block_size = self.model.context_buckets[-1]
            if input_metadata.is_prompt:
                seq_ids = input_metadata.slot_mapping[:, 0] // block_size
            else:
                seq_ids = input_metadata.block_tables
            
            logits = self.model(input_ids, cache_ids=positions, start_ids=seq_ids)
        return logits
        #     next_tokens = self.sampler(logits, input_metadata)
        # return next_tokens
    
    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.model.chkpt_model.lm_head,
                                   hidden_states, sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None,
                     **kwargs):
        from transformers_neuronx.mistral.model import MistralForSampling
        
        split_model_dir = f"{model_name_or_path}-split"
        if os.path.isdir(os.path.join(model_name_or_path,
                                      "pytorch_model.bin")):
            split_model_dir = model_name_or_path
        elif not os.path.exists(f"{model_name_or_path}-split"):
            from transformers import MistralForCausalLM
            from transformers_neuronx.module import save_pretrained_split

            hf_model = MistralForCausalLM.from_pretrained(model_name_or_path,
                                                        low_cpu_mem_usage=True)
            save_pretrained_split(hf_model, f"{model_name_or_path}-split")

        self.model = MistralForSampling.from_pretrained(split_model_dir,
                                                      **kwargs)
        self.model.to_neuron()
