# SPDX-License-Identifier: Apache-2.0

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
from typing import Type

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer

class TeleFLMModel(LlamaModel):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: Type[LlamaDecoderLayer] = LlamaDecoderLayer):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        """
        Refer to µScaling paper: 
        @inproceedings{
            yao2025nanolm,
            title={Nano{LM}: an Affordable {LLM} Study Benchmark via Accurate Loss Prediction across Scales},
            author={Yiqun Yao and Siqi Fan and Xiusheng Huang and Xuezhi Fang and Xiang Li and Ziyi Ni and Xin Jiang and Xuying Meng and Peng Han and Shuo Shang and Kang Liu and Aixin Sun and Yequan Wang},
            booktitle={ICLR 2025 First Workshop on Open Science for Foundation Models},
            year={2025},
            url={https://openreview.net/forum?id=IwaPYg1SCA}
        }
        """
        self.use_mup = self.config.use_mup
        if self.use_mup:
            self.input_mult = self.config.input_mult

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(input_ids)
        if self.use_mup:
            embedding = embedding * self.input_mult
        return embedding

class TeleFLMForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # mup
        self.use_mup = self.config.use_mup
        if self.use_mup:
            self.mup_scale_factor = self.config.mup_scale_factor
            self.output_mult = self.config.output_mult / self.mup_scale_factor
            logit_scale = self.output_mult
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    self.config.vocab_size,
                                                    logit_scale)