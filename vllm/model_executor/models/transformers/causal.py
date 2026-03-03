# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
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
"""Transformers modeling backend mixin for causal language models."""

from typing import TYPE_CHECKING

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig


class CausalMixin(VllmModelForTextGeneration):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForTextGeneration.__init__ and call the next class in MRO
        super(VllmModelForTextGeneration, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        # Tell `Base.load_weights` to skip
        # `lm_head` if the model has tied word embeddings
        tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)
        if tie_word_embeddings:
            self.skip_prefixes.append("lm_head.")

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.get_input_embeddings()
                )

            logit_scale = getattr(self.text_config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                self.text_config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

    def compute_logits(self, hidden_states: "torch.Tensor") -> "torch.Tensor | None":
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
