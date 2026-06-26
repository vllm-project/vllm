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

        # A tied-embedding model may still carry a *separately-learned*
        # `lm_head.bias` (the weight is tied, the bias is not). Detect it so we
        # keep loading + applying the bias while still tying the weight.
        has_lm_head_bias = self._has_lm_head_bias()

        # Tell `Base.load_weights` to skip the tied `lm_head.weight`. We must
        # NOT skip the whole `lm_head.` prefix, or a present `lm_head.bias`
        # would be dropped too.
        tie_word_embeddings = self._get_tie_word_embeddings()
        if tie_word_embeddings:
            self.skip_prefixes.append("lm_head.weight")

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                bias=has_lm_head_bias,
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

    def _has_lm_head_bias(self) -> bool:
        """True if the checkpoint ships a (non-tied) ``lm_head.bias`` tensor.

        HF exporters that emit a learned output bias set an explicit config
        flag; some configs spell it differently, so accept any known alias and
        fall back to the model's own ``lm_head`` module when present.
        """
        for attr in ("lm_head_bias", "use_lm_head_bias", "output_bias"):
            if getattr(self.text_config, attr, None):
                return True
        lm_head = getattr(self.model, "lm_head", None)
        return getattr(lm_head, "bias", None) is not None

    def compute_logits(self, hidden_states: "torch.Tensor") -> "torch.Tensor | None":
        # Apply the learned output bias (if any). The unquantized logits path
        # only honours a bias passed here, never ``lm_head.bias`` directly.
        bias = getattr(self.lm_head, "bias", None)
        logits = self.logits_processor(self.lm_head, hidden_states, bias)
        return logits
