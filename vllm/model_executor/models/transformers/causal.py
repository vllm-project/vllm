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

from collections.abc import Iterable
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.models.auto.auto_factory import _get_model_class

from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix

from .base import Base

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

LM_HEAD_TRANSFORM_ATTR = "lm_head_transform"
"""Attribute in which to store the module that transforms the hidden states before the
output embeddings, for models which have one (see `CausalMixin._output_head`)."""


class OutputHead(NamedTuple):
    embeddings: str = "lm_head"
    """Name of the output embeddings in the Transformers `ForCausalLM` class."""
    transform: str | None = None
    """Name of the module that transforms the hidden states before the output
    embeddings, if the model has one."""
    transform_module: nn.Module | None = None


class CausalMixin(VllmModelForTextGeneration, Base):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForTextGeneration.__init__ and call the next class in MRO
        super(VllmModelForTextGeneration, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        tie_word_embeddings = self._get_tie_word_embeddings()

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if tie_word_embeddings:
                for module in self.model.get_input_embeddings().modules():
                    if isinstance(module, VocabParallelEmbedding):
                        self.lm_head = self.lm_head.tie_weights(module)
                        break

            self.logits_processor = LogitsProcessor(
                self.text_config.vocab_size,
                scale=getattr(self.text_config, "logit_scale", 1.0),
                soft_cap=getattr(self.text_config, "final_logit_softcapping", None),
            )

            if (transform := self._output_head.transform_module) is not None:
                self.init_parameters(transform)
                setattr(self, LM_HEAD_TRANSFORM_ATTR, transform)
        else:
            self.lm_head = PPMissingLayer()

    @cached_property
    def _output_head(self) -> OutputHead:
        """Find the output head of the model's Transformers `ForCausalLM` class.

        Most models project the hidden states with an output embedding called
        `lm_head`. Some (e.g. ModernBERT decoder) first pass them through a module
        of their own (dense -> activation -> norm) and name the output embeddings
        differently. Only classes which override `get_output_embeddings` can do
        this, so the `ForCausalLM` class is only built (on `meta`) for those.
        """
        mapping = AutoModelForCausalLM._model_mapping
        if type(self.config) not in mapping:
            return OutputHead()
        model_class = _get_model_class(self.config, mapping)
        if model_class.get_output_embeddings is PreTrainedModel.get_output_embeddings:
            return OutputHead()

        with torch.device("meta"):
            causal_lm = model_class._from_config(
                self.config, dtype=self.model_config.dtype
            )
        output_embeddings = causal_lm.get_output_embeddings()
        children = {m: n for n, m in causal_lm.named_children()}
        if output_embeddings not in children:
            return OutputHead()

        # Modules which are neither the base model nor the output embeddings
        base_model_class = type(self.model)
        others = [
            (name, module)
            for name, module in causal_lm.named_children()
            if module is not output_embeddings
            and not isinstance(module, base_model_class)
            and next(module.parameters(), None) is not None
        ]
        if len(others) > 1:
            logger.warning_once(
                "%s has more than one module besides its base model and output "
                "embeddings (%s). The Transformers modeling backend cannot tell the "
                "order in which they are applied, so it will ignore them.",
                model_class.__name__,
                ", ".join(name for name, _ in others),
            )
            others = []
        transform, transform_module = others[0] if others else (None, None)
        output_head = OutputHead(
            children[output_embeddings], transform, transform_module
        )
        logger.debug("Identified output head as: %s", output_head)
        return output_head

    def _rename_output_head(self, name: str) -> str:
        """Rename the output head weights to match this module's attributes."""
        embeddings, transform, _ = self._output_head
        if transform is not None and name.startswith(f"{transform}."):
            return LM_HEAD_TRANSFORM_ATTR + name.removeprefix(transform)
        if embeddings != "lm_head" and name.startswith(f"{embeddings}."):
            return "lm_head" + name.removeprefix(embeddings)
        return name

    def load_weights(self, weights: Iterable[tuple[str, "torch.Tensor"]]) -> set[str]:
        """A thin wrapper around `Base.load_weights` to handle the output head."""
        lm_head_bias = set()
        load_device = self.vllm_config.load_config.device or self.device_config.device

        def auto_load_lm_head_bias(weights):
            for name, weight in weights:
                name = self._rename_output_head(name)
                if name.endswith("lm_head.bias") and self.pp_group.is_last_rank:
                    # load_weights is called outside vLLM's device context,
                    # so we must enter it again when registering the bias.
                    with torch.device(load_device):
                        self.lm_head._register_bias()
                    self.lm_head.bias.weight_loader(self.lm_head.bias, weight)
                    lm_head_bias.add(name)
                else:
                    yield name, weight

        return super().load_weights(auto_load_lm_head_bias(weights)) | lm_head_bias

    def compute_logits(self, hidden_states: "torch.Tensor") -> "torch.Tensor | None":
        if (transform := getattr(self, LM_HEAD_TRANSFORM_ATTR, None)) is not None:
            hidden_states = transform(hidden_states)
        logits = self.logits_processor(self.lm_head, hidden_states, self.lm_head.bias)
        return logits
