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
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.config import VllmConfig

logger = init_logger(__name__)

PROBE_HIDDEN_STATES = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]])
"""Distinct values, so that a transform which is not a plain scale is detectable."""


class ProbeOutput(dict):
    """Decoder output holding known hidden states, with every other field unset."""

    last_hidden_state = PROBE_HIDDEN_STATES

    def __getattr__(self, name: str) -> None:
        return None

    def __getitem__(self, index: Any) -> torch.Tensor:
        return PROBE_HIDDEN_STATES

    def to_tuple(self) -> tuple[torch.Tensor]:
        return (PROBE_HIDDEN_STATES,)


class ProbeDecoder(nn.Module):
    """Decoder returning known hidden states, however the wrapper reaches it."""

    def forward(self, *args, **kwargs) -> ProbeOutput:
        return ProbeOutput()

    def __getattr__(self, name: str) -> nn.Module:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self


class ProbeHead(nn.Identity):
    """Head passing the hidden states through, with the `weight` wrappers read."""

    weight = PROBE_HIDDEN_STATES


def get_logit_scale(
    text_config: "PretrainedConfig", trust_remote_code: bool = False
) -> float:
    """The scale Transformers applies to the logits after the decoder.

    This backend replaces the `ForCausalLM` wrapper, which is where Transformers
    scales the logits, so the scale has to be folded into our logits processor.
    There is no config field or API to read it from, and the conventions differ
    per model (`logit_scale` multiplies, `logits_scaling` divides except under
    muP, `logits_mup_width_multiplier` divides the hidden states, ...), so run
    the wrapper's own `forward` over known hidden states and measure it.
    """
    try:
        scale = probe_logit_scale(text_config, trust_remote_code)
    except Exception:
        logger.warning_once(
            "Could not determine whether %s scales the logits, assuming it does "
            "not. Logprobs may be wrong if it does.",
            text_config.model_type,
        )
        return 1.0

    if scale is None:
        logger.warning_once(
            "%s does not scale the logits linearly, which this backend cannot "
            "reproduce (e.g. final logit softcapping). Logprobs may be wrong.",
            text_config.model_type,
        )
        return 1.0
    return scale


def probe_logit_scale(
    text_config: "PretrainedConfig", trust_remote_code: bool = False
) -> float | None:
    """The measured logit scale, or `None` if the transform is not a plain scale.

    Raises whatever running the wrapper's `forward` raises, so that callers can
    tell "measured no scaling" from "could not measure".
    """
    model = build_logit_scale_probe(text_config, trust_remote_code)
    logits = type(model).forward(model, inputs_embeds=PROBE_HIDDEN_STATES).logits
    scale = (logits / PROBE_HIDDEN_STATES).flatten()
    if not torch.allclose(scale, scale[0].expand_as(scale)):
        return None
    return scale[0].item()


def build_logit_scale_probe(
    text_config: "PretrainedConfig", trust_remote_code: bool
) -> nn.Module:
    """A `ForCausalLM` whose decoder and head are stubs, so only scaling is left."""
    model_cls = MODEL_FOR_CAUSAL_LM_MAPPING.get(type(text_config), None)
    if model_cls is not None:
        # Skip `__init__` so that no weights are allocated
        model = model_cls.__new__(model_cls)
        nn.Module.__init__(model)
        model.config = text_config
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                text_config, trust_remote_code=trust_remote_code
            )
    decoder = ProbeDecoder()
    # `__init__` caches config fields that `forward` reads off the model itself,
    # or off the decoder, depending on the model
    for name, value in vars(text_config).items():
        if not name.startswith("_") and isinstance(value, (int, float)):
            setattr(model, name, value)
            setattr(decoder, name, value)
    setattr(model, type(model).base_model_prefix, decoder)
    model.lm_head = ProbeHead()
    return model


class CausalMixin(VllmModelForTextGeneration):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForTextGeneration.__init__ and call the next class in MRO
        super(VllmModelForTextGeneration, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        # Tell `Base.load_weights` to skip
        # `lm_head` if the model has tied word embeddings
        tie_word_embeddings = self._get_tie_word_embeddings()
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
                for module in self.model.get_input_embeddings().modules():
                    if isinstance(module, VocabParallelEmbedding):
                        self.lm_head = self.lm_head.tie_weights(module)
                        break

            logit_scale = get_logit_scale(
                self.text_config, self.model_config.trust_remote_code
            )
            self.logits_processor = LogitsProcessor(
                self.text_config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

    def load_weights(self, weights: Iterable[tuple[str, "torch.Tensor"]]) -> set[str]:
        """A thin wrapper around `Base.load_weights` to handle the lm_head bias."""

        lm_head_bias = set()

        def auto_load_lm_head_bias(weights):
            for name, weight in weights:
                if name.endswith("lm_head.bias") and self.pp_group.is_last_rank:
                    self.lm_head._register_bias()
                    self.lm_head.bias.weight_loader(self.lm_head.bias, weight)
                    lm_head_bias.add(name)
                else:
                    yield name, weight

        return super().load_weights(auto_load_lm_head_bias(weights)) | lm_head_bias

    def compute_logits(self, hidden_states: "torch.Tensor") -> "torch.Tensor | None":
        logits = self.logits_processor(self.lm_head, hidden_states, self.lm_head.bias)
        return logits
