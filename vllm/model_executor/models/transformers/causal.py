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

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import PreTrainedModel

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.transformers.utils import (
    head_forward_kwargs,
    init_on_device_without_buffers,
    replace_base_model,
)
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.transformers_utils.dynamic_module import get_transformers_model_class

from .base import Base

if TYPE_CHECKING:
    from vllm.config import VllmConfig


OUTPUT_HEAD_ATTR = "output_head"
"""Attribute in which to store the output head."""


class _OutputEmbeddings:
    """Calls `project` in place of the output embeddings. Not an `nn.Module`, so
    that the weights stay registered only in `lm_head`."""

    def __init__(self, project: Callable[[torch.Tensor], torch.Tensor]):
        self.project = project

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.project(hidden_states)


def to_output_head(
    model: PreTrainedModel,
    base_model_cls: type[nn.Module],
    project: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[nn.Linear, dict[str, str]] | None:
    """Turn `model`, a `ForCausalLM`, into its output head in place.

    The base model, the submodule of type `base_model_cls`, is replaced with
    `OpaqueBaseModel`, so `model(inputs_embeds=h)` applies everything the
    `ForCausalLM` applies after its base model (e.g. a dense -> act -> norm
    transform, a soft cap, a scale), with `project` in place of the output
    embeddings.

    Returns the replaced output embeddings and the checkpoint name -> `lm_head` name
    of their parameters, or `None` (leaving `model` unchanged) if this is not
    possible.
    """
    try:
        model.get_input_embeddings()
    except NotImplementedError:
        return None
    embeddings = model.get_output_embeddings()
    vocab_size = model.config.get_text_config().vocab_size
    if not isinstance(embeddings, nn.Linear) or embeddings.out_features != vocab_size:
        return None
    if not replace_base_model(model, base_model_cls):
        return None
    params = {id(embeddings.weight): "lm_head.weight"}
    if embeddings.bias is not None:
        params[id(embeddings.bias)] = "lm_head.bias"
    renames: dict[str, str] = {}
    for name, module in list(model.named_modules(remove_duplicate=False)):
        for param_name, param in list(module.named_parameters(recurse=False)):
            if id(param) in params:
                renames[maybe_prefix(name, param_name)] = params[id(param)]
                # Aliases (e.g. Roberta's `lm_head.bias`) are loaded into `lm_head`
                if module is not embeddings:
                    setattr(module, param_name, None)
        for child_name, child in list(module.named_children()):
            if child is embeddings:
                delattr(module, child_name)
                setattr(module, child_name, _OutputEmbeddings(project))
    return embeddings, renames


def probe_output_head(model: PreTrainedModel, hidden_size: int) -> bool:
    """Whether the output head `model` (see `to_output_head`) applies nothing but
    its output embeddings."""
    # Heads with parameters still have them on "meta", so can only be run there
    has_parameters = next(model.parameters(), None) is not None
    device = "meta" if has_parameters else "cpu"
    output_embeddings = [
        value
        for module in model.modules()
        for value in vars(module).values()
        if isinstance(value, _OutputEmbeddings)
    ]
    projections = [e.project for e in output_embeddings]
    for e in output_embeddings:
        e.project = lambda hidden_states: hidden_states
    hidden_states = torch.randn(1, 3, hidden_size, device=device)
    try:
        logits = model(inputs_embeds=hidden_states, **head_forward_kwargs(model)).logits
    finally:
        for e, project in zip(output_embeddings, projections):
            e.project = project
    return not has_parameters and torch.equal(logits, hidden_states)


class CausalMixin(VllmModelForTextGeneration, Base):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForTextGeneration.__init__ and call the next class in MRO
        super(VllmModelForTextGeneration, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        self.output_head_renames: dict[str, str] = {}
        """Checkpoint name -> vLLM name, for the output head's weights."""
        self.output_head_kwargs: dict[str, bool] = {}
        """Arguments for the output head's forward, from `head_forward_kwargs`."""

        if self.pp_group.is_last_rank:
            found = self._create_output_head()
            output_head, in_features = found or (None, self.text_config.hidden_size)
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                in_features,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if self._get_tie_word_embeddings():
                for module in self.model.get_input_embeddings().modules():
                    if isinstance(module, VocabParallelEmbedding):
                        self.lm_head = self.lm_head.tie_weights(module)
                        break

            if found is not None:
                self.logits_processor = LogitsProcessor(self.text_config.vocab_size)
            else:
                self.logits_processor = LogitsProcessor(
                    self.text_config.vocab_size,
                    scale=getattr(self.text_config, "logit_scale", 1.0),
                    soft_cap=getattr(self.text_config, "final_logit_softcapping", None),
                )
            if output_head is not None:
                # The head runs after the projection, so every rank needs all logits
                self.logits_processor.use_all_gather = True
                self.recursive_replace(output_head, prefix=OUTPUT_HEAD_ATTR)
                self.init_parameters(output_head)
                setattr(self, OUTPUT_HEAD_ATTR, output_head)
        else:
            self.lm_head = PPMissingLayer()

    def _create_output_head(self) -> tuple[PreTrainedModel | None, int] | None:
        """The output head of the model's Transformers class (see `to_output_head`)
        and the input size of its output embeddings, or `None` if there is none, in
        which case `lm_head` and the config's logit scaling are assumed.

        The head is `None` if it is only the output embeddings, which `lm_head`
        computes without the overhead of calling the head."""
        for architecture in self.config.architectures or []:
            model_cls = get_transformers_model_class(architecture, self.model_config)
            if model_cls is not None:
                break
        else:
            return None
        with init_on_device_without_buffers("meta"):
            model = model_cls._from_config(self.config, dtype=self.model_config.dtype)
        output_head = to_output_head(model, type(self.model), self._project)
        if output_head is None:
            return None
        embeddings, renames = output_head
        self.output_head_renames = renames
        if probe_output_head(model, self.text_config.hidden_size):
            return None, embeddings.in_features
        for name, _ in model.named_parameters():
            self.output_head_renames[name] = maybe_prefix(OUTPUT_HEAD_ATTR, name)
        self.output_head_kwargs = head_forward_kwargs(model)
        return model, embeddings.in_features

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The output embeddings, which give the logits on every TP rank."""
        return self.logits_processor(self.lm_head, hidden_states, self.lm_head.bias)

    def load_weights(self, weights: Iterable[tuple[str, "torch.Tensor"]]) -> set[str]:
        """A thin wrapper around `Base.load_weights` to handle the output head."""
        lm_head_bias = set()
        load_device = self.vllm_config.load_config.device or self.device_config.device

        def auto_load_lm_head_bias(weights):
            for name, weight in weights:
                name = self.output_head_renames.get(name, name)
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
        output_head = getattr(self, OUTPUT_HEAD_ATTR, None)
        if output_head is None:
            return self._project(hidden_states)
        output = output_head(
            inputs_embeds=hidden_states.unsqueeze(0), **self.output_head_kwargs
        )
        return output.logits.squeeze(0)
