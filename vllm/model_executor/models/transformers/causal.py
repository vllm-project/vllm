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

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import regex as re
import torch
from torch import nn
from transformers import PreTrainedModel

from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.transformers.utils import (
    OpaqueBaseModel,
    head_forward_kwargs,
    init_on_device_without_buffers,
    replace_base_model,
    stub_base_model,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)
from vllm.transformers_utils.dynamic_module import get_transformers_model_class

from .base import Base

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

OUTPUT_HEAD_ATTR = "output_head"
"""Attribute in which to store the output head."""


class _OutputEmbeddings:
    """Calls `project` in place of the output embeddings, or is the identity until
    `project` is set. Not an `nn.Module`, so that the weights stay registered only
    in `lm_head`."""

    def __init__(self):
        self.project: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.project is None:
            return hidden_states
        return self.project(hidden_states)


@dataclass
class OutputHead:
    """What a `ForCausalLM` applies after its base model."""

    module: PreTrainedModel | None
    """The `ForCausalLM` with its base model stubbed out, or `None` if it applies
    nothing but its output embeddings, `scale` and `soft_cap`, which `lm_head` and
    `LogitsProcessor` compute on their own."""
    output_embeddings: _OutputEmbeddings
    """Stands in for the output embeddings in `module`."""
    in_features: int
    """The input size of the output embeddings."""
    renames: dict[str, str]
    """Checkpoint name -> vLLM name of the head's weights and their modules."""
    forward_kwargs: dict[str, bool]
    """The `head_forward_kwargs` of `module`."""
    scale: float = 1.0
    soft_cap: float | None = None

    def weights_mapper(self) -> WeightsMapper:
        renames = {
            re.compile(rf"^{re.escape(k)}$"): v
            for k, v in self.renames.items()
            if k != v
        }
        return WeightsMapper(orig_to_new_regex=renames)


def to_output_head(
    model: PreTrainedModel, base_model_cls: type[nn.Module] = OpaqueBaseModel
) -> OutputHead | None:
    """Turn `model`, a `ForCausalLM`, into its output head in place.

    The base model, the submodule of type `base_model_cls`, is replaced with
    `OpaqueBaseModel` (unless it already is one), and the output embeddings with
    `_OutputEmbeddings`, so
    `model(inputs_embeds=h)` applies everything the `ForCausalLM` applies after its
    base model (e.g. a dense -> act -> norm transform, a soft cap, a scale).

    Returns `None` (leaving `model` unchanged) if this is not possible.
    """
    embeddings = model.get_output_embeddings()
    vocab_size = model.config.get_text_config().vocab_size
    if not isinstance(embeddings, nn.Linear) or embeddings.out_features != vocab_size:
        return None
    if not replace_base_model(model, base_model_cls):
        return None
    output_embeddings = _OutputEmbeddings()
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
                setattr(module, child_name, output_embeddings)
    for name, _ in model.named_parameters():
        renames[name] = maybe_prefix(OUTPUT_HEAD_ATTR, name)
    # Quantization configs name modules, not parameters
    for name, new_name in list(renames.items()):
        if "." in name:
            renames.setdefault(name.rpartition(".")[0], new_name.rpartition(".")[0])
    return OutputHead(
        module=model,
        output_embeddings=output_embeddings,
        in_features=embeddings.in_features,
        renames=renames,
        forward_kwargs=head_forward_kwargs(model),
    )


def probe_output_head(
    output_head: OutputHead, hidden_size: int
) -> tuple[float, float | None] | None:
    """The `scale` and `soft_cap` of `LogitsProcessor` which `output_head` applies
    after its output embeddings, which must still be the identity, or `None` if it
    applies anything else."""
    assert (model := output_head.module) is not None
    # Heads with parameters apply more than a scale or a soft cap
    if next(model.parameters(), None) is not None:
        return None
    hidden_states = 50 * torch.randn(1, 3, hidden_size)
    hidden_states[0, 0, :2] = torch.tensor([1.0, 1e6])
    logits = model(inputs_embeds=hidden_states, **output_head.forward_kwargs).logits
    # A scale maps 1 to itself and a soft cap maps large logits to itself
    scale, soft_cap = logits[0, 0, :2].tolist()
    if torch.allclose(logits, scale * hidden_states):
        return scale, None
    if torch.allclose(logits, soft_cap * torch.tanh(hidden_states / soft_cap)):
        return 1.0, soft_cap
    return None


class CausalMixin(VllmModelForTextGeneration, Base):
    embedding_modules = {"lm_head": "output_embeddings"}

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForTextGeneration.__init__ and call the next class in MRO
        super(VllmModelForTextGeneration, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        if not self.pp_group.is_last_rank:
            self.lm_head = PPMissingLayer()
            return

        output_head = self._output_head
        vocab_size = self.text_config.vocab_size
        if output_head is None:
            in_features = self.text_config.hidden_size
            scale = getattr(self.text_config, "logit_scale", 1.0)
            soft_cap = getattr(self.text_config, "final_logit_softcapping", None)
        else:
            in_features = output_head.in_features
            scale, soft_cap = output_head.scale, output_head.soft_cap
        self.logits_processor = LogitsProcessor(
            vocab_size, scale=scale, soft_cap=soft_cap
        )
        self.lm_head = ParallelLMHead(
            vocab_size,
            in_features,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self._get_tie_word_embeddings():
            for module in self.model.get_input_embeddings().modules():
                if isinstance(module, VocabParallelEmbedding):
                    self.lm_head = self.lm_head.tie_weights(module)
                    break
        if output_head is not None:
            self._use_output_head(output_head)
        # Batch-sharded sampling checks for this, and kept heads need the full vocab
        if output_head is None or output_head.module is None:
            self.compute_logits_local = lambda hidden_states: self.logits_processor(
                self.lm_head, hidden_states, self.lm_head.bias, skip_gather=True
            )

    def _create_hf_to_vllm_mapper(self):
        super()._create_hf_to_vllm_mapper()
        # Created here so the quantization config also gets the head's renames
        self._output_head = None
        if self.pp_group.is_last_rank:
            self._output_head = self._create_output_head()
        if self._output_head is not None:
            # Before the base model's mappings, which expect the base model's names
            mapper = self._output_head.weights_mapper()
            self.hf_to_vllm_mapper = mapper | self.hf_to_vllm_mapper

    def _create_output_head(self) -> OutputHead | None:
        """The output head of the model's Transformers class, or `None` if there is
        none, in which case `lm_head` and the config's logit scaling are assumed."""
        for architecture in self.config.architectures or []:
            model_cls = get_transformers_model_class(architecture, self.model_config)
            if model_cls is not None:
                break
        else:
            return None
        # Tying would look for the input embeddings in the base model, a stub
        config = copy.deepcopy(self.config)
        config.tie_word_embeddings = False
        config.get_text_config().tie_word_embeddings = False
        hidden_size = self.text_config.hidden_size
        try:
            with init_on_device_without_buffers("meta"), stub_base_model(self.model):
                model = model_cls._from_config(config, dtype=self.model_config.dtype)
            output_head = to_output_head(model.eval())
            if output_head is None:
                return None
            # Fails for heads which need more of the base model than its output
            hidden_states = torch.empty(1, 3, hidden_size, device="meta")
            model(inputs_embeds=hidden_states, **output_head.forward_kwargs)
        except Exception as e:
            logger.warning(
                "Could not run the output head of %s, so `lm_head` and the "
                "config's logit scaling are assumed: %r",
                model_cls.__name__,
                e,
            )
            return None
        if logit_transform := probe_output_head(output_head, hidden_size):
            output_head.scale, output_head.soft_cap = logit_transform
            output_head.module = None
        return output_head

    def _use_output_head(self, output_head: OutputHead):
        """Run the head on the logits of `lm_head`, unless it is only the output
        embeddings."""
        if (module := output_head.module) is None:
            return
        # The head runs after the projection, so every rank needs all logits
        self.logits_processor.use_all_gather = True
        output_head.output_embeddings.project = self._project
        self.replace_modules(module, prefix=OUTPUT_HEAD_ATTR)
        self.init_parameters(module)
        self.output_head_kwargs = output_head.forward_kwargs
        setattr(self, OUTPUT_HEAD_ATTR, module)

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The output embeddings, which give the logits on every TP rank."""
        flat = hidden_states.flatten(0, -2)
        logits = self.logits_processor(self.lm_head, flat, self.lm_head.bias)
        return logits.unflatten(0, hidden_states.shape[:-1])

    def load_weights(self, weights: Iterable[tuple[str, "torch.Tensor"]]) -> set[str]:
        """A thin wrapper around `Base.load_weights` to handle the lm_head bias."""
        lm_head_bias = set()
        load_device = self.vllm_config.load_config.device or self.device_config.device

        def auto_load_lm_head_bias(weights):
            for name, weight in weights:
                is_bias = name.endswith("bias") and self.pp_group.is_last_rank
                if is_bias and self.hf_to_vllm_mapper.map_name(name) == "lm_head.bias":
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
            return self.logits_processor(self.lm_head, hidden_states, self.lm_head.bias)
        output = output_head(
            inputs_embeds=hidden_states.unsqueeze(0), **self.output_head_kwargs
        )
        return output.logits.squeeze(0)
