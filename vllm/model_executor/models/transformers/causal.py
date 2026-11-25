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
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import regex as re
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.transformers.utils import (
    init_on_device_without_buffers,
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


class _BaseModelOutput(BaseModelOutputWithPast):
    """Has every other output a `ForCausalLM` may read (e.g. `cross_attentions`),
    as `None`."""

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return None


class _OpaqueBaseModel(nn.Module):
    """Stands in for `base_model` and for its submodules (e.g. OPT's
    `model.decoder`), returning `inputs_embeds` as its output."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        # Not registered, so the head doesn't own the base model's parameters
        self.__dict__["_base_model"] = base_model

    def forward(self, *args, inputs_embeds: torch.Tensor, **kwargs):
        return _BaseModelOutput(last_hidden_state=inputs_embeds)

    def get_input_embeddings(self) -> nn.Module:
        # Read by `PreTrainedModel.get_output_embeddings`, which returns `None` if
        # this fails
        return self.__dict__["_base_model"].get_input_embeddings()

    def __getattr__(self, name: str):
        base_model = self.__dict__.get("_base_model")
        if base_model is not None and name in base_model._modules:
            return _OpaqueBaseModel(base_model._modules[name])
        return super().__getattr__(name)


_stubbed_base_models: dict[type, nn.Module] = {}


def _new(cls, *args, **kwargs):
    if (base_model := _stubbed_base_models.get(cls)) is not None:
        return _OpaqueBaseModel(base_model)
    return object.__new__(cls)


@contextmanager
def stub_base_model(base_model: nn.Module) -> Iterator[None]:
    """Make the class of `base_model` build `_OpaqueBaseModel(base_model)` instead,
    so that a `ForCausalLM` can be built without building its base model again.

    `_new` stays on the class afterwards and builds it as usual, because deleting
    an assigned `__new__` leaves CPython passing the class's arguments on to
    `object.__new__`, which rejects them."""
    cls = type(base_model)
    if cls.__new__ is object.__new__:
        cls.__new__ = _new  # type: ignore[assignment]
    if cls.__new__ is _new:
        _stubbed_base_models[cls] = base_model
    try:
        yield
    finally:
        _stubbed_base_models.pop(cls, None)


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

    model: PreTrainedModel | None
    """The `ForCausalLM` with its base model stubbed out, or `None` if it applies
    nothing but its output embeddings, `scale` and `soft_cap`, which `lm_head` and
    `LogitsProcessor` compute on their own."""
    in_features: int
    """The input size of the output embeddings."""
    output_embeddings: _OutputEmbeddings = field(default_factory=_OutputEmbeddings)
    """Stands in for the output embeddings in `model`."""
    weights_mapper: WeightsMapper = field(default_factory=WeightsMapper)
    """Maps the checkpoint names of the head's weights and modules to vLLM's."""
    scale: float = 1.0
    soft_cap: float | None = None


def _module_renames(renames: dict[str, str]) -> dict[str, str]:
    """The largest modules whose weights all move together under `renames`, mapped
    to where they move (e.g. `decoder` -> `lm_head` for Ettin's `decoder.weight` and
    `decoder.bias`). Unlike weight names, these also rename names below them, such as
    LoRA's `decoder.lora_A.weight` and quantization configs' module names."""
    targets: dict[str, set[str]] = {}
    for name, new_name in renames.items():
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            suffix = "".join(f".{part}" for part in parts[i:])
            # "" if the weight's name below this module changes, so it can't move whole
            target = new_name.removesuffix(suffix) if new_name.endswith(suffix) else ""
            targets.setdefault(".".join(parts[:i]), set()).add(target)
    moved = {k: t.pop() for k, t in targets.items() if len(t) == 1 and "" not in t}
    return {
        k: v for k, v in moved.items() if k.rpartition(".")[0] not in moved and k != v
    }


def to_output_head(model: PreTrainedModel) -> OutputHead | None:
    """Turn `model`, a `ForCausalLM` built with `stub_base_model`, into its output
    head in place.

    The output embeddings are replaced with `_OutputEmbeddings`, so
    `model(inputs_embeds=h)` applies everything the `ForCausalLM` applies after its
    base model (e.g. a dense -> act -> norm transform, a soft cap, a scale).

    Returns `None` (leaving `model` unchanged) if this is not possible.
    """
    embeddings = model.get_output_embeddings()
    vocab_size = model.config.get_text_config().vocab_size
    if not isinstance(embeddings, nn.Linear) or embeddings.out_features != vocab_size:
        return None
    if not any(isinstance(m, _OpaqueBaseModel) for m in model.modules()):
        return None
    output_embeddings = _OutputEmbeddings()
    params = {id(embeddings.weight): "lm_head.weight"}
    if embeddings.bias is not None:
        params[id(embeddings.bias)] = "lm_head.bias"
    # Tying is off, so the sources of ties to the output embeddings (e.g. Roberta's
    # `lm_head.bias`) are separate parameters, and checkpoints only hold those
    embeddings_name = next(n for n, m in model.named_modules() if m is embeddings)
    tied = getattr(model, "_tied_weights_keys", None)
    for target, source in tied.items() if isinstance(tied, dict) else ():
        module_name, _, param_name = target.rpartition(".")
        if module_name == embeddings_name:
            with suppress(AttributeError):
                params[id(model.get_parameter(source))] = f"lm_head.{param_name}"
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
        renames[name] = maybe_prefix("output_head", name)
    # Into `output_head` first, as Ettin's `lm_head` moves there and `decoder` to it
    module_renames = sorted(
        _module_renames(renames).items(),
        key=lambda kv: not kv[1].startswith("output_head."),
    )
    prefixes = {re.compile(rf"^{re.escape(k)}(?=\.|$)"): v for k, v in module_renames}
    return OutputHead(
        model=model,
        output_embeddings=output_embeddings,
        in_features=embeddings.in_features,
        weights_mapper=WeightsMapper(orig_to_new_regex=prefixes),
    )


def probe_output_head(
    output_head: OutputHead, hidden_size: int
) -> tuple[float, float | None] | None:
    """The `scale` and `soft_cap` of `LogitsProcessor` which `output_head` applies
    after its output embeddings, which must still be the identity, or `None` if it
    applies anything else."""
    assert (model := output_head.model) is not None
    # Heads with parameters apply more than a scale or a soft cap
    if next(model.parameters(), None) is not None:
        return None
    # On the CPU in float64 whatever vLLM's defaults are, so that bf16 rounding can't
    # hide a scale and meta tensors can't be read
    kwargs = dict(dtype=torch.float64, device="cpu")
    hidden_states = torch.linspace(-50, 50, 3 * hidden_size, **kwargs)
    hidden_states = hidden_states.view(1, 3, hidden_size)
    hidden_states[0, 0, :2] = torch.tensor([1.0, 1e6], **kwargs)
    try:
        logits = model(inputs_embeds=hidden_states).logits.double()
    except Exception:
        return None
    if logits.shape != hidden_states.shape:
        return None
    # A scale maps 1 to itself and a soft cap maps large logits to itself
    scale, soft_cap = logits[0, 0, :2].tolist()
    if torch.allclose(logits, scale * hidden_states):
        return scale, None
    if torch.allclose(logits, soft_cap * torch.tanh(hidden_states / soft_cap)):
        return 1.0, soft_cap
    return None


class CausalMixin(VllmModelForTextGeneration, Base):
    embedding_modules = {"lm_head": "output_embeddings"}
    # A kept head runs on one row per sampled token, not per token as LoRA expects
    lora_unsupported_prefixes = ["output_head."]
    lm_head: nn.Module
    logits_processor: LogitsProcessor

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForTextGeneration.__init__ and call the next class in MRO
        super(VllmModelForTextGeneration, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        spec = self._head_spec
        self.output_head: nn.Module | None = None
        if spec.model is None:
            # Batch-sharded sampling checks for this, and kept heads need the full vocab
            self.compute_logits_local = lambda hidden_states: self.logits_processor(
                self.lm_head, hidden_states, self.lm_head.bias, skip_gather=True
            )
        if not self.pp_group.is_last_rank:
            self.lm_head = PPMissingLayer()
            if spec.model is not None:
                self.output_head = PPMissingLayer()
            return

        vocab_size = self.text_config.vocab_size
        self.logits_processor = LogitsProcessor(
            vocab_size, scale=spec.scale, soft_cap=spec.soft_cap
        )
        self.lm_head = ParallelLMHead(
            vocab_size,
            spec.in_features,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self._get_tie_word_embeddings():
            for module in self.model.get_input_embeddings().modules():
                if isinstance(module, VocabParallelEmbedding):
                    self.lm_head = self.lm_head.tie_weights(module)
                    break
        if (model := spec.model) is not None:
            # The head runs after the projection, so every rank needs all logits
            self.logits_processor.use_all_gather = True
            spec.output_embeddings.project = self._project
            self.recursive_replace(model, prefix="output_head")
            self.init_parameters(model)
            self.output_head = model

    def _create_hf_to_vllm_mapper(self):
        super()._create_hf_to_vllm_mapper()
        # Created here so the quantization config also gets the head's renames
        self._head_spec = self._create_output_head()
        # Before the base model's mappings, which expect the base model's names
        mapper = self._head_spec.weights_mapper
        self.hf_to_vllm_mapper = mapper | self.hf_to_vllm_mapper

    def _create_output_head(self) -> OutputHead:
        """The output head of the model's Transformers class, or `lm_head` with the
        config's logit scaling if it has none or it cannot be run."""
        hidden_size = self.text_config.hidden_size
        fallback = OutputHead(
            model=None,
            in_features=hidden_size,
            scale=getattr(self.text_config, "logit_scale", None) or 1.0,
            soft_cap=getattr(self.text_config, "final_logit_softcapping", None),
        )
        for architecture in self.config.architectures or []:
            model_cls = get_transformers_model_class(architecture, self.model_config)
            if model_cls is not None:
                break
        else:
            return fallback
        # Already built as `self.model` (e.g. Llama4, whose `AutoModel` is its
        # `ForConditionalGeneration`), so there is no head after it to run
        if model_cls is type(self.model):
            return fallback
        # Tying would look for the input embeddings in the base model, a stub
        config = copy.deepcopy(self.config)
        config.tie_word_embeddings = False
        config.get_text_config().tie_word_embeddings = False
        try:
            with init_on_device_without_buffers("meta"), stub_base_model(self.model):
                model = model_cls._from_config(config, dtype=self.model_config.dtype)
            output_head = to_output_head(model.eval())
            if output_head is None:
                return fallback
            # Fails for heads which need more of the base model than its output
            hidden_states = torch.empty(1, 3, hidden_size, device="meta")
            model(inputs_embeds=hidden_states)
        except Exception as e:
            logger.warning(
                "Could not run the output head of %s, so `lm_head` and the "
                "config's logit scaling are assumed: %r",
                model_cls.__name__,
                e,
            )
            return fallback
        if logit_transform := probe_output_head(output_head, hidden_size):
            output_head.scale, output_head.soft_cap = logit_transform
            output_head.model = None
        return output_head

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The output embeddings, which give the logits on every TP rank."""
        flat = hidden_states.flatten(0, -2)
        logits = self.logits_processor(self.lm_head, flat, self.lm_head.bias)
        return logits.unflatten(0, hidden_states.shape[:-1])

    def load_weights(self, weights: Iterable[tuple[str, "torch.Tensor"]]) -> set[str]:
        """A thin wrapper around `Base.load_weights` to handle the lm_head bias."""
        lm_head_bias: set[str] = set()
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
                    lm_head_bias.add("lm_head.bias")
                else:
                    yield name, weight

        return super().load_weights(auto_load_lm_head_bias(weights)) | lm_head_bias

    def compute_logits(self, hidden_states: "torch.Tensor") -> "torch.Tensor | None":
        if self.output_head is None:
            return self.logits_processor(self.lm_head, hidden_states, self.lm_head.bias)
        output = self.output_head(inputs_embeds=hidden_states.unsqueeze(0))
        return output.logits.squeeze(0)
