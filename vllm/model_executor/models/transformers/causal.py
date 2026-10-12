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

import contextlib
import inspect
import numbers
import operator
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import torch
from torch import fx, nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from typing_extensions import TypeIs

from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.transformers.fx_utils import (
    _DTYPE_CASTS,
    find_node,
    is_fn,
    is_op,
    trace,
)
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.transformers_utils.dynamic_module import get_transformers_model_class

from .base import Base

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


TRANSFORM_ATTR = "lm_head_transform"
"""Attribute in which to store the output head's transform, if it has one."""

_CONCRETE_ARGS = {
    "inputs_embeds": None,
    "labels": None,
    "return_dict": True,
    "use_cache": False,
    "output_attentions": False,
    "output_hidden_states": False,
    "output_router_logits": False,
}
"""`ForCausalLM.forward` arguments vLLM never sets, so control flow on them traces."""

_PASSTHROUGH = _DTYPE_CASTS | {"contiguous"}
"""Methods which do not change the values of the logits."""


@dataclass
class OutputHead:
    embeddings: nn.Linear
    """The output embeddings, which `ParallelLMHead` replaces."""
    transform: fx.GraphModule | None = None
    """Applied to the hidden states before the output embeddings, if needed."""
    scale: float = 1.0
    soft_cap: float | None = None
    renames: dict[str, str] = field(default_factory=dict)
    """Checkpoint name -> vLLM name, for the output head's weights."""


@contextlib.contextmanager
def _unwrapped_forward(module: nn.Module):
    """Temporarily strip the decorators from `module.forward`, which fx cannot trace."""
    cls = type(module)
    forward = inspect.unwrap(cls.forward)
    module.__class__ = type(cls.__name__, (cls,), {"forward": forward})
    try:
        yield
    finally:
        module.__class__ = cls


def _trace(module: nn.Module) -> fx.Graph | None:
    with _unwrapped_forward(module):
        return trace(module, concrete_args=_CONCRETE_ARGS, partial=False)


def _inline(model: nn.Module, graph: fx.Graph, node: fx.Node) -> bool:
    """Replace the `call_module` `node` with the traced body of its module."""
    if (body := _trace(model.get_submodule(node.target))) is None:
        return False
    placeholders = [n for n in body.nodes if n.op == "placeholder"]
    env: dict[fx.Node, object] = {}
    for i, placeholder in enumerate(placeholders):
        if i < len(node.args):
            env[placeholder] = node.args[i]
        elif placeholder.name in node.kwargs:
            env[placeholder] = node.kwargs[placeholder.name]
        elif placeholder.users and not placeholder.args:
            return False
        else:
            env[placeholder] = placeholder.args[0] if placeholder.args else None
    with graph.inserting_before(node):
        for n in body.nodes:
            if n.op == "placeholder":
                continue
            if n.op == "output":
                result = fx.node.map_arg(n.args[0], env.__getitem__)
                break
            copy = graph.node_copy(n, env.__getitem__)
            if copy.op in ("call_module", "get_attr"):
                copy.target = f"{node.target}.{copy.target}"
            env[n] = copy
    if not isinstance(result, fx.Node):
        return False
    node.replace_all_uses_with(result)
    graph.erase_node(node)
    return True


def _is_passthrough(node: object) -> TypeIs[fx.Node]:
    """Whether `node` is a cast, or the selection of the positions to compute logits
    for (vLLM passes `compute_logits` only those positions already)."""
    if not isinstance(node, fx.Node):
        return False
    if node.op == "call_method" and node.target in _PASSTHROUGH:
        return True
    if not is_fn(node, operator.getitem):
        return False
    index = node.args[1] if isinstance(node.args[1], tuple) else (node.args[1],)
    return any(
        isinstance(i, fx.Node)
        and i.op == "placeholder"
        and i.target == "logits_to_keep"
        for i in index
    )


def _skip_passthrough(node: object) -> object:
    while _is_passthrough(node):
        node = node.args[0]
    return node


def _is_hidden_states(model: nn.Module, node: fx.Node) -> bool:
    """Whether `node` is (an item of) the output of the base model, which is the
    module that holds the input embeddings."""
    while is_fn(node, getattr) or is_fn(node, operator.getitem):
        node = node.args[0]
    if node.op != "call_module":
        return False
    input_embeddings = model.get_input_embeddings()
    return any(
        m is input_embeddings for m in model.get_submodule(node.target).modules()
    )


def _transform(
    model: PreTrainedModel, graph: fx.Graph, embeddings_call: fx.Node
) -> fx.GraphModule | None:
    """Everything between the base model and the output embeddings, without casts."""
    end = _skip_passthrough(embeddings_call.args[0])
    if not isinstance(end, fx.Node):
        raise LookupError("the output embeddings are not called on a tensor")
    region: set[fx.Node] = set()
    starts: set[fx.Node] = set()
    stack = [end]
    while stack:
        node = stack.pop()
        if node in region or node in starts:
            continue
        if _is_hidden_states(model, node):
            starts.add(node)
        elif node.op == "placeholder":
            raise LookupError(f"the output head reads the input {node.name!r}")
        else:
            region.add(node)
            stack.extend(_skip_passthrough(n) for n in node.all_input_nodes)
    if len(starts) != 1:
        raise LookupError("the output head does not start from the hidden states")
    if not region:
        return None

    transform = fx.Graph()
    env: dict[fx.Node, fx.Node] = {starts.pop(): transform.placeholder("hidden_states")}
    for node in graph.nodes:
        if node in region:
            env[node] = transform.node_copy(node, lambda n: env[_skip_passthrough(n)])
    transform.output(env[end])
    return fx.GraphModule(model, transform, class_name="OutputHeadTransform")


def _constant(node: fx.Node, value: object) -> float | None:
    """The other operand of the binary op `node`, if it is a number."""
    args = list(node.args)
    if len(args) != 2 or value not in args:
        return None
    args.remove(value)
    other = args[0]
    is_number = isinstance(other, numbers.Number) and not isinstance(other, bool)
    return float(other) if is_number else None


def _is_div(node: fx.Node) -> bool:
    return is_op(node, "truediv") or is_op(node, "div")


def _post_ops(embeddings_call: fx.Node) -> tuple[float, float | None]:
    """The scale and soft cap applied to the logits, as `LogitsProcessor` does."""
    scale, soft_cap = 1.0, None
    node = embeddings_call
    while len(node.users) == 1:
        user = next(iter(node.users))
        if _is_passthrough(user):
            node = user
        elif (
            soft_cap is None
            and scale == 1.0
            and _is_div(user)
            and user.args[0] is node
            and (cap := _constant(user, node)) is not None
            and len(user.users) == 1
            and is_op(tanh := next(iter(user.users)), "tanh")
            and len(tanh.users) == 1
            and is_op(mul := next(iter(tanh.users)), "mul")
            and _constant(mul, tanh) == cap
        ):
            soft_cap, node = cap, mul
        elif is_op(user, "mul") and (factor := _constant(user, node)) is not None:
            scale, node = scale * factor, user
        elif (
            _is_div(user)
            and user.args[0] is node
            and (divisor := _constant(user, node)) is not None
        ):
            scale, node = scale / divisor, user
        else:
            break

    def is_output(user: fx.Node) -> bool:
        target = user.target
        return user.op == "output" or (
            isinstance(target, type) and issubclass(target, ModelOutput)
        )

    if unsupported := [user for user in node.users if not is_output(user)]:
        ops = ", ".join(
            getattr(u.target, "__name__", str(u.target)) for u in unsupported
        )
        raise NotImplementedError(f"the logits pass through unsupported ops ({ops})")
    return scale, soft_cap


def find_output_head(model: PreTrainedModel) -> OutputHead | None:
    """Find the output head of `model`, a `ForCausalLM` instance (e.g. on meta).

    The head is whatever `model.forward` applies to the base model's hidden states to
    produce the logits, split into a transform (e.g. dense -> act -> norm), the output
    embeddings (which become `ParallelLMHead`) and a soft cap and/or scale (which
    become `LogitsProcessor` arguments).

    Returns `None` if it cannot be found. Raises if it is found but vLLM cannot
    reproduce it.
    """
    try:
        model.get_input_embeddings()
    except NotImplementedError:
        return None
    embeddings = model.get_output_embeddings()
    if not isinstance(embeddings, nn.Linear):
        return None
    names = {module: name for name, module in model.named_modules()}
    if (graph := _trace(model)) is None:
        return None

    def calls(module: nn.Module):
        return (
            lambda n: n.op == "call_module" and model.get_submodule(n.target) is module
        )

    # Inline modules which hold the output embeddings (e.g. Roberta's `lm_head`)
    while (
        container := find_node(
            graph,
            lambda n: n.op == "call_module"
            and names[embeddings].startswith(f"{n.target}."),
        )
    ) is not None:
        if not _inline(model, graph, container):
            return None
    if (embeddings_call := find_node(graph, calls(embeddings))) is None:
        return None

    try:
        transform = _transform(model, graph, embeddings_call)
    except LookupError as exc:
        logger.debug("Could not find the output head of %s: %s", type(model), exc)
        return None
    try:
        scale, soft_cap = _post_ops(embeddings_call)
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"The Transformers modeling backend cannot reproduce the output head of "
            f"{type(model).__name__}: {exc}."
        ) from exc

    # Map every name the output embeddings' parameters have outside the base model
    renames: dict[str, str] = {}
    params = {id(embeddings.weight): "lm_head.weight"}
    if embeddings.bias is not None:
        params[id(embeddings.bias)] = "lm_head.bias"
    for name, param in model.named_parameters(remove_duplicate=False):
        if id(param) in params and not name.startswith(f"{model.base_model_prefix}."):
            renames[name] = params[id(param)]
    if transform is not None:
        for name, _ in transform.named_parameters(remove_duplicate=False):
            renames[name] = f"{TRANSFORM_ATTR}.{name}"
        for name, _ in transform.named_buffers(remove_duplicate=False):
            renames[name] = f"{TRANSFORM_ATTR}.{name}"

    output_head = OutputHead(embeddings, transform, scale, soft_cap, renames)
    logger.debug("Identified output head as: %s", output_head)
    return output_head


class CausalMixin(VllmModelForTextGeneration, Base):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForTextGeneration.__init__ and call the next class in MRO
        super(VllmModelForTextGeneration, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        tie_word_embeddings = self._get_tie_word_embeddings()
        output_head = self._output_head

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                output_head.embeddings.in_features
                if output_head is not None
                else self.text_config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if tie_word_embeddings:
                for module in self.model.get_input_embeddings().modules():
                    if isinstance(module, VocabParallelEmbedding):
                        self.lm_head = self.lm_head.tie_weights(module)
                        break

            if output_head is not None:
                scale, soft_cap = output_head.scale, output_head.soft_cap
            else:
                scale = getattr(self.text_config, "logit_scale", 1.0)
                soft_cap = getattr(self.text_config, "final_logit_softcapping", None)
            self.logits_processor = LogitsProcessor(
                self.text_config.vocab_size, scale=scale, soft_cap=soft_cap
            )

            if output_head is not None and output_head.transform is not None:
                transform = output_head.transform
                self.recursive_replace(transform, prefix=TRANSFORM_ATTR)
                self.init_parameters(transform)
                setattr(self, TRANSFORM_ATTR, transform)
        else:
            self.lm_head = PPMissingLayer()
            if output_head is not None and output_head.transform is not None:
                setattr(self, TRANSFORM_ATTR, PPMissingLayer())

    @cached_property
    def _output_head(self) -> OutputHead | None:
        """The output head of the model's Transformers class, or `None` if it cannot
        be found, in which case `lm_head` and the config's logit scaling are assumed.
        """
        for architecture in self.config.architectures or []:
            model_cls = get_transformers_model_class(architecture, self.model_config)
            if model_cls is not None:
                break
        else:
            return None
        with torch.device("meta"):
            model = model_cls._from_config(self.config, dtype=self.model_config.dtype)
        return find_output_head(model)

    def load_weights(self, weights: Iterable[tuple[str, "torch.Tensor"]]) -> set[str]:
        """A thin wrapper around `Base.load_weights` to handle the output head."""
        lm_head_bias = set()
        load_device = self.vllm_config.load_config.device or self.device_config.device
        renames = self._output_head.renames if self._output_head is not None else {}

        def auto_load_lm_head_bias(weights):
            for name, weight in weights:
                name = renames.get(name, name)
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
        if (transform := getattr(self, TRANSFORM_ATTR, None)) is not None:
            hidden_states = transform(hidden_states)
        logits = self.logits_processor(self.lm_head, hidden_states, self.lm_head.bias)
        return logits
