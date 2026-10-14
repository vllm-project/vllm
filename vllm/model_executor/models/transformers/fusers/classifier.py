# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Classifier fuser: extract the head of a `For*Classification` model.

The head is everything between the pooled hidden states and the classifier, e.g.
ModernBERT's `head` (dense, activation, norm). vLLM's pooler replaces whatever
pools the base model's output, so only the head after it is kept.
"""

import copy
import inspect
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import fx, nn
from transformers.modeling_outputs import BaseModelOutput

from vllm.model_executor.models.transformers.fusers.base import BaseFuser
from vllm.model_executor.models.transformers.fx_utils import (
    _AllLeafTracer,
    find_node,
    forward_parameters,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class _ClassifierTracer(_AllLeafTracer):
    """Tracer that stubs the base model with `[1, seq_len, hidden_size]` output."""

    def _infer_meta(self, kind: str, target: object, args: tuple, kwargs: dict):
        root = self.root
        assert isinstance(root, nn.Module)
        if kind == "placeholder" and str(target) in ("input_ids", "attention_mask"):
            return torch.ones(1, 8, dtype=torch.long, device="meta")
        if kind == "call_module" and str(target) == root.base_model_prefix:
            param = next(root.parameters())
            hidden_size = root.config.get_text_config().hidden_size
            hidden_states = torch.empty(
                1, 8, hidden_size, dtype=param.dtype, device="meta"
            )
            return BaseModelOutput(last_hidden_state=hidden_states)
        return super()._infer_meta(kind, target, args, kwargs)

    def create_proxy(self, kind, target, args, kwargs, *extra, **extra_kwargs):
        proxy = super().create_proxy(kind, target, args, kwargs, *extra, **extra_kwargs)
        # Keep the meta value on the node, where matching can read it
        proxy.node.meta["val"] = getattr(proxy, "meta", None)
        return proxy


def _module_call(self: nn.Module, *args, **kwargs):
    return nn.Module.__call__(self, *args, **kwargs)


def trace_classification_model(model: nn.Module) -> fx.Graph | None:
    """Trace `model.forward` up to wherever tracing stops (usually the loss)."""
    cls = type(model)
    tracer = _ClassifierTracer()
    tracer.varkw = next(
        (
            p.name
            for p in forward_parameters(cls).values()
            if p.kind is inspect.Parameter.VAR_KEYWORD
        ),
        None,
    )
    # fx traces `type(model).forward`, which Transformers wraps in decorators
    unwrapped = type(cls.__name__, (cls,), {"forward": inspect.unwrap(cls.forward)})
    model.__class__ = unwrapped
    # `@support_torch_compile` overrides `__call__`, which would hide the base model
    # call from fx, so restore the (fx patched) `nn.Module.__call__`
    base = model.get_submodule(model.base_model_prefix)
    base_cls = type(base)
    base.__class__ = type(base_cls.__name__, (base_cls,), {"__call__": _module_call})
    # `forward` sets `config.problem_type` because the traced `labels` isn't `None`
    config = model.config
    model.config = copy.deepcopy(config)
    try:
        return tracer.trace(model)
    except Exception:
        return getattr(tracer, "graph", None)
    finally:
        model.__class__ = cls
        base.__class__ = base_cls
        model.config = config


def _descendants(node: fx.Node) -> set[fx.Node]:
    seen: set[fx.Node] = set()
    stack = [node]
    while stack:
        for user in stack.pop().users:
            if user not in seen:
                seen.add(user)
                stack.append(user)
    return seen


def _ancestors(node: fx.Node) -> set[fx.Node]:
    seen: set[fx.Node] = set()
    stack = [node]
    while stack:
        for arg in stack.pop().all_input_nodes:
            if arg not in seen:
                seen.add(arg)
                stack.append(arg)
    return seen


def _hidden_and_classifier(
    graph: fx.Graph, module: nn.Module
) -> tuple[fx.Node, fx.Node] | None:
    """The base model's last hidden state and the classifier call, if traced."""
    base = find_node(
        graph,
        lambda n: n.op == "call_module" and n.target == module.base_model_prefix,
    )
    classifier = find_node(
        graph,
        lambda n: n.op == "call_module" and n.target in ("classifier", "score"),
    )
    if base is None or classifier is None:
        return None
    hidden = find_node(
        graph,
        lambda n: len(n.args) == 2
        and n.args[0] is base
        and (
            (n.target is operator.getitem and n.args[1] == 0)
            or (n.target is getattr and n.args[1] == "last_hidden_state")
        ),
    )
    return None if hidden is None else (hidden, classifier)


@dataclass
class ClassifierFuser(BaseFuser):
    """Fuser for the head between the pooled hidden states and the classifier."""

    head: fx.Graph
    """The head, from the pooled hidden states to the classifier's output."""

    module_names: list[str]
    """Names of the model's children called by `head`."""

    redefines_forward = False

    def info(self, name: str) -> str:
        return f"Fused: {' -> '.join(self.module_names)} ({name}) -> head"

    @classmethod
    def match(
        cls, graph: fx.Graph, module: nn.Module, *, pooled: bool = True
    ) -> "ClassifierFuser | None":
        """Match a head that only transforms hidden states that vLLM can pool.

        If `pooled`, the model must pool in plain tensor ops (e.g. ModernBERT's
        `[:, 0]` or masked mean), which vLLM's pooler replaces. A module that pools
        (DeBERTa's `pooler`, RoBERTa's `classifier`), a base model output other
        than the last hidden state (BERT's `pooler_output`), or a trace that stops
        before the classifier (DistilBERT's inline `nn.ReLU()`) are not matched.
        """
        nodes = _hidden_and_classifier(graph, module)
        if nodes is None:
            return None
        hidden, classifier = nodes

        # Everything downstream of `hidden` that the classifier's output depends on
        region = _descendants(hidden) & (_ancestors(classifier) | {classifier})
        order = [n for n in graph.nodes if n in region]

        def head_from(start: fx.Node) -> list[fx.Node] | None:
            """The nodes after `start`, if they depend on nothing else."""
            head = [n for n in order if n in _descendants(start)]
            inputs = {arg for node in head for arg in node.all_input_nodes}
            return head if inputs <= {start, *head} else None

        if not pooled:
            start = hidden
            head = head_from(start)
        else:
            # The head starts at the first pooled node that the rest depends on alone
            candidates = (
                n
                for n in order
                if n is not classifier
                and isinstance(val := n.meta.get("val"), torch.Tensor)
                and val.ndim == 2
            )
            start, head = next(
                ((n, h) for n in candidates if (h := head_from(n)) is not None),
                (hidden, None),
            )
            # vLLM's pooler can only replace pooling done in plain tensor ops
            pooling = _ancestors(start) & region | {start}
            if any(n.op == "call_module" for n in pooling):
                return None
        if head is None:
            return None

        graph_head = fx.Graph()
        env = {start: graph_head.placeholder("hidden_states")}
        for node in head:
            env[node] = graph_head.node_copy(node, env.__getitem__)
        graph_head.output(env[classifier])
        module_names = [str(n.target) for n in head if n.op == "call_module"]
        return cls(head=graph_head, module_names=module_names)

    @staticmethod
    def classifies_hidden_states(graph: fx.Graph, module: nn.Module) -> bool:
        """Whether the classifier reads the last hidden state, ignoring dropout.

        If so, applying the classifier alone to vLLM's pooled hidden states is
        equivalent to the model's own head (e.g. RoBERTa, or decoders' `score`).
        """
        nodes = _hidden_and_classifier(graph, module)
        if nodes is None:
            return False
        hidden, classifier = nodes
        node = classifier.args[0]
        while (
            isinstance(node, fx.Node)
            and node.op == "call_module"
            and isinstance(module.get_submodule(str(node.target)), nn.Dropout)
        ):
            node = node.args[0]
        return node is hidden

    @classmethod
    def match_model(
        cls, module: nn.Module, *, pooled: bool
    ) -> "ClassifierFuser | None":
        """Trace a `For*Classification` model and match its head."""
        graph = trace_classification_model(module)
        return None if graph is None else cls.match(graph, module, pooled=pooled)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        return True

    def fuse(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> nn.Module:
        """Run the head in the model dtype and only the classifier in `head_dtype`."""
        model_config = vllm_config.model_config
        graph = self.head
        hidden_states = next(iter(graph.nodes))
        classifier = next(n for n in reversed(graph.nodes) if n.op == "output").args[0]
        with graph.inserting_after(hidden_states):
            to_model = graph.call_method("to", (hidden_states, model_config.dtype))
        hidden_states.replace_all_uses_with(
            to_model, delete_user_cb=lambda user: user is not to_model
        )
        with graph.inserting_before(classifier):
            to_head = graph.call_method(
                "to", (classifier.args[0], model_config.head_dtype)
            )
        classifier.replace_input_with(classifier.args[0], to_head)
        return fx.GraphModule(module, graph, class_name="ClassificationHead")
