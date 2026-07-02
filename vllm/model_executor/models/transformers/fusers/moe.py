# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE fuser: route an HF MoE block through `FusedMoE` with vLLM's own routing."""

import ast
import inspect
import textwrap
import types
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain

import torch
from torch import fx, nn

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.transformers.fx_utils import (
    find_node,
    is_op,
    peel,
    trace,
)
from vllm.model_executor.models.utils import maybe_prefix, sequence_parallel_chunk


def named_state(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """`module`'s own state (i.e. named parameters and buffers)."""
    return chain(module.named_parameters(), module.named_buffers())


def _own_returns(node: ast.AST) -> Iterator[ast.Return]:
    """`return` statements in `node`'s own scope, not in nested functions."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, ast.Return):
            yield child
        elif not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(child))


def _returns_tuple(cls: type[nn.Module]) -> bool:
    """Does `cls.forward()` return a tuple?"""
    try:
        source = textwrap.dedent(inspect.getsource(inspect.unwrap(cls.forward)))
        forward = ast.parse(source).body[0]
    except (OSError, SyntaxError, TypeError, IndexError):
        return True
    # Names bound to a tuple literal, e.g. `out = hidden, logits` then `return out`.
    tuple_names = {
        target.id
        for node in ast.walk(forward)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def yields_tuple(value: ast.expr | None) -> bool:
        if isinstance(value, ast.Tuple):
            return True
        if isinstance(value, ast.Name):
            return value.id in tuple_names
        if isinstance(value, ast.IfExp):
            return yields_tuple(value.body) or yields_tuple(value.orelse)
        return False

    return any(yields_tuple(ret.value) for ret in _own_returns(forward))


def _is_scalar_gate(module: nn.Module) -> bool:
    """A linear projecting to a single logit (the shared-expert sigmoid gate)."""
    weight = getattr(module, "weight", None)
    return (
        isinstance(module, nn.Linear)
        and weight is not None
        and weight.ndim == 2
        and weight.shape[0] == 1
    )


def _reaches(node: fx.Node, key: str) -> set[fx.Node]:
    """Returns the set of nodes reachable from `node` by following `key` edges."""
    seen: set[fx.Node] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(getattr(n, key))
    return seen


class SharedExpertMLP(nn.Module):
    """Wraps an HF shared expert, applying the output gating it is paired with."""

    def __init__(self, shared_experts: nn.Module, gate: nn.Module | None = None):
        super().__init__()
        self.shared_experts = shared_experts
        self.gate = gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.shared_experts(hidden_states)
        if self.gate is not None:
            out = torch.sigmoid(self.gate(hidden_states)[0]) * out
        return out


def _moe_block_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Standard MoE block forward.

    Routing and any shared experts are handled inside `self.experts: MoERunner`."""
    orig_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, orig_shape[-1])
    num_tokens = hidden_states.shape[0]
    is_sequence_parallel = self.experts.moe_config.is_sequence_parallel
    if is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)
    out = self.experts(hidden_states, router_logits=hidden_states)
    if is_sequence_parallel:
        out = tensor_model_parallel_all_gather(out, 0)[:num_tokens]
    return out.reshape(orig_shape)


@dataclass
class MoEBlockFuser:
    """Fuser for MoE block `experts`, `gate` and `shared_experts` (optional)."""

    gate_name: str
    scoring_func: str
    shared_name: str | None
    shared_gate_name: str | None

    @staticmethod
    def _match_router(gate: nn.Module) -> str | None:
        """Matches `topk(score(linear(x)))`, `score` being `softmax`/`sigmoid`."""
        if [name for name, _ in named_state(gate)] != ["weight"]:
            return None
        graph = trace(gate)
        if graph is None:
            return None
        topk = find_node(graph, lambda n: is_op(n, "topk"))
        if topk is None:
            return None
        # Exactly one scoring op upstream of the top-k, fed (transitively) by a linear.
        scorers = [
            n
            for n in _reaches(topk, "all_input_nodes")
            if is_op(n, "softmax") or is_op(n, "sigmoid")
        ]
        if len(scorers) != 1:
            return None
        scorer = scorers[0]
        if not any(is_op(n, "linear") for n in _reaches(scorer, "all_input_nodes")):
            return None
        return "softmax" if is_op(scorer, "softmax") else "sigmoid"

    @staticmethod
    def _match_shared_experts(
        graph: fx.Graph, experts: str
    ) -> tuple[str | None, str | None]:
        """Detects the shared expert and its optional gate by dataflow."""
        experts_predicate = lambda n: n.op == "call_module" and n.target == experts
        if (experts_node := find_node(graph, experts_predicate)) is None:
            return None, None
        from_experts = _reaches(experts_node, "users")
        for add in graph.nodes:
            if not is_op(add, "add"):
                continue
            operands = [a for a in add.args if isinstance(a, fx.Node)]
            # Exactly one side is the experts' output; the other is the shared path.
            sides = [a in from_experts for a in operands]
            if len(operands) != 2 or sides.count(True) != 1:
                continue
            cone = _reaches(operands[sides.index(False)], "all_input_nodes")
            modules = [n for n in cone if n.op == "call_module" and n.target != experts]
            # A sigmoid wrapping one of those modules marks the shared-expert gate.
            gate = next(
                (
                    src
                    for n in cone
                    if is_op(n, "sigmoid")
                    and isinstance(src := peel(n.args[0]), fx.Node)
                    and src in modules
                ),
                None,
            )
            shared = [n for n in modules if n is not gate]
            if len(shared) != 1:
                return None, None
            return shared[0].target, (gate.target if gate is not None else None)
        return None, None

    @classmethod
    def match(cls, moe_block: nn.Module, experts_name: str) -> "MoEBlockFuser | None":
        # Standard MoE block returns a single tensor.
        if _returns_tuple(type(moe_block)):
            return None
        # Router: the child that scores + top-k selects.
        gate_name = scoring_func = None
        for name, child in moe_block.named_children():
            if name != experts_name and (func := cls._match_router(child)) is not None:
                gate_name, scoring_func = name, func
                break
        if gate_name is None or scoring_func is None:
            return None
        # Shared expert: a child the block adds to the experts' output.
        shared_name = shared_gate_name = None
        others = [
            n
            for n, _ in moe_block.named_children()
            if n not in {experts_name, gate_name}
        ]
        if others:
            graph = trace(moe_block)
            if graph is None:
                return None
            shared_name, shared_gate_name = cls._match_shared_experts(
                graph, experts_name
            )
            if shared_gate_name is not None and not _is_scalar_gate(
                getattr(moe_block, shared_gate_name)
            ):
                return None
        # Fail closed: `rewrite_forward` runs only the experts and the detected
        # shared expert, so any other stateful child would be dropped.
        accounted = {experts_name, gate_name, shared_name, shared_gate_name}
        for name, child in moe_block.named_children():
            if name not in accounted and next(named_state(child), None) is not None:
                return None
        return cls(gate_name, scoring_func, shared_name, shared_gate_name)

    def gate(self, moe_block: nn.Module, prefix: str) -> ReplicatedLinear:
        """Rebuild the HF gate as a `ReplicatedLinear` for vLLM's fused MoE."""
        num_experts, hidden_size = getattr(moe_block, self.gate_name).weight.shape
        gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            prefix=maybe_prefix(prefix, self.gate_name),
        )
        setattr(moe_block, self.gate_name, gate)
        return gate

    def shared_experts(
        self, moe_block: nn.Module, prefix: str
    ) -> SharedExpertMLP | None:
        """Build the HF shared expert (and its optional gate)
        as a `SharedExpertMLP` for vLLM's fused MoE."""
        if self.shared_name is None:
            return None
        shared_experts = getattr(moe_block, self.shared_name)
        gate = None
        if self.shared_gate_name is not None:
            hf_gate = getattr(moe_block, self.shared_gate_name)
            gate = ReplicatedLinear(
                hf_gate.in_features,
                hf_gate.out_features,
                bias=hf_gate.bias is not None,
                prefix=maybe_prefix(prefix, self.shared_gate_name),
            )
            setattr(moe_block, self.shared_gate_name, gate)
        return SharedExpertMLP(shared_experts, gate)

    def rewrite_forward(self, moe_block: nn.Module) -> None:
        """Rewrite `moe_block.forward` to route through vLLM's fused MoE."""
        moe_block.forward = types.MethodType(_moe_block_forward, moe_block)
