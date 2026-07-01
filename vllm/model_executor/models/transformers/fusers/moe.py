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
from vllm.model_executor.models.transformers.fusers.glu import GLUFuser
from vllm.model_executor.models.transformers.fx_utils import (
    find_node,
    is_op,
    peel,
    trace,
)
from vllm.model_executor.models.utils import (
    ShardId,
    maybe_prefix,
    sequence_parallel_chunk,
)


def named_state(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """`module`'s own state -- its named parameters and buffers together."""
    return chain(module.named_parameters(), module.named_buffers())


def _returns_tuple(cls: type[nn.Module]) -> bool:
    """True if the class's forward returns a tuple (not a single tensor)."""
    try:
        source = textwrap.dedent(inspect.getsource(inspect.unwrap(cls.forward)))
        tree = ast.parse(source)
    except (OSError, SyntaxError, TypeError):
        return True
    return any(
        isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
        for node in ast.walk(tree)
    )


def _is_scalar_gate(module: nn.Module) -> bool:
    """A linear projecting to a single logit (the shared-expert sigmoid gate)."""
    weight = getattr(module, "weight", None)
    return (
        isinstance(module, nn.Linear)
        and weight is not None
        and weight.ndim == 2
        and weight.shape[0] == 1
    )


def _match_router(gate: nn.Module) -> str | None:
    """Detects a router by dataflow: linear + softmax/sigmoid + top-k."""
    if [name for name, _ in named_state(gate)] != ["weight"]:
        return None
    graph = trace(gate)
    if graph is None:
        return None
    nodes = list(graph.nodes)
    if not any(is_op(n, "linear") for n in nodes):
        return None
    if not any(is_op(n, "topk") for n in nodes):
        return None
    softmax = any(is_op(n, "softmax") for n in nodes)
    sigmoid = any(is_op(n, "sigmoid") for n in nodes)
    if softmax == sigmoid:  # need exactly one scoring function
        return None
    return "softmax" if softmax else "sigmoid"


def _reaches(node: fx.Node, key: str) -> set[fx.Node]:
    """`node` and every fx node connected to it via `key` (`users`/`inputs`)."""
    seen: set[fx.Node] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(getattr(n, key))
    return seen


def _match_shared_expert(
    graph: fx.Graph, experts: str = "experts"
) -> tuple[str | None, str | None]:
    """Detects the shared expert and its optional gate by dataflow."""
    experts_predicate = lambda n: n.op == "call_module" and n.target == experts
    if (experts_node := find_node(graph, experts_predicate)) is None:
        return None, None
    from_experts = _reaches(experts_node, "users")
    for node in graph.nodes:
        if not is_op(node, "add"):
            continue
        operands = [a for a in node.args if isinstance(a, fx.Node)]
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


class SharedExpertMLP(nn.Module):
    """Wraps an HF shared expert, applying the output gating it is paired with."""

    def __init__(self, shared_expert: nn.Module, gate: nn.Module | None = None):
        super().__init__()
        self.shared_expert = shared_expert
        self.gate = gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.shared_expert(hidden_states)
        if self.gate is not None:
            out = torch.sigmoid(self.gate(hidden_states)[0]) * out
        return out


def _moe_block_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Native MoE block forward: the internal router routes + runs shared experts.

    Under sequence-parallel MoE, tokens are scattered across TP ranks before the
    experts and all-gathered back after (matching vLLM's native MoE blocks).
    """
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
class MoEFuser:
    """Fuser for an HF MoE block, routing through vLLM's `FusedMoE` with the
    same shared expert (if any) and gate (if any) as the original block."""

    gate_name: str
    scoring_func: str
    shared_name: str | None
    shared_gate_name: str | None
    shared_glu: GLUFuser | None

    @classmethod
    def match(cls, mlp: nn.Module) -> "MoEFuser | None":
        # The native block forward returns a single tensor.
        if _returns_tuple(type(mlp)):
            return None
        # Router: the child that scores + top-k selects (traced per child).
        gate_name = scoring_func = None
        for name, child in mlp.named_children():
            if name != "experts" and (func := _match_router(child)) is not None:
                gate_name, scoring_func = name, func
                break
        if gate_name is None or scoring_func is None:
            return None
        # Shared expert: whatever the block adds to the experts' output, found by
        # dataflow so any MLP works, not just a GLU. A GLU inside it is fused by
        # the base pass when it descends into the block child; we only precompute
        # its gate/up remap (see `orig_to_new_stacked`).
        graph = trace(mlp)
        if graph is None:
            return None
        shared_name, shared_gate_name = _match_shared_expert(graph)
        if shared_gate_name is not None and not _is_scalar_gate(
            getattr(mlp, shared_gate_name)
        ):
            return None
        shared_glu = None
        if (
            shared_name is not None
            and (sgraph := trace(shared := getattr(mlp, shared_name))) is not None
        ):
            shared_glu = GLUFuser.match(sgraph, shared)
        # Fail closed: `rewrite_forward` runs only the experts and the detected
        # shared expert, so any other stateful child would be dropped.
        accounted = {"experts", gate_name, shared_name, shared_gate_name}
        for name, child in mlp.named_children():
            if name not in accounted and next(named_state(child), None) is not None:
                return None
        return cls(gate_name, scoring_func, shared_name, shared_gate_name, shared_glu)

    def build_gate(self, mlp: nn.Module, prefix: str) -> ReplicatedLinear:
        """Rebuild the HF router as a `ReplicatedLinear` producing logits.

        Shapes come from the original router weight (`[num_experts, hidden]`),
        the plain `F.linear` weight it loads by identity into. Left unquantized
        -- routers run in full precision -- matching the checkpoint's weight.
        """
        num_experts, hidden_size = getattr(mlp, self.gate_name).weight.shape
        gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            prefix=maybe_prefix(prefix, self.gate_name),
        )
        setattr(mlp, self.gate_name, gate)
        return gate

    def build_shared_experts(
        self, mlp: nn.Module, prefix: str
    ) -> SharedExpertMLP | None:
        """Wrap the HF shared expert, folding in its sigmoid output gate.

        The MLP is the original module (converted in place by the base pass); the
        sibling gate is rebuilt as a `ReplicatedLinear` (its `[1, hidden]` weight
        is used as a plain `F.linear` weight) so the wrapper holds a reference
        that survives the base pass's replacement, and it loads by identity.
        """
        if self.shared_name is None:
            return None
        shared_expert = getattr(mlp, self.shared_name)
        gate = None
        if self.shared_gate_name is not None:
            hf_gate = getattr(mlp, self.shared_gate_name)
            gate = ReplicatedLinear(
                hf_gate.in_features,
                hf_gate.out_features,
                bias=hf_gate.bias is not None,
                prefix=maybe_prefix(prefix, self.shared_gate_name),
            )
            setattr(mlp, self.shared_gate_name, gate)
        return SharedExpertMLP(shared_expert, gate)

    def orig_to_new_stacked(self, prefix: str) -> dict[str, tuple[str, ShardId]]:
        """Merge the shared expert's gate/up projections (scoped to its qualname).

        Registered here so loading is correct regardless of whether the base pass
        reaches the shared expert via the block child or the runner alias first.
        """
        if self.shared_name is None or self.shared_glu is None:
            return {}
        return self.shared_glu.orig_to_new_stacked(
            maybe_prefix(prefix, self.shared_name)
        )

    def rewrite_forward(self, mlp: nn.Module) -> None:
        """Bind the native block forward (`experts` routes + runs shared)."""
        mlp.forward = types.MethodType(_moe_block_forward, mlp)
