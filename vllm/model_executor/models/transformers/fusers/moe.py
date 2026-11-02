# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE fuser: run an HF MoE block's gate, experts and shared experts in `FusedMoE`."""

import ast
import functools
import inspect
import textwrap
import types
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import fx, nn
from typing_extensions import TypeIs

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.transformers.fx_utils import (
    find_node,
    is_complete,
    is_op,
    output_value,
    peel,
    trace,
)
from vllm.model_executor.models.transformers.utils import named_state
from vllm.model_executor.models.utils import maybe_prefix, sequence_parallel_chunk

logger = init_logger(__name__)


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


def _forced_dtype(nodes: Iterable[fx.Node]) -> torch.dtype | None:
    """The floating dtype `nodes` cast to, if any.

    Computations that must run in higher precision say so in their forward (e.g.
    `hidden_states.type(torch.float32)`), so the dtype is readable from the graph
    even when the config does not name it."""
    for node in nodes:
        if node.op not in ("call_method", "call_function"):
            continue
        name = str(node.target).rsplit(".", 1)[-1]
        if name == "float":
            return torch.float32
        if name in ("to", "type"):
            for arg in (*node.args[1:], *node.kwargs.values()):
                if isinstance(arg, torch.dtype) and arg.is_floating_point:
                    return arg
    return None


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


def _renormalizes(topk: fx.Node) -> bool:
    """Are the top-k weights divided by their own sum?

    Every HF router spells renormalization as a division by a sum over the
    selection, optionally offset by an epsilon."""
    downstream = _reaches(topk, "users")
    for node in downstream:
        if not (is_op(node, "div") or is_op(node, "truediv")) or len(node.args) < 2:
            continue
        denominator = node.args[1]
        if not isinstance(denominator, fx.Node):
            continue
        cone = _reaches(denominator, "all_input_nodes")
        if any(is_op(n, "sum") and n in downstream for n in cone):
            return True
    return False


def _unwrap(node: object, moe_block: nn.Module) -> object:
    """Strip reshapes, dtype casts and `nn.Identity` calls."""
    while True:
        node = peel(node)
        if (
            is_op(node, "reshape")
            or is_op(node, "view")
            or (
                isinstance(node, fx.Node)
                and node.op == "call_module"
                and isinstance(moe_block.get_submodule(node.target), nn.Identity)
            )
        ):
            node = node.args[0]
        else:
            return node


def _reads_input(node: object, x: fx.Node, moe_block: nn.Module) -> TypeIs[fx.Node]:
    """Is `node` a child module of `moe_block` called on just `x`?"""
    return (
        isinstance(node, fx.Node)
        and node.op == "call_module"
        and len(node.args) == 1
        and not node.kwargs
        and _unwrap(node.args[0], moe_block) is x
    )


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
    scoring_func: str | None
    """`None` if routing stays in the Transformers gate."""
    shared_name: str | None
    shared_gate_name: str | None
    router_dtype: torch.dtype | None = None
    renormalize: bool | None = None

    @staticmethod
    def _match_router(
        gate: nn.Module,
    ) -> tuple[str, torch.dtype | None, bool | None] | None:
        """Matches `topk(score(linear(x)))` or `score(topk(linear(x)))`, `score` being
        `softmax`/`sigmoid`.

        Returns the scoring function, the dtype the router computes in, and
        whether the top-k weights are renormalized (`None` if the trace was cut
        short before it could say)."""
        state = {name for name, _ in named_state(gate)}
        if "weight" not in state or state - {"weight", "e_score_correction_bias"}:
            return None
        graph = trace(gate)
        if graph is None:
            return None
        # The routing top-k is the last one; any earlier one scores expert groups.
        topks = [node for node in graph.nodes if is_op(node, "topk")]
        if not topks:
            return None
        topk = topks[-1]
        is_scorer = lambda n: is_op(n, "softmax") or is_op(n, "sigmoid")
        post_topk = False
        # Find all scoring functions that reach the top-k node.
        scorers = [n for n in _reaches(topk, "all_input_nodes") if is_scorer(n)]
        if not scorers:
            # Find all scoring functions that are users of the top-k node.
            scorers = [n for n in _reaches(topk, "users") if is_scorer(n)]
            post_topk = True
        if len(scorers) != 1:
            return None
        scorer = scorers[0]
        logits_cone = _reaches(scorer, "all_input_nodes")
        if not any(is_op(n, "linear") for n in logits_cone):
            return None
        is_softmax = is_op(scorer, "softmax")
        # vLLM always scores every expert, so matching a router that softmaxes
        # the selection alone takes renormalizing. Otherwise look for the
        # division, reading its absence as "no" only from a complete trace.
        renormalize: bool | None
        if (post_topk and is_softmax) or _renormalizes(topk):
            renormalize = True
        else:
            renormalize = False if is_complete(graph) else None
        scoring_func = "softmax" if is_softmax else "sigmoid"
        return scoring_func, _forced_dtype(logits_cone), renormalize

    @staticmethod
    def _match_experts(
        graph: fx.Graph, experts_name: str, moe_block: nn.Module
    ) -> tuple[fx.Node, fx.Node, fx.Node] | None:
        """Matches `experts(x, ids, weights)` with `_, weights, ids = gate(x)`, `x`
        being the block's input.

        Returns the experts call, `x` and the gate call."""
        experts = find_node(
            graph, lambda n: n.op == "call_module" and n.target == experts_name
        )
        if experts is None:
            return None
        # Bind by signature so keyword calls match too. Fail closed on any other
        # argument: the rewritten forward would drop it.
        forward = inspect.signature(moe_block.get_submodule(experts_name).forward)
        try:
            bound = forward.bind(*experts.args, **experts.kwargs)
        except TypeError:
            return None
        if len(bound.arguments) != 3:
            return None
        x, ids, weights = (_unwrap(a, moe_block) for a in bound.arguments.values())
        if not (isinstance(x, fx.Node) and x.op == "placeholder"):
            return None
        if not (is_op(ids, "getitem") and is_op(weights, "getitem")):
            return None
        gate = ids.args[0]
        if not (
            _reads_input(gate, x, moe_block)
            and ids.args == (gate, 2)
            and weights.args == (gate, 1)
        ):
            return None
        return experts, x, gate

    @staticmethod
    def _match_shared_experts(
        node: object, x: fx.Node, moe_block: nn.Module
    ) -> tuple[str, str | None] | None:
        """Matches `shared(x)` or `sigmoid(shared_gate(x)) * shared(x)`.

        Returns the shared expert's name and its optional gate's name."""
        if _reads_input(node, x, moe_block):
            return node.target, None
        if not (is_op(node, "mul") and len(node.args) == 2):
            return None
        for gated, shared in (node.args, node.args[::-1]):
            gated, shared = peel(gated), peel(shared)
            if not (is_op(gated, "sigmoid") and _reads_input(shared, x, moe_block)):
                continue
            gate = peel(gated.args[0])
            if _reads_input(gate, x, moe_block) and _is_scalar_gate(
                moe_block.get_submodule(gate.target)
            ):
                return shared.target, gate.target
        return None

    @classmethod
    def match(
        cls,
        moe_block: nn.Module,
        experts_name: str,
        divides_shared_output: bool = False,
    ) -> "MoEBlockFuser | None":
        """Matches a block returning `experts(x, ids, weights) [* c] [+ shared(x)]`
        with `_, weights, ids = gate(x)`, `x` being the block's input, up to
        reshapes, casts and `nn.Identity`.

        Args:
            moe_block: The MoE block.
            experts_name: The name of the experts in `moe_block`.
            divides_shared_output: Whether vLLM routing would divide the shared
                expert's output by `routed_scaling_factor` (fp16 overflow
                protection), which Transformers decoder layers do not undo.

        Returns:
            A fuser if more than the experts can run in vLLM, else `None`.

        """
        # Standard MoE block returns a single tensor.
        if _returns_tuple(type(moe_block)):
            return None
        if (graph := trace(moe_block)) is None:
            return None
        if (matched := cls._match_experts(graph, experts_name, moe_block)) is None:
            return None
        experts, x, gate = matched
        unwrap = functools.partial(_unwrap, moe_block=moe_block)
        gate_name = gate.target
        router = cls._match_router(getattr(moe_block, gate_name))
        scoring_func, router_dtype, renormalize = router or (None, None, None)

        def is_routed(n: object) -> bool:
            return n is experts or (
                is_op(n, "mul")
                and not isinstance(n.args[1], fx.Node)
                and unwrap(n.args[0]) is experts
            )

        # Output: the experts' output, plus the shared expert's if any.
        routed = out = unwrap(output_value(graph))
        shared = None
        if not is_routed(out):
            if not (is_op(out, "add") and len(out.args) == 2):
                return None
            routed, shared = map(unwrap, out.args)
            if is_routed(shared):
                routed, shared = shared, routed
            if not is_routed(routed):
                return None
        shared_name = shared_gate_name = None
        if shared is not None:
            if (match := cls._match_shared_experts(shared, x, moe_block)) is None:
                return None
            shared_name, shared_gate_name = match
        # Fail closed: `rewrite_forward` runs only the experts and the detected
        # shared expert, so any other stateful child would be dropped.
        accounted = {experts_name, gate_name, shared_name, shared_gate_name}
        for name, child in moe_block.named_children():
            if name not in accounted and next(named_state(child), None) is not None:
                return None
        if scoring_func is not None and shared_name and divides_shared_output:
            logger.warning_once(
                "%s could be fused but routing it in vLLM would apply "
                "`routed_scaling_factor` by dividing the shared expert output, "
                "which only fp16 overflow protection expects the decoder layer to "
                "compensate for. Falling back to routing in Transformers; run in "
                "bfloat16 to fuse it.",
                type(moe_block).__name__,
            )
            scoring_func = router_dtype = renormalize = None
        # Without vLLM routing, only a shared expert can move to vLLM, and only
        # if the block does not scale the experts' output (vLLM routing does).
        if scoring_func is None and (shared_name is None or routed is not experts):
            return None
        return cls(
            gate_name,
            scoring_func,
            shared_name,
            shared_gate_name,
            router_dtype,
            renormalize,
        )

    def gate(
        self, moe_block: nn.Module, prefix: str, out_dtype: torch.dtype | None = None
    ) -> GateLinear:
        """Rebuild the HF gate as a `GateLinear` for vLLM's fused MoE."""
        hf_gate = getattr(moe_block, self.gate_name)
        num_experts, hidden_size = hf_gate.weight.shape
        gate = GateLinear(
            hidden_size,
            num_experts,
            bias=False,
            out_dtype=out_dtype or self.router_dtype,
            prefix=maybe_prefix(prefix, self.gate_name),
        )
        if (bias := getattr(hf_gate, "e_score_correction_bias", None)) is not None:
            gate.register_buffer("e_score_correction_bias", bias)
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
