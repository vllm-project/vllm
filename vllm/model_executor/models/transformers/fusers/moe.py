# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE fuser: run an HF MoE block's gate, experts and shared experts in `FusedMoE`."""

import ast
import functools
import inspect
import math
import operator
import textwrap
import types
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

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
    is_fn,
    is_leaf_call,
    is_op,
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


def _run_moe(experts: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run `experts: MoERunner`, which also handles routing and any shared experts."""
    orig_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, orig_shape[-1])
    num_tokens = hidden_states.shape[0]
    is_sequence_parallel = experts.moe_config.is_sequence_parallel
    if is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)
    out = experts(hidden_states, router_logits=hidden_states)
    if is_sequence_parallel:
        out = tensor_model_parallel_all_gather(out, 0)[:num_tokens]
    return out.reshape(orig_shape)


def _scale(node: fx.Node) -> tuple[float, object] | None:
    """`(c, x)` if `node` is `x * c`, `c * x` or `x / c` for a constant `c`."""
    if node.kwargs or len(node.args) != 2:
        return None
    a, b = node.args
    if is_op(node, "mul") and isinstance(b, (int, float)):
        return float(b), a
    if is_op(node, "mul") and isinstance(a, (int, float)):
        return float(a), b
    if (is_op(node, "truediv") or is_op(node, "div")) and isinstance(b, (int, float)):
        return 1 / b, a
    return None


def _only_feeds(source: fx.Node, node: object) -> bool:
    """Is `source` used only to compute `node`, along the arguments `_unwrap` and
    `_scale` follow, with every value on the way (and `node`) used once?"""
    while isinstance(node, fx.Node) and len(node.users) == 1:
        if node is source:
            return True
        node = scale[1] if (scale := _scale(node)) else node.args[0]
    return False


def _erase_unused(graph: fx.Graph, modules: set[str]) -> None:
    """Erase unused nodes, keeping calls to child modules not in `modules` and
    in-place writes, which may have side effects."""
    for node in reversed(graph.nodes):
        if node.users or node.op in ("placeholder", "output"):
            continue
        if node.op == "call_module" and node.target not in modules:
            continue
        if is_fn(node, operator.setitem) or str(node.target).endswith("_"):
            continue
        graph.erase_node(node)


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
    graph: fx.Graph | None = field(default=None, repr=False, compare=False)
    """The block's forward with the routed and shared experts replaced by one call
    to the fused MoE."""

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
    ) -> tuple[fx.Node, fx.Node, fx.Node, fx.Node] | None:
        """Matches `experts(x, ids, weights)` with `_, weights, ids = gate(x)`.

        Returns the experts call, its hidden states argument, `x` (that argument
        without reshapes and casts) and the gate call."""
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
        hidden_states = next(iter(bound.arguments.values()))
        x, ids, weights = (_unwrap(a, moe_block) for a in bound.arguments.values())
        if not (isinstance(hidden_states, fx.Node) and isinstance(x, fx.Node)):
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
        return experts, hidden_states, x, gate

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
        routed_scaling_factor: float = 1.0,
    ) -> "MoEBlockFuser | None":
        """Matches a block computing `experts(x, ids, weights) [* c] [+ shared(x)]`
        with `_, weights, ids = gate(x)`, up to reshapes, casts and `nn.Identity`.

        Anything else in the block's forward, e.g. a norm before or after the
        experts, is kept around the fused MoE.

        Args:
            moe_block: The MoE block.
            experts_name: The name of the experts in `moe_block`.
            routed_scaling_factor: The scale vLLM routing applies to the experts'
                output, which `c` must equal.

        Returns:
            A fuser if more than the experts can run in vLLM, else `None`.

        """
        # Standard MoE block returns a single tensor.
        if _returns_tuple(type(moe_block)):
            return None
        if (graph := trace(moe_block)) is None:
            return None
        if not is_complete(graph):
            return None
        # Only a graph that runs as traced can replace the forward, so it must not
        # depend on optional inputs, which tracing fixes.
        placeholders = [n for n in graph.nodes if n.op == "placeholder"]
        if any(n.users for n in placeholders[1:]):
            return None
        for node in graph.nodes:
            constant = node.op == "get_attr" and str(node.target).startswith("_tensor")
            if is_leaf_call(node) or constant:
                return None
        source = graph.python_code("self").src
        if (matched := cls._match_experts(graph, experts_name, moe_block)) is None:
            return None
        experts, hidden_states, x, gate = matched
        unwrap = functools.partial(_unwrap, moe_block=moe_block)
        gate_name = gate.target
        router = cls._match_router(getattr(moe_block, gate_name))
        scoring_func, router_dtype, renormalize = router or (None, None, None)

        def scales_experts(n: fx.Node) -> bool:
            return (scale := _scale(n)) is not None and unwrap(scale[1]) is experts

        scaled = find_node(graph, scales_experts)
        scaled_arg = None
        if scaled is not None:
            factor, scaled_arg = _scale(scaled) or (None, None)
            # vLLM routing applies `routed_scaling_factor`, not `c`
            if factor is None or not math.isclose(factor, routed_scaling_factor):
                return None
        routed = scaled or experts
        # The experts' output plus the shared expert's, if any.
        combine = find_node(
            graph,
            lambda n: (
                is_op(n, "add")
                and len(n.args) == 2
                and not n.kwargs
                and any(unwrap(arg) is routed for arg in n.args)
            ),
        )
        shared_name = shared_gate_name = None
        if combine is not None:
            routed_idx = next(
                i for i, a in enumerate(combine.args) if unwrap(a) is routed
            )
            shared = unwrap(combine.args[1 - routed_idx])
            if (match := cls._match_shared_experts(shared, x, moe_block)) is None:
                combine = None
            else:
                shared_name, shared_gate_name = match
        # Without vLLM routing, only a shared expert can move to vLLM, and only
        # if the block does not scale the experts' output (vLLM routing does).
        if scoring_func is None and (shared_name is None or scaled is not None):
            return None
        # The fused MoE returns the experts' output scaled and combined, so the
        # partial values must not be used anywhere else.
        if scaled is not None and not _only_feeds(experts, scaled_arg):
            return None
        if combine is not None:
            routed_arg = combine.args[routed_idx]
            if not _only_feeds(experts, routed_arg):
                return None

        # Splice: replace the experts call with the fused MoE and bypass the
        # scale and the shared expert, which it applies.
        with graph.inserting_before(experts):
            runner = graph.get_attr(experts_name)
            fused = graph.call_function(_run_moe, (runner, hidden_states))
        experts.replace_all_uses_with(fused)
        if scaled is not None:
            # Read the operand again: it may be the replaced experts call
            _, operand = _scale(scaled) or (None, None)
            assert isinstance(operand, fx.Node)
            scaled.replace_all_uses_with(operand)
        if combine is not None:
            combine.replace_all_uses_with(combine.args[routed_idx])
        moved = {gate_name, shared_name, shared_gate_name} - {None}
        _erase_unused(graph, moved | {experts_name})
        # Fail closed if the experts, router or shared expert are used elsewhere.
        for node in graph.nodes:
            refs_moved = str(node.target).split(".")[0] in moved | {experts_name}
            is_ref = node.op in ("call_module", "get_attr") and node is not runner
            if is_ref and refs_moved:
                return None
        # Fail closed if the trace specialized on the input's sequence length.
        # This only catches specialization that changes the generated code.
        retraced = trace(moe_block, seq_len=5)
        if retraced is None or retraced.python_code("self").src != source:
            return None
        return cls(
            gate_name,
            scoring_func,
            shared_name,
            shared_gate_name,
            router_dtype,
            renormalize,
            graph,
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
        assert self.graph is not None
        code = self.graph.python_code("self")
        namespace = dict(code.globals)
        filename = f"<{type(moe_block).__name__}.forward>"
        exec(compile(code.src, filename, "exec"), namespace)
        moe_block.forward = types.MethodType(namespace["forward"], moe_block)
