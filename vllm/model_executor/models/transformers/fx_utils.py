# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fx tracing and forward-source rewriting for the Transformers backend fusers.

A small engine, independent of any particular pattern: trace a module's forward
with `torch.fx` (tolerating a partial graph), inspect the resulting nodes, and
rewrite the forward's *source* (AST) so only matched calls change while the rest
stays live Python. `fusion.py` builds the concrete fusion patterns on top.
"""

import ast
import inspect
import operator
import textwrap
from collections.abc import Callable

import torch
from torch import fx, nn
from torch.nn import functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

_LEAF_CALL_LENGTHS: dict[Callable, int] = {}
"""Callables traced as leaf calls (see `_as_leaf_call`), mapped to the number of
values each returns."""


def _infer_len(node: fx.Node, root: nn.Module | None) -> int | None:
    """Concrete length of a proxy's value, inferred from its node chain.

    Lets tracing pass through the shape unpacks and `*`-splats (e.g.
    `(*input_shape, -1, head_dim)`) that precede the patterns in HF attention.
    """
    # `x.shape` has the rank of `x`, when known.
    if (
        is_fn(node, getattr)
        and node.args[1] == "shape"
        and (rank := _rank(node.args[0], root)) is not None
    ):
        return rank
    # Slices of known-length values.
    if is_op(node, "getitem"):
        src_len = _infer_len(node.args[0], root)
        index = node.args[1]
        if src_len is not None and isinstance(index, slice):
            return len(range(*index.indices(src_len)))
    # Nodes wrapped with `_as_leaf_call` declare their length explicitly.
    if node.op == "call_function" and node.target in _LEAF_CALL_LENGTHS:
        return _LEAF_CALL_LENGTHS[node.target]
    return None


def _is_split(node: object) -> bool:
    return is_op(node, "split") or is_op(node, "chunk")


_RANK_PRESERVING_METHODS = frozenset({"transpose", "contiguous", "clone"})


def _rank(node: object, root: nn.Module | None) -> int | None:
    """The tensor rank of `node`'s value, derived from its producing node chain."""
    node = peel(node)  # dtype casts preserve rank
    if not isinstance(node, fx.Node):
        return None
    # vLLM always feeds the model [1, seq_len, hidden_size] hidden states
    if node.op == "placeholder":
        return 3 if node.target == "hidden_states" else None
    # Linears map only the last dim and norms are elementwise: both preserve rank.
    # Other children (embeddings, rope, ...) may not, so theirs is unknown.
    if node.op == "call_module":
        try:
            child = root.get_submodule(str(node.target)) if root else None
        except AttributeError:
            child = None
        weight = getattr(child, "weight", None)
        preserves = isinstance(child, nn.Linear) or (
            weight is not None and weight.ndim == 1
        )
        return _rank(node.args[0], root) if preserves and node.args else None
    # `split`/`chunk` yield tuples whose elements keep the source tensor's rank
    if _is_split(node) or (is_op(node, "getitem") and _is_split(node.args[0])):
        return _rank(node.args[0], root)
    if node.op == "call_method":
        if node.target in ("view", "reshape", "expand"):
            sizes = node.args[1:]
            if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                return len(sizes[0])
            return len(sizes) or None
        if node.target == "unsqueeze":
            rank = _rank(node.args[0], root)
            return None if rank is None else rank + 1
        if node.target in _RANK_PRESERVING_METHODS:
            return _rank(node.args[0], root)
    if is_op(node, "cat"):
        elements = node.args[0]
        if isinstance(elements, (tuple, list)) and elements:
            return _rank(elements[0], root)
    return None


class _SizedProxy(fx.Proxy):
    """Proxy whose `len` is inferred from the graph (see `_infer_len`)."""

    def __len__(self) -> int:
        assert isinstance(self.tracer, _AllLeafTracer)
        length = self.tracer.infer_len(self.node)
        if length is None:
            return super().__len__()
        return length


class _AllLeafTracer(fx.Tracer):
    """Tracer that treats every submodule as a leaf.

    Each child stays one `call_module` node, so matching sees the module's own
    forward structure (activations aren't decomposed into e.g. `sigmoid * x`).
    `iter` traces through the leading shape unpacks (see `_infer_len`); anything
    else untraceable ends the trace early and the partial graph is matched.
    """

    varkw: str | None = None
    """Name of the traced forward's `**kwargs` parameter, if any."""

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        return True

    def proxy(self, node: fx.Node) -> fx.Proxy:
        return _SizedProxy(node, self)

    def infer_len(self, node: fx.Node) -> int | None:
        return _infer_len(node, getattr(self, "root", None))

    def _is_varkw(self, node: object) -> bool:
        return (
            isinstance(node, fx.Node)
            and node.op == "placeholder"
            and str(node.target).lstrip("*") == self.varkw
        )

    def iter(self, obj: fx.Proxy):
        # Assume kwargs is always empty to simplify tracing.
        node = obj.node
        if self._is_varkw(node) or (
            node.op == "call_method"
            and node.target == "keys"
            and self._is_varkw(node.args[0])
        ):
            return iter(())
        length = self.infer_len(node)
        if length is None:
            return super().iter(obj)
        return iter([obj[i] for i in range(length)])


def _as_leaf_call(fn: Callable, length: int | None = None) -> Callable:
    """Wrap any callable so tracing records it as one opaque `call_function` node.

    Lets the trace continue past untraceable bodies. Only the proxy arguments carry into
    the node's dataflow; the rest are dropped rather than lifted into the graph.
    `length` declares how many values the callable returns, so unpacking its result also
    traces. Called without proxies (i.e. outside tracing), the wrapper is a passthrough.
    """

    def leaf(*args, **kwargs):
        proxies = tuple(arg for arg in args if isinstance(arg, fx.Proxy))
        if not proxies:
            return fn(*args, **kwargs)
        if length is not None:
            _LEAF_CALL_LENGTHS[fn] = length
        return proxies[0].tracer.create_proxy("call_function", fn, proxies, {})

    return leaf


def _leaf_attention_interfaces():
    """Patch `AttentionInterface.get_interface` so traced forwards see a leaf node.

    `vllm_attention_function` needs runtime context so it is untraceable.
    Every interface returns `(attn_output, attn_weights)`."""
    from unittest import mock

    from transformers.modeling_utils import AttentionInterface

    original = AttentionInterface.get_interface

    def get_interface(self, *args, **kwargs):
        return _as_leaf_call(original(self, *args, **kwargs), length=2)

    return mock.patch.object(AttentionInterface, "get_interface", get_interface)


def trace(module: nn.Module) -> fx.Graph | None:
    """Trace `module.forward`, returning the partial graph on failure."""
    parameters = forward_parameters(type(module))
    # vLLM never passes `past_key_values` so it is always the default value of `None`.
    # Make this concrete to simplify tracing.
    concrete_args = None
    if "past_key_values" in parameters:
        concrete_args = {"past_key_values": None}
    # Get the name of the kwargs parameter passed to module.forward (usually "kwargs")
    tracer = _AllLeafTracer()
    tracer.varkw = next(
        (
            p.name
            for p in parameters.values()
            if p.kind is inspect.Parameter.VAR_KEYWORD
        ),
        None,
    )
    try:
        with _leaf_attention_interfaces():
            return tracer.trace(module, concrete_args=concrete_args)
    except Exception as exc:
        logger.debug("Could not fully trace %s: %s", type(module), exc)
        return getattr(tracer, "graph", None)


def recover_forward(cls: type[nn.Module]) -> tuple[ast.FunctionDef, Callable]:
    """Parse the source of `cls.forward`, ready for rewriting."""
    fn = inspect.unwrap(cls.forward)
    if fn.__code__.co_freevars:
        raise ValueError("forward is a closure")
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    funcdef = tree.body[0]
    if not isinstance(funcdef, ast.FunctionDef):
        raise ValueError("source is not a plain function definition")
    # `fn` is already unwrapped; don't re-apply its decorators
    funcdef.decorator_list.clear()
    # Annotations may not evaluate outside the defining module (e.g. with
    # postponed evaluation); they're not needed at runtime
    funcdef.returns = None
    args = funcdef.args
    for arg in (
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
        *filter(None, (args.vararg, args.kwarg)),
    ):
        arg.annotation = None
    # Recompiling outside the class body would break name mangling
    for node in ast.walk(funcdef):
        name = getattr(node, "attr", None) or getattr(node, "id", None)
        if name and name.startswith("__") and not name.endswith("__"):
            raise ValueError(f"{name} would be name mangled")
    return funcdef, fn


def forward_parameters(cls: type[nn.Module]) -> dict[str, inspect.Parameter]:
    """`cls.forward`'s signature parameters, or empty if uninspectable."""
    try:
        return dict(inspect.signature(cls.forward).parameters)
    except (TypeError, ValueError):
        return {}


def forward_input_count(cls: type[nn.Module]) -> int:
    """The number of tensor inputs `cls.forward` declares, excluding `self` and
    any `*args`/`**kwargs`. Read from the signature, so it is independent of
    whether the trace completes (unlike counting placeholders)."""
    params = list(forward_parameters(cls).values())
    if not params:
        return 1  # uninspectable: assume a single input and let matching decide
    fixed = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    return sum(1 for p in params[1:] if p.kind in fixed)


def compile_forward(funcdef: ast.FunctionDef, fn: Callable) -> Callable:
    """Compile `funcdef` in `fn`'s module so tracebacks point at the source."""
    module = ast.Module(body=[funcdef], type_ignores=[])
    ast.fix_missing_locations(module)
    ast.increment_lineno(module, fn.__code__.co_firstlineno - 1)
    code = compile(module, fn.__code__.co_filename, "exec")
    namespace: dict = {}
    exec(code, fn.__globals__, namespace)
    return namespace[funcdef.name]


def single_self_call(funcdef: ast.FunctionDef, name: str) -> ast.Call:
    """The unique `self.<name>(arg)` call in `funcdef`.

    Raises unless `name` appears exactly once, as such a call, so the source
    rewrite agrees with the fx match.
    """
    uses = [
        node
        for node in ast.walk(funcdef)
        if isinstance(node, ast.Attribute) and node.attr == name
    ]
    if len(uses) != 1:
        raise ValueError(f"{name} is referenced {len(uses)} times")
    calls = [
        node
        for node in ast.walk(funcdef)
        if isinstance(node, ast.Call)
        and node.func is uses[0]
        and len(node.args) == 1
        and not isinstance(node.args[0], ast.Starred)
        and not node.keywords
    ]
    if (
        len(calls) != 1
        or not isinstance(uses[0].value, ast.Name)
        or uses[0].value.id != "self"
    ):
        raise ValueError(f"{name} is not a single-argument call on self")
    return calls[0]


def innermost_block(
    block: list[ast.stmt], node: ast.AST
) -> tuple[list[ast.stmt], int] | None:
    """The innermost statement list containing `node`, and the index within."""
    for index, stmt in enumerate(block):
        if not any(child is node for child in ast.walk(stmt)):
            continue
        child_blocks = [
            getattr(stmt, fld, None) for fld in ("body", "orelse", "finalbody")
        ]
        child_blocks += [h.body for h in getattr(stmt, "handlers", [])]
        child_blocks += [c.body for c in getattr(stmt, "cases", [])]
        for child_block in child_blocks:
            if (
                isinstance(child_block, list)
                and child_block
                and (found := innermost_block(child_block, node)) is not None
            ):
                return found
        return block, index
    return None


def replace_expr(module: ast.AST, old: ast.expr, new: ast.expr) -> None:
    """Replace the expression `old` (by identity) with `new` within `module`."""

    class _Replacer(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            if node is old:
                return new
            return super().generic_visit(node)

    _Replacer().visit(module)


def find_node(graph: fx.Graph, predicate: Callable[[fx.Node], bool]) -> fx.Node | None:
    """The first node in `graph` matching `predicate`, or `None`."""
    return next((n for n in graph.nodes if predicate(n)), None)


def output_value(graph: fx.Graph) -> object | None:
    """The value the graph's `output` node returns, if the trace reached one."""
    output = find_node(graph, lambda n: n.op == "output")
    if output is None or not output.args:
        return None
    return output.args[0]


def is_linear(node: fx.Node, module: nn.Module) -> bool:
    """Is node `nn.Linear.__call__()`."""
    return node.op == "call_module" and isinstance(
        module.get_submodule(node.target), nn.Linear
    )


_DTYPE_CASTS = frozenset({"to", "float", "double", "half", "bfloat16", "type_as"})


def peel(node: object) -> object:
    """Strip dtype-cast wrappers (`.to(...)`, `.float()`, `.type_as(...)`)."""
    while (
        isinstance(node, fx.Node)
        and node.op == "call_method"
        and node.target in _DTYPE_CASTS
    ):
        node = node.args[0]
    return node


def is_fn(node: object, target: Callable) -> bool:
    """Is node `<target>()`."""
    return (
        isinstance(node, fx.Node)
        and node.op == "call_function"
        and node.target is target
    )


def is_method(node: object, name: str) -> bool:
    """Is node `.<name>()`."""
    return (
        isinstance(node, fx.Node) and node.op == "call_method" and node.target == name
    )


def is_op(node: object, name: str) -> bool:
    """
    Is node `torch.<name>()`, `F.<name>()`, `operator.<name>()`, or `Tensor.<name>()`.
    """
    return any(
        is_fn(node, getattr(module, name, None)) for module in (torch, F, operator)
    ) or (hasattr(torch.Tensor, name) and is_method(node, name))
