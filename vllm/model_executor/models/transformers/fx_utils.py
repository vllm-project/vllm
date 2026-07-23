# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fx tracing and forward-source rewriting for the Transformers backend fusers.

A small engine, independent of any particular pattern: trace a module's forward
with `torch.fx` (tolerating a partial graph), inspect the resulting nodes, and
rewrite the forward's *source* (AST) so only matched calls change while the rest
stays live Python. `fusion.py` builds the concrete fusion patterns on top.
"""

import ast
import contextlib
import inspect
import operator
import textwrap
from collections.abc import Callable
from itertools import chain
from unittest import mock

import torch
from torch import fx, nn
from torch.nn import functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

_LEAF_CALL_LENGTHS: dict[Callable, int] = {}
"""Callables traced as leaf calls (see `_as_leaf_call`), mapped to the number of
values each returns."""

_UNKNOWN = object()
"""Sentinel meta value for proxies whose concrete value could not be inferred.
Distinct from `None`, which is a valid concrete value (e.g. `attn_weights`)."""

_MODULE_CALL = nn.Module.__call__
"""The unpatched `nn.Module.__call__`. During tracing fx patches it to record
`call_module` nodes; meta execution must call modules for real."""


def is_leaf_call(node: object) -> bool:
    """Is node a call recorded by `_as_leaf_call` (e.g. an attention interface)."""
    return (
        isinstance(node, fx.Node)
        and node.op == "call_function"
        and node.target in _LEAF_CALL_LENGTHS
    )


def _reference_weight(module: nn.Module) -> torch.Tensor | None:
    """A weight whose trailing dim is the module's hidden size.

    Linears and 2-D gate weights are `[out, hidden]`; norm weights are
    `[hidden]`. Used to fabricate a placeholder input of matching size/dtype."""
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child.weight
    for param in module.parameters():
        if param.ndim in (1, 2):
            return param
    return None


class _MetaProxy(fx.Proxy):
    """Proxy carrying the meta-tensor value of the traced expression.

    Shape questions (`len`, iteration, `.shape` unpacks) are answered by
    executing each op on the meta values, so PyTorch's meta kernels are the
    single source of shape inference — no per-op rules."""

    meta: object = _UNKNOWN

    def __len__(self) -> int:
        if self.meta is not _UNKNOWN:
            return len(self.meta)
        return super().__len__()  # type: ignore[misc]

    def __getattr__(self, k: str) -> "_MetaAttribute":
        return _MetaAttribute(self, k)


class _MetaAttribute(_MetaProxy, fx.proxy.Attribute):
    """Attribute proxy (e.g. `x.shape`) carrying its meta value.

    `Proxy.__getattr__` constructs `Attribute` directly, bypassing
    `Tracer.proxy`, so the meta value must be grafted on here too."""

    def __init__(self, root: fx.Proxy, attr: str):
        super().__init__(root, attr)
        root_meta = getattr(root, "meta", _UNKNOWN)
        if root_meta is not _UNKNOWN:
            with contextlib.suppress(Exception):
                self.meta = getattr(root_meta, attr)


class _AllLeafTracer(fx.Tracer):
    """Tracer that treats every submodule as a leaf.

    Each child stays one `call_module` node, so matching sees the module's own
    forward structure (activations aren't decomposed into e.g. `sigmoid * x`).
    Every traced op is also executed on meta tensors (see `_MetaProxy`) so
    shape unpacks and `*`-splats trace through; anything else untraceable ends
    the trace early and the partial graph is matched.
    """

    varkw: str | None = None
    """Name of the traced forward's `**kwargs` parameter, if any."""

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        return True

    def proxy(self, node: fx.Node) -> fx.Proxy:
        return _MetaProxy(node, self)

    def create_proxy(self, kind, target, args, kwargs, *extra, **extra_kwargs):
        proxy = super().create_proxy(kind, target, args, kwargs, *extra, **extra_kwargs)
        if isinstance(proxy, _MetaProxy) and proxy.meta is _UNKNOWN:
            # A failure stays _UNKNOWN; only fatal if a shape question is asked.
            with contextlib.suppress(Exception):
                proxy.meta = self._infer_meta(kind, target, args, kwargs)
        return proxy

    def _infer_meta(self, kind: str, target: object, args: tuple, kwargs: dict):
        """Execute the op on meta tensors; PyTorch infers the output value."""
        if kind == "placeholder":
            # vLLM always feeds the model [1, seq_len, hidden_size] hidden states.
            weight = _reference_weight(self.root)
            if str(target) == "hidden_states" and weight is not None:
                return torch.empty(
                    1, 8, weight.shape[-1], dtype=weight.dtype, device="meta"
                )
            return _UNKNOWN
        # Leaf calls don't execute; fabricate a value of the declared length.
        if kind == "call_function" and target in _LEAF_CALL_LENGTHS:
            return (_UNKNOWN,) * _LEAF_CALL_LENGTHS[target]
        if kind == "get_attr":
            value = operator.attrgetter(str(target))(self.root)
            if isinstance(value, torch.Tensor):
                value = torch.empty_like(value, device="meta")
            return value
        unknown = False

        def meta_of(arg: object) -> object:
            nonlocal unknown
            if isinstance(arg, fx.Proxy):
                meta = getattr(arg, "meta", _UNKNOWN)
                unknown = unknown or meta is _UNKNOWN
                return meta
            return arg

        meta_args = fx.node.map_aggregate(args, meta_of)
        meta_kwargs = fx.node.map_aggregate(kwargs, meta_of)
        if unknown:
            return _UNKNOWN
        if kind == "call_function":
            return target(*meta_args, **meta_kwargs)
        if kind == "call_method":
            receiver, *rest = meta_args
            return getattr(receiver, str(target))(*rest, **meta_kwargs)
        if kind == "call_module":
            # Run the child's forward with all its state on "meta", without
            # mutating it (at match time params may be meta but buffers real).
            # fx patches `nn.Module.__call__` while tracing; restore the real
            # one so this execution is not itself recorded.
            child = self.root.get_submodule(str(target))
            state = {
                name: torch.empty_like(tensor, device="meta")
                for name, tensor in chain(
                    child.named_parameters(), child.named_buffers()
                )
            }
            with mock.patch.object(nn.Module, "__call__", _MODULE_CALL):
                return torch.func.functional_call(child, state, meta_args, meta_kwargs)
        return _UNKNOWN

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
        meta = getattr(obj, "meta", _UNKNOWN)
        if meta is _UNKNOWN:
            return super().iter(obj)
        return iter([obj[i] for i in range(len(meta))])


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


def upstream_linear(node: object, module: nn.Module) -> fx.Node | None:
    """Nearest linear producing `node`, walking back through splits/reshapes.

    Never walks through a leaf call (e.g. an attention interface): its inputs
    are what attention consumes, not what produced the value."""
    stack = [node]
    seen: set[fx.Node] = set()
    while stack:
        current = stack.pop()
        if not isinstance(current, fx.Node) or current in seen:
            continue
        seen.add(current)
        if is_linear(current, module):
            return current
        if current.op in ("call_function", "call_method") and not is_leaf_call(current):
            stack.extend(current.args)
    return None


def downstream_linear(node: fx.Node, module: nn.Module) -> fx.Node | None:
    """Nearest linear consuming `node`'s output, walking through casts/scalings.

    Never walks through a leaf call (e.g. an attention interface): what crosses
    it is consumed by the attention computation, not projected."""
    queue = list(node.users)
    seen: set[fx.Node] = set()
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if is_linear(current, module):
            return current
        if current.op in ("call_function", "call_method") and not is_leaf_call(current):
            queue.extend(current.users)
    return None


def returned_linear(graph: fx.Graph, module: nn.Module) -> str | None:
    """Name of the Linear producing the graph's (first) output value."""
    value = output_value(graph)
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    linear = upstream_linear(value, module)
    return None if linear is None else str(linear.target)


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
