# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared interfaces and tracing helpers for explicit JIT warmup keys."""

from __future__ import annotations

import ast
import inspect
import itertools
import operator
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

__all__ = [
    "VllmJitKernel",
    "WarmupIntRange",
    "get_ast_full_name",
    "get_function_source_node",
]


CompileKeyT = TypeVar("CompileKeyT")


@dataclass(frozen=True)
class WarmupIntRange:
    start: int
    stop: int
    step: int = 1


WarmupValues = Any
CompileKeyDispatchFn = Callable[..., CompileKeyT]


def _expand_warmup_values(values: WarmupValues) -> tuple[Any, ...]:
    if isinstance(values, WarmupIntRange):
        return tuple(range(values.start, values.stop, values.step))
    if isinstance(values, (list, tuple)):
        return tuple(values)
    return (values,)


@dataclass(frozen=True)
class _CompileKeyDispatchTrace:
    field_exprs: tuple[tuple[str, ast.AST], ...]
    globals: Mapping[str, Any]
    input_names: frozenset[str]
    defaults: Mapping[str, Any]

    def compile_key(
        self,
        compile_key_type: type[CompileKeyT],
        kwargs: Mapping[str, Any],
    ) -> CompileKeyT:
        dispatch_values = {**self.defaults, **kwargs}
        return compile_key_type(
            **{
                field: _eval_dispatch_expr(expr, dispatch_values, self.globals)
                for field, expr in self.field_exprs
            }
        )


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP_OPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def get_ast_full_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = get_ast_full_name(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def get_function_source_node(fn: Callable[..., Any]) -> ast.FunctionDef:
    source_fn = getattr(fn, "fn", fn)
    source = textwrap.dedent(inspect.getsource(source_fn))
    tree = ast.parse(source)
    function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(function_defs) != 1:
        name = getattr(source_fn, "__name__", type(source_fn).__name__)
        raise ValueError(f"Expected one function in {name}, found {len(function_defs)}")
    return function_defs[0]


def _eval_dispatch_expr(
    node: ast.AST,
    kwargs: Mapping[str, Any],
    globals_: Mapping[str, Any],
) -> Any:
    if isinstance(node, ast.Name):
        if node.id in kwargs:
            return kwargs[node.id]
        if node.id in globals_:
            return globals_[node.id]
        raise ValueError(f"Unknown dispatch name: {node.id}")

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.IfExp):
        branch = node.body if _eval_dispatch_expr(
            node.test, kwargs, globals_
        ) else node.orelse
        return _eval_dispatch_expr(branch, kwargs, globals_)

    if isinstance(node, ast.Tuple):
        return tuple(_eval_dispatch_expr(elt, kwargs, globals_) for elt in node.elts)

    if isinstance(node, ast.List):
        return [_eval_dispatch_expr(elt, kwargs, globals_) for elt in node.elts]

    if isinstance(node, ast.BoolOp):
        values = (
            _eval_dispatch_expr(value, kwargs, globals_) for value in node.values
        )
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise ValueError(f"Unsupported dispatch boolean op: {ast.dump(node.op)}")

    if isinstance(node, ast.Compare):
        left = _eval_dispatch_expr(node.left, kwargs, globals_)
        for op_node, comparator in zip(node.ops, node.comparators):
            right = _eval_dispatch_expr(comparator, kwargs, globals_)
            op = _CMP_OPS.get(type(op_node))
            if op is None:
                raise ValueError(
                    f"Unsupported dispatch comparison op: {ast.dump(op_node)}"
                )
            if not op(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_dispatch_expr(node.operand, kwargs, globals_)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_dispatch_expr(node.operand, kwargs, globals_)

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported dispatch binary op: {ast.dump(node.op)}")
        return op(
            _eval_dispatch_expr(node.left, kwargs, globals_),
            _eval_dispatch_expr(node.right, kwargs, globals_),
        )

    if isinstance(node, ast.Call):
        args = [_eval_dispatch_expr(arg, kwargs, globals_) for arg in node.args]
        fn = _eval_dispatch_expr(node.func, kwargs, globals_)
        call_kwargs = {
            keyword.arg: _eval_dispatch_expr(keyword.value, kwargs, globals_)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        return fn(*args, **call_kwargs)

    if isinstance(node, ast.Attribute):
        value = _eval_dispatch_expr(node.value, kwargs, globals_)
        return getattr(value, node.attr)

    raise ValueError(f"Unsupported dispatch expression: {ast.dump(node)}")


def _collect_input_names(
    node: ast.AST,
    candidate_names: set[str],
) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and child.id in candidate_names
    }


def _trace_compile_key_dispatch(fn: CompileKeyDispatchFn[Any]) -> _CompileKeyDispatchTrace:
    source_fn = getattr(fn, "__func__", fn)
    globals_ = source_fn.__globals__
    function_def = get_function_source_node(fn)

    returns = [
        node.value
        for node in ast.walk(function_def)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    if len(returns) != 1 or not isinstance(returns[0], ast.Call):
        raise ValueError(
            f"Expected {fn.__name__} to return exactly one CompileKey(...) call"
        )

    field_exprs: list[tuple[str, ast.AST]] = []
    signature = inspect.signature(fn)
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    candidate_names = set(signature.parameters)
    input_names: set[str] = set()
    for keyword in returns[0].keywords:
        if keyword.arg is None:
            raise ValueError(f"{fn.__name__} cannot use **kwargs in CompileKey")
        field_exprs.append((keyword.arg, keyword.value))
        input_names.update(_collect_input_names(keyword.value, candidate_names))

    return _CompileKeyDispatchTrace(
        tuple(field_exprs), globals_, frozenset(input_names), defaults
    )


class VllmJitKernel(Generic[CompileKeyT], ABC):
    """Kernel wrapper that owns dispatch, warmup keys, and compilation."""

    CompileKey: type[CompileKeyT]

    def __init__(self) -> None:
        self.compile_key_dispatch_trace = _trace_compile_key_dispatch(self.dispatch)

    def compile_key(self, kwargs: Mapping[str, Any]) -> CompileKeyT:
        return self.compile_key_dispatch_trace.compile_key(self.CompileKey, kwargs)

    def _trace_dispatch(
        self, dispatch: CompileKeyDispatchFn[CompileKeyT]
    ) -> Callable[..., list[CompileKeyT]]:
        compile_key_dispatch_trace = _trace_compile_key_dispatch(dispatch)

        def traced(**kwargs: WarmupValues) -> list[CompileKeyT]:
            kwarg_names = tuple(
                name
                for name in kwargs
                if name in compile_key_dispatch_trace.input_names
            )
            kwarg_values = tuple(
                _expand_warmup_values(kwargs[name]) for name in kwarg_names
            )
            return list(
                dict.fromkeys(
                    compile_key_dispatch_trace.compile_key(
                        self.CompileKey, dict(zip(kwarg_names, values))
                    )
                    for values in itertools.product(*kwarg_values)
                )
            )

        return traced

    @abstractmethod
    def dispatch(self, **kwargs: Any) -> CompileKeyT:
        """Build one compile key from one concrete dispatch point."""
        raise NotImplementedError

    @abstractmethod
    def get_warmup_keys(self, *args: Any, **kwargs: Any) -> list[CompileKeyT]:
        """Return compile keys that should be warmed for this kernel."""
        raise NotImplementedError

    @abstractmethod
    def compile(self, compile_key: CompileKeyT) -> None:
        """Compile one warmup key."""
        raise NotImplementedError
