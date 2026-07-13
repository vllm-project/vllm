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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

__all__ = [
    "VllmJitKernel",
    "WarmupIntRange",
    "get_ast_full_name",
    "get_function_source_node",
    "zip_inputs",
]


CompileKeyT = TypeVar("CompileKeyT")


@dataclass(frozen=True)
class WarmupIntRange:
    start: int
    stop: int
    step: int = 1


WarmupValues = Any
CompileKeyDispatchFn = Callable[..., CompileKeyT]


@dataclass(frozen=True)
class _WarmupInputRows:
    """Warmup dispatch inputs expanded in lockstep."""

    rows: tuple[Mapping[str, WarmupValues], ...]


def _expand_warmup_values(values: WarmupValues) -> tuple[Any, ...]:
    if isinstance(values, WarmupIntRange):
        return tuple(range(values.start, values.stop, values.step))
    if isinstance(values, (list, tuple)):
        return tuple(values)
    return (values,)


def zip_inputs(*rows: Mapping[str, WarmupValues]) -> _WarmupInputRows:
    """Group row-wise dispatch inputs that should be expanded in lockstep."""
    if not rows:
        raise ValueError("zip_inputs requires at least one dispatch input row")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("zip_inputs rows must be mappings")

    first_names = frozenset(rows[0])
    if not first_names:
        raise ValueError("zip_inputs rows require at least one dispatch input name")
    if not all(isinstance(name, str) for name in first_names):
        raise ValueError("zip_inputs dispatch input names must be strings")

    input_rows: list[Mapping[str, WarmupValues]] = []
    for row in rows:
        names = frozenset(row)
        if names != first_names:
            raise ValueError("zip_inputs rows must use the same dispatch input names")
        input_rows.append(dict(row))

    return _WarmupInputRows(rows=tuple(input_rows))


def _expand_warmup_value_grid(
    values: Mapping[str, WarmupValues],
    input_names: frozenset[str],
) -> tuple[dict[str, Any], ...]:
    names = tuple(name for name in values if name in input_names)
    if not names:
        return ({},)

    expanded_values = tuple(_expand_warmup_values(values[name]) for name in names)
    return tuple(
        dict(zip(names, value_set)) for value_set in itertools.product(*expanded_values)
    )


def _expand_warmup_input_rows(
    rows: tuple[Mapping[str, WarmupValues], ...],
    input_names: frozenset[str],
) -> tuple[dict[str, Any], ...]:
    active_names = frozenset(name for name in rows[0] if name in input_names)
    if not active_names:
        return ({},)

    return tuple(
        {name: value for name, value in row.items() if name in active_names}
        for row in rows
    )


def _merge_warmup_kwargs(parts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        for name, value in part.items():
            if name in merged:
                raise ValueError(
                    f"Warmup dispatch input '{name}' is specified more than once"
                )
            merged[name] = value
    return merged


@dataclass(frozen=True)
class _CompileKeyDispatchTrace:
    local_exprs: tuple[tuple[str, ast.AST], ...]
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
        for name, expr in self.local_exprs:
            dispatch_values[name] = _eval_dispatch_expr(
                expr, dispatch_values, self.globals
            )
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


def _dispatch_expr_source(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node)


def _dispatch_expr_error(node: ast.AST, reason: str) -> ValueError:
    return ValueError(
        f"{reason}: {_dispatch_expr_source(node)}. "
        "Supported dispatch expressions are names, constants, attributes, "
        "tuple/list literals, conditional expressions, comparisons, boolean "
        "operators, unary not/minus, arithmetic, and calls without **kwargs."
    )


class _DispatchExprEvaluator(ast.NodeVisitor):
    def __init__(
        self,
        values: Mapping[str, Any],
        globals_: Mapping[str, Any],
    ) -> None:
        self.values = values
        self.globals = globals_

    def eval(self, node: ast.AST) -> Any:
        return self.visit(node)

    def generic_visit(self, node: ast.AST) -> Any:
        raise _dispatch_expr_error(node, "Unsupported dispatch expression")

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.values:
            return self.values[node.id]
        if node.id in self.globals:
            return self.globals[node.id]
        raise _dispatch_expr_error(node, f"Unknown dispatch name '{node.id}'")

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        return self.visit(node.body if self.visit(node.test) else node.orelse)

    def visit_Tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_List(self, node: ast.List) -> list[Any]:
        return [self.visit(elt) for elt in node.elts]

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            result = None
            for value in node.values:
                result = self.visit(value)
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = None
            for value in node.values:
                result = self.visit(value)
                if result:
                    return result
            return result
        raise _dispatch_expr_error(node, "Unsupported dispatch boolean operator")

    def visit_Compare(self, node: ast.Compare) -> bool:
        left = self.visit(node.left)
        for op_node, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            op = _CMP_OPS.get(type(op_node))
            if op is None:
                raise _dispatch_expr_error(
                    node, "Unsupported dispatch comparison operator"
                )
            if not op(left, right):
                return False
            left = right
        return True

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise _dispatch_expr_error(node, "Unsupported dispatch unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise _dispatch_expr_error(node, "Unsupported dispatch binary operator")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_Call(self, node: ast.Call) -> Any:
        args = [self.visit(arg) for arg in node.args]
        fn = self.visit(node.func)
        call_kwargs: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise _dispatch_expr_error(
                    node, "Dispatch helper calls cannot use **kwargs"
                )
            call_kwargs[keyword.arg] = self.visit(keyword.value)
        return fn(*args, **call_kwargs)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        return getattr(self.visit(node.value), node.attr)


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
    return _DispatchExprEvaluator(kwargs, globals_).eval(node)


def _collect_input_names(
    node: ast.AST,
    candidate_names: set[str],
    local_names: set[str] | None = None,
) -> set[str]:
    if local_names is None:
        local_names = set()
    return {
        child.id
        for child in ast.walk(node)
        if (
            isinstance(child, ast.Name)
            and child.id in candidate_names
            and child.id not in local_names
        )
    }


def _collect_dispatch_body(
    fn: CompileKeyDispatchFn[Any],
    function_def: ast.FunctionDef,
) -> tuple[list[tuple[str, ast.AST]], ast.Call]:
    local_exprs: list[tuple[str, ast.AST]] = []
    for statement in function_def.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue

        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(
                statement.targets[0], ast.Name
            ):
                raise _dispatch_expr_error(
                    statement, "Dispatch assignments must target one local name"
                )
            local_exprs.append((statement.targets[0].id, statement.value))
            continue

        if isinstance(statement, ast.AnnAssign):
            if statement.value is None:
                raise _dispatch_expr_error(
                    statement, "Dispatch annotations must assign a value"
                )
            if not isinstance(statement.target, ast.Name):
                raise _dispatch_expr_error(
                    statement, "Dispatch assignments must target one local name"
                )
            local_exprs.append((statement.target.id, statement.value))
            continue

        if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call):
            return local_exprs, statement.value

        if isinstance(statement, ast.Return):
            raise _dispatch_expr_error(
                statement, "Dispatch must return one CompileKey(...) call"
            )

        raise _dispatch_expr_error(
            statement,
            "Dispatch may only contain local assignments before CompileKey return",
        )

    raise ValueError(f"Expected {fn.__name__} to return one CompileKey(...) call")


def _trace_compile_key_dispatch(
    fn: CompileKeyDispatchFn[Any],
) -> _CompileKeyDispatchTrace:
    source_fn = getattr(fn, "__func__", fn)
    globals_ = source_fn.__globals__
    function_def = get_function_source_node(fn)

    local_exprs, return_call = _collect_dispatch_body(fn, function_def)

    field_exprs: list[tuple[str, ast.AST]] = []
    signature = inspect.signature(fn)
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    candidate_names = set(signature.parameters)
    input_names: set[str] = set()
    local_names = {name for name, _ in local_exprs}
    for _, expr in local_exprs:
        input_names.update(_collect_input_names(expr, candidate_names))
    for keyword in return_call.keywords:
        if keyword.arg is None:
            raise ValueError(f"{fn.__name__} cannot use **kwargs in CompileKey")
        field_exprs.append((keyword.arg, keyword.value))
        input_names.update(
            _collect_input_names(keyword.value, candidate_names, local_names)
        )

    return _CompileKeyDispatchTrace(
        tuple(local_exprs),
        tuple(field_exprs),
        globals_,
        frozenset(input_names),
        defaults,
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

        def traced(
            *input_groups: _WarmupInputRows,
            **kwargs: WarmupValues,
        ) -> list[CompileKeyT]:
            for group in input_groups:
                if not isinstance(group, _WarmupInputRows):
                    raise TypeError(
                        "_trace_dispatch positional arguments must be "
                        "zip_inputs(...) groups"
                    )
            expanded_input_groups = tuple(
                _expand_warmup_input_rows(
                    group.rows, compile_key_dispatch_trace.input_names
                )
                for group in input_groups
            )
            expanded_kwargs = _expand_warmup_value_grid(
                kwargs, compile_key_dispatch_trace.input_names
            )
            dispatch_value_groups = (*expanded_input_groups, expanded_kwargs)
            return list(
                dict.fromkeys(
                    compile_key_dispatch_trace.compile_key(
                        self.CompileKey, _merge_warmup_kwargs(dispatch_values)
                    )
                    for dispatch_values in itertools.product(*dispatch_value_groups)
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

    def warmup(self, *args: Any, **kwargs: Any) -> None:
        """Compile this kernel's warmup keys."""
        for compile_key in self.get_warmup_keys(*args, **kwargs):
            self.compile(compile_key)
