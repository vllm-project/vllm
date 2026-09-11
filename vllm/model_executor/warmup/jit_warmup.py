# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared interfaces and tracing helpers for explicit JIT warmup keys."""

from __future__ import annotations

import ast
import builtins
import inspect
import itertools
import operator
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from functools import wraps
from typing import Any, Generic, TypeVar, cast

__all__ = [
    "JitWarmupRegistry",
    "VllmJitKernel",
    "WarmupChoices",
    "WarmupIntRange",
    "get_ast_full_name",
    "get_function_source_node",
    "zip_inputs",
]


CompileKeyT = TypeVar("CompileKeyT")


@dataclass(frozen=True)
class WarmupIntRange:
    """Expand integers with range semantics or a custom monotonic progression."""

    start: int
    stop: int
    step: int = 1
    advance: Callable[[int], int] | None = None


@dataclass(frozen=True)
class WarmupChoices:
    """Expand an explicit finite set of values in a traced warmup method."""

    values: tuple[Any, ...]

    def __init__(self, *values: Any) -> None:
        object.__setattr__(self, "values", values)


def _when(condition: bool) -> None:
    """Filter symbolic cases in an AST-traced warmup-cases method."""
    raise RuntimeError("_when() is only valid in an AST-traced warmup method")


WarmupValues = Any
CompileKeyDispatchFn = Callable[..., CompileKeyT]
WarmupPredicateFn = Callable[..., bool]
_LocalExprs = tuple[tuple[str, ast.AST], ...]


def _eval_local_exprs(
    local_exprs: _LocalExprs,
    values: Mapping[str, Any],
    globals_: Mapping[str, Any],
) -> dict[str, Any]:
    evaluated_values = dict(values)
    for name, expr in local_exprs:
        evaluated_values[name] = _eval_dispatch_expr(expr, evaluated_values, globals_)
    return evaluated_values


@dataclass(frozen=True)
class _WarmupInputRows:
    """Warmup dispatch inputs expanded in lockstep."""

    rows: tuple[Mapping[str, WarmupValues], ...]


def _expand_warmup_values(values: WarmupValues) -> tuple[Any, ...]:
    if isinstance(values, WarmupChoices):
        return values.values
    if isinstance(values, WarmupIntRange):
        if values.advance is None:
            return tuple(range(values.start, values.stop, values.step))
        if values.step != 1:
            raise ValueError("WarmupIntRange cannot set both step and advance")

        expanded: list[int] = []
        value = values.start
        while value < values.stop:
            expanded.append(value)
            next_value = values.advance(value)
            if next_value <= value:
                raise ValueError("WarmupIntRange.advance must return a greater value")
            value = next_value
        return tuple(expanded)
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
    local_exprs: _LocalExprs
    field_exprs: tuple[tuple[str, ast.AST], ...]
    globals: Mapping[str, Any]
    input_names: frozenset[str]
    # Named parameters are excluded from direct **kwargs forwarding.
    named_parameters: frozenset[str] | None
    defaults: Mapping[str, Any]

    def input_names_for(self, available_names: set[str]) -> frozenset[str]:
        if self.named_parameters is None:
            return self.input_names
        return self.input_names | (available_names - self.named_parameters)

    def compile_key(
        self,
        compile_key_type: type[CompileKeyT],
        kwargs: Mapping[str, Any],
    ) -> CompileKeyT:
        dispatch_values = _eval_local_exprs(
            self.local_exprs, {**self.defaults, **kwargs}, self.globals
        )
        named_parameters = self.named_parameters
        # Materialize direct fields before evaluating named AST expressions.
        fields: dict[str, Any] = {}
        if named_parameters is not None:
            fields = {
                name: value
                for name, value in kwargs.items()
                if name not in named_parameters
            }
        for field, expr in self.field_exprs:
            if field in fields:
                raise TypeError(f"CompileKey field '{field}' is specified twice")
            fields[field] = _eval_dispatch_expr(expr, dispatch_values, self.globals)
        return compile_key_type(**fields)


@dataclass(frozen=True)
class _WarmupPredicateTrace:
    local_exprs: _LocalExprs
    return_expr: ast.AST
    globals: Mapping[str, Any]
    input_names: frozenset[str]
    defaults: Mapping[str, Any]

    def matches(self, kwargs: Mapping[str, Any]) -> bool:
        dispatch_values = _eval_local_exprs(
            self.local_exprs, {**self.defaults, **kwargs}, self.globals
        )
        return bool(
            _eval_dispatch_expr(self.return_expr, dispatch_values, self.globals)
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
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
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
        "subscriptions, tuple/list literals, conditional expressions, "
        "comparisons, boolean operators, unary not/minus, arithmetic, and "
        "calls without **kwargs."
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
        if hasattr(builtins, node.id):
            return getattr(builtins, node.id)
        raise _dispatch_expr_error(node, f"Unknown dispatch name '{node.id}'")

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Lambda(self, node: ast.Lambda) -> Callable[..., Any]:
        arguments = node.args
        if (
            arguments.posonlyargs
            or arguments.vararg is not None
            or arguments.kwonlyargs
            or arguments.kwarg is not None
            or arguments.defaults
        ):
            raise _dispatch_expr_error(node, "Traced lambdas require positional args")
        names = tuple(argument.arg for argument in arguments.args)

        def evaluate(*values: Any) -> Any:
            if len(values) != len(names):
                raise TypeError(f"Expected {len(names)} lambda arguments")
            return type(self)(
                dict(self.values) | dict(zip(names, values)), self.globals
            ).eval(node.body)

        return evaluate

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
        args: list[Any] = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                args.extend(self.visit(arg.value))
            else:
                args.append(self.visit(arg))
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

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        return self.visit(node.value)[self.visit(node.slice)]


def get_ast_full_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = get_ast_full_name(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def get_function_source_node(fn: Callable[..., Any]) -> ast.FunctionDef | ast.Lambda:
    source_fn = getattr(fn, "fn", fn)
    source = textwrap.dedent(inspect.getsource(source_fn))
    tree = ast.parse(source)
    function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(function_defs) == 1:
        return function_defs[0]

    lambdas = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]
    if len(lambdas) == 1:
        return lambdas[0]

    name = getattr(source_fn, "__name__", type(source_fn).__name__)
    raise ValueError(
        f"Expected one function or lambda in {name}, found "
        f"{len(function_defs)} functions and {len(lambdas)} lambdas"
    )


def _eval_dispatch_expr(
    node: ast.AST,
    kwargs: Mapping[str, Any],
    globals_: Mapping[str, Any],
) -> Any:
    return _DispatchExprEvaluator(kwargs, globals_).eval(node)


def _validate_dispatch_expr(node: ast.AST) -> None:
    """Enforce the dispatch AST subset before compiling traced code."""
    allowed_nodes = (
        ast.Name,
        ast.Constant,
        ast.Lambda,
        ast.arguments,
        ast.arg,
        ast.IfExp,
        ast.Tuple,
        ast.List,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Compare,
        *tuple(_CMP_OPS),
        ast.UnaryOp,
        ast.Not,
        ast.USub,
        ast.BinOp,
        *tuple(_BIN_OPS),
        ast.Call,
        ast.keyword,
        ast.Starred,
        ast.Attribute,
        ast.Subscript,
        ast.Load,
    )
    for child in ast.walk(node):
        if not isinstance(child, allowed_nodes):
            raise _dispatch_expr_error(child, "Unsupported dispatch expression")
        if isinstance(child, ast.Call) and any(
            keyword.arg is None for keyword in child.keywords
        ):
            raise _dispatch_expr_error(
                child, "Dispatch helper calls cannot use **kwargs"
            )


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


def _named_assignment(
    statement: ast.stmt,
    subject: str,
) -> tuple[str, ast.AST] | None:
    match statement:
        case ast.Assign(targets=[ast.Name(id=name)], value=value):
            return name, value
        case ast.AnnAssign(target=ast.Name(id=name), value=value) if value is not None:
            return name, value
        case ast.Assign() | ast.AnnAssign():
            raise _dispatch_expr_error(
                statement, f"{subject} assignments require one name"
            )
        case _:
            return None


def _collect_expression_body(
    fn: Callable[..., Any],
    function_def: ast.FunctionDef | ast.Lambda,
) -> tuple[list[tuple[str, ast.AST]], ast.AST]:
    if isinstance(function_def, ast.Lambda):
        return [], function_def.body

    local_exprs: list[tuple[str, ast.AST]] = []
    for statement in function_def.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue

        if (
            assignment := _named_assignment(statement, "Traced warmup helper")
        ) is not None:
            local_exprs.append(assignment)
            continue

        if isinstance(statement, ast.Return):
            if statement.value is None:
                raise _dispatch_expr_error(
                    statement, "Traced warmup helper must return an expression"
                )
            return local_exprs, statement.value

        raise _dispatch_expr_error(
            statement,
            "Traced warmup helper may only contain local assignments before return",
        )

    raise ValueError(f"Expected {fn.__name__} to return an expression")


def _function_trace_inputs(
    fn: Callable[..., Any],
) -> tuple[dict[str, Any], set[str]]:
    signature = inspect.signature(fn)
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    bound_self = getattr(fn, "__self__", None)
    if bound_self is not None:
        defaults["self"] = bound_self
    return defaults, set(signature.parameters)


def _trace_compile_key_dispatch(
    fn: CompileKeyDispatchFn[Any],
) -> _CompileKeyDispatchTrace:
    source_fn = getattr(fn, "__func__", fn)
    globals_ = source_fn.__globals__
    function_def = get_function_source_node(fn)
    if isinstance(function_def, ast.Lambda):
        raise _dispatch_expr_error(
            function_def, "Dispatch must be a function definition"
        )

    local_exprs, return_expr = _collect_expression_body(fn, function_def)
    if not isinstance(return_expr, ast.Call):
        raise _dispatch_expr_error(
            return_expr,
            "Dispatch must return one CompileKey(...) call",
        )

    field_exprs: list[tuple[str, ast.AST]] = []
    defaults, candidate_names = _function_trace_inputs(fn)
    # Fields captured by dispatch **kwargs are forwarded, not AST-evaluated.
    signature = inspect.signature(fn)
    variadic_keyword = next(
        (
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        ),
        None,
    )
    candidate_names.discard(variadic_keyword)
    input_names: set[str] = set()
    local_names = {name for name, _ in local_exprs}
    for _, expr in local_exprs:
        input_names.update(_collect_input_names(expr, candidate_names))
    named_parameters: frozenset[str] | None = None
    for keyword in return_expr.keywords:
        # CompileKey may unpack only that **kwargs parameter, once.
        if keyword.arg is None:
            if (
                named_parameters is not None
                or variadic_keyword is None
                or not isinstance(keyword.value, ast.Name)
                or keyword.value.id != variadic_keyword
            ):
                raise ValueError(
                    f"{fn.__name__} may unpack only its own **kwargs parameter "
                    "once in CompileKey"
                )
            named_parameters = frozenset(candidate_names)
            continue
        field_exprs.append((keyword.arg, keyword.value))
        input_names.update(
            _collect_input_names(keyword.value, candidate_names, local_names)
        )

    return _CompileKeyDispatchTrace(
        tuple(local_exprs),
        tuple(field_exprs),
        globals_,
        frozenset(input_names),
        named_parameters,
        defaults,
    )


def _trace_warmup_predicate(
    fn: WarmupPredicateFn,
) -> _WarmupPredicateTrace:
    source_fn = getattr(fn, "__func__", fn)
    globals_ = source_fn.__globals__
    function_def = get_function_source_node(fn)

    local_exprs, return_expr = _collect_expression_body(fn, function_def)
    defaults, candidate_names = _function_trace_inputs(fn)
    input_names: set[str] = set()
    local_names = {name for name, _ in local_exprs}
    for _, expr in local_exprs:
        input_names.update(_collect_input_names(expr, candidate_names))
    input_names.update(_collect_input_names(return_expr, candidate_names, local_names))

    return _WarmupPredicateTrace(
        tuple(local_exprs),
        return_expr,
        globals_,
        frozenset(input_names),
        defaults,
    )


class VllmJitKernel(Generic[CompileKeyT], ABC):
    """Kernel wrapper that owns dispatch, warmup keys, and compilation."""

    CompileKey: type[CompileKeyT]

    def __init__(self) -> None:
        dispatch = type(self).dispatch
        self._dispatch_trace = (
            None
            if dispatch is VllmJitKernel.dispatch
            else _trace_compile_key_dispatch(self.dispatch)
        )
        self._compiled_cache: dict[Any, Any] = {}

    def compile_key(self, kwargs: Mapping[str, Any]) -> CompileKeyT:
        if self._dispatch_trace is None:
            raise TypeError(f"{type(self).__name__} does not define dispatch()")
        return self._dispatch_trace.compile_key(self.CompileKey, kwargs)

    def _get_or_compile(
        self,
        compile_key: CompileKeyT,
        *,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> Any:
        """Return a cached executor, compiling it on a monitored cache miss."""
        if compile_key not in self._compiled_cache:
            self.compile(compile_key)

        try:
            return self._compiled_cache[compile_key]
        except KeyError as exc:
            details = [f"compile_key={compile_key!r}"]
            if runtime_context:
                details.append(f"runtime_context={dict(runtime_context)!r}")
            raise RuntimeError(
                f"{type(self).__name__}.compile(...) did not cache its JIT "
                f"executor ({', '.join(details)})"
            ) from exc

    def _trace_dispatch(
        self, dispatch: CompileKeyDispatchFn[CompileKeyT]
    ) -> Callable[..., list[CompileKeyT]]:
        compile_key_dispatch_trace = _trace_compile_key_dispatch(dispatch)

        def traced(
            *input_groups: _WarmupInputRows,
            _when: WarmupPredicateFn | None = None,
            **kwargs: WarmupValues,
        ) -> list[CompileKeyT]:
            for group in input_groups:
                if not isinstance(group, _WarmupInputRows):
                    raise TypeError(
                        "_trace_dispatch positional arguments must be "
                        "zip_inputs(...) groups"
                    )
            predicate_trace = (
                _trace_warmup_predicate(_when) if _when is not None else None
            )
            predicate_only_names: frozenset[str] = frozenset()
            if predicate_trace is not None:
                compile_key_fields = frozenset(
                    field.name for field in fields(cast(Any, self.CompileKey))
                )
                predicate_only_names = (
                    predicate_trace.input_names
                    - compile_key_dispatch_trace.input_names
                    - compile_key_fields
                )
            # Unmatched **kwargs fields also belong to the expansion space.
            available_names = set(kwargs).union(
                *(group.rows[0] for group in input_groups)
            )
            input_names = compile_key_dispatch_trace.input_names_for(available_names)
            if predicate_trace is not None:
                input_names = input_names | predicate_trace.input_names
            expanded_input_groups = tuple(
                _expand_warmup_input_rows(group.rows, input_names)
                for group in input_groups
            )
            # Expand independent keyword inputs into cartesian-product axes.
            expanded_kwarg_axes = tuple(
                (name, _expand_warmup_values(value))
                for name, value in kwargs.items()
                if name in input_names
            )
            dispatch_value_axes = (
                *expanded_input_groups,
                *(values for _, values in expanded_kwarg_axes),
            )
            input_group_count = len(expanded_input_groups)
            kwarg_names = tuple(name for name, _ in expanded_kwarg_axes)
            compile_keys: dict[CompileKeyT, None] = {}
            for dispatch_value_set in itertools.product(*dispatch_value_axes):
                dispatch_values = _merge_warmup_kwargs(
                    (
                        *dispatch_value_set[:input_group_count],
                        dict(
                            zip(
                                kwarg_names,
                                dispatch_value_set[input_group_count:],
                            )
                        ),
                    )
                )
                if predicate_trace is not None and not predicate_trace.matches(
                    dispatch_values
                ):
                    continue
                compile_key = compile_key_dispatch_trace.compile_key(
                    self.CompileKey,
                    {
                        name: value
                        for name, value in dispatch_values.items()
                        if name not in predicate_only_names
                    },
                )
                compile_keys[compile_key] = None
            return list(compile_keys)

        return traced

    def _expand_warmup_cases(
        self,
        cases_fn: Callable[..., Any],
        *args: Any,
        _value_expander: Callable[[WarmupValues], tuple[Any, ...]] = (
            _expand_warmup_values
        ),
        **kwargs: Any,
    ) -> Iterator[Mapping[str, Any]]:
        """Expand symbolic domains declared inside a warmup-cases method."""
        function_def = get_function_source_node(cases_fn)
        if isinstance(function_def, ast.Lambda):
            raise _dispatch_expr_error(
                function_def, "Warmup cases must be a function definition"
            )
        signature = inspect.signature(cases_fn)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        static_values = dict(bound.arguments)
        bound_self = getattr(cases_fn, "__self__", None)
        if bound_self is not None:
            static_values["self"] = bound_self
        globals_ = getattr(cases_fn, "__func__", cases_fn).__globals__

        domains: list[tuple[str, tuple[Any, ...]]] = []
        local_exprs: list[tuple[str, ast.AST]] = []
        predicates: list[ast.AST] = []
        return_expr: ast.AST | None = None
        for statement in function_def.body:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            assignment = _named_assignment(statement, "Warmup case")
            if assignment is not None:
                name, value_expr = assignment
                call_name = (
                    (get_ast_full_name(value_expr.func) or "").split(".")[-1]
                    if isinstance(value_expr, ast.Call)
                    else ""
                )
                if call_name in {"WarmupIntRange", "WarmupChoices"}:
                    value = _eval_dispatch_expr(value_expr, static_values, globals_)
                    domains.append((name, _value_expander(value)))
                else:
                    local_exprs.append((name, value_expr))
                continue
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and (get_ast_full_name(statement.value.func) or "").endswith("_when")
            ):
                if len(statement.value.args) != 1 or statement.value.keywords:
                    raise _dispatch_expr_error(
                        statement, "_when requires exactly one positional expression"
                    )
                predicates.append(statement.value.args[0])
                continue
            if isinstance(statement, ast.Return):
                if return_expr is not None or statement.value is None:
                    raise _dispatch_expr_error(
                        statement, "Warmup cases require one final return expression"
                    )
                return_expr = statement.value
                continue
            raise _dispatch_expr_error(
                statement,
                "AST-traced warmup cases support assignments, _when, and return",
            )

        if return_expr is None or not isinstance(return_expr, ast.Call):
            raise ValueError("AST-traced warmup cases must return one case(...) call")
        if any(keyword.arg is None for keyword in return_expr.keywords):
            raise _dispatch_expr_error(
                return_expr, "Warmup case expressions do not support **kwargs"
            )

        class _InlineDomainRewriter(ast.NodeTransformer):
            def visit_Call(self, node: ast.Call) -> ast.AST:
                call_name = (get_ast_full_name(node.func) or "").split(".")[-1]
                if call_name not in {"WarmupIntRange", "WarmupChoices"}:
                    return self.generic_visit(node)
                name = f"__warmup_domain_{len(domains)}"
                domain = _eval_dispatch_expr(node, static_values, globals_)
                domains.append((name, _value_expander(domain)))
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)

        return_expr = cast(ast.Call, _InlineDomainRewriter().visit(return_expr))

        domain_names = tuple(name for name, _ in domains)
        dynamic_names = set(domain_names)
        dynamic_local_exprs: list[tuple[str, ast.AST]] = []
        for name, expr in local_exprs:
            referenced_names = {
                node.id for node in ast.walk(expr) if isinstance(node, ast.Name)
            }
            if referenced_names & dynamic_names:
                dynamic_local_exprs.append((name, expr))
                dynamic_names.add(name)
            else:
                static_values[name] = _eval_dispatch_expr(expr, static_values, globals_)

        for _, expr in dynamic_local_exprs:
            _validate_dispatch_expr(expr)
        for predicate in predicates:
            _validate_dispatch_expr(predicate)
        _validate_dispatch_expr(return_expr)

        case_body: list[ast.stmt] = [
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=cast(ast.expr, expr),
            )
            for name, expr in dynamic_local_exprs
        ]
        if predicates:
            predicate = (
                predicates[0]
                if len(predicates) == 1
                else ast.BoolOp(
                    op=ast.And(),
                    values=[cast(ast.expr, value) for value in predicates],
                )
            )
            case_body.append(
                ast.If(
                    test=ast.UnaryOp(op=ast.Not(), operand=cast(ast.expr, predicate)),
                    body=[ast.Return(value=ast.Constant(value=None))],
                    orelse=[],
                )
            )
        case_body.append(ast.Return(value=cast(ast.expr, return_expr)))
        # FunctionDef fields vary across the supported Python versions, so no
        # single constructor overload matches every mypy target.
        case_function = ast.FunctionDef(  # type: ignore[call-overload]
            name="__vllm_warmup_case",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=name) for name in domain_names],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=case_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        case_globals = dict(globals_)
        case_globals.update(static_values)
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[case_function], type_ignores=[])
                ),
                "<jit-warmup>",
                "exec",
            ),
            case_globals,
        )
        evaluate_case = cast(Callable[..., Any], case_globals[case_function.name])

        domain_values = tuple(values for _, values in domains)
        for values in itertools.product(*domain_values):
            case = evaluate_case(*values)
            if case is None:
                continue
            if not isinstance(case, Mapping):
                raise TypeError("AST-traced warmup cases must return a mapping")
            if any(
                isinstance(value, WarmupChoices | WarmupIntRange)
                for value in case.values()
            ):
                raise TypeError("Warmup domains must be direct expressions")
            yield case

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

    def register_warmup(self, *args: Any, **kwargs: Any) -> None:
        """Register this kernel with the active runner's warmup registry."""
        JitWarmupRegistry.register(self, *args, **kwargs)

    def warmup(self, *args: Any, **kwargs: Any) -> None:
        """Compile this kernel's warmup keys."""
        for compile_key in self.get_warmup_keys(*args, **kwargs):
            self.compile(compile_key)


def _same_value(left: Any, right: Any) -> bool:
    """True if two registration values are the same object or compare equal.

    Identity is checked first so shared singletons (e.g. ``vllm_config``, torch
    dtypes) short-circuit before any potentially deep or non-boolean ``__eq__``.
    """
    if left is right:
        return True
    try:
        return bool(left == right)
    except (TypeError, ValueError, RuntimeError):
        return False


def _same_registration(
    left: tuple[tuple[Any, ...], dict[str, Any]],
    right: tuple[tuple[Any, ...], dict[str, Any]],
) -> bool:
    """True if two ``(args, kwargs)`` registrations are equivalent."""
    left_args, left_kwargs = left
    right_args, right_kwargs = right
    if len(left_args) != len(right_args) or left_kwargs.keys() != right_kwargs.keys():
        return False
    return all(
        _same_value(x, y) for x, y in zip(left_args, right_args, strict=True)
    ) and all(_same_value(left_kwargs[k], right_kwargs[k]) for k in left_kwargs)


class JitWarmupRegistry:
    """Collect and compile JIT kernels selected during runner setup."""

    _active: ContextVar[JitWarmupRegistry | None] = ContextVar(
        "active_jit_warmup_registry",
        default=None,
    )

    def __init__(self, vllm_config: Any) -> None:
        self.vllm_config = vllm_config
        self._registrations: dict[
            VllmJitKernel[Any],
            list[tuple[tuple[Any, ...], dict[str, Any]]],
        ] = {}

    @classmethod
    def capture(cls, init_fn: Callable[..., None]) -> Callable[..., None]:
        """Collect warmup registrations made while the decorated callable runs."""

        @wraps(init_fn)
        def wrapped(instance: Any, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
            registry = cls(vllm_config)
            instance.jit_warmup_registry = registry
            with registry.activate():
                init_fn(instance, vllm_config, *args, **kwargs)

        return wrapped

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Collect registrations made in this context."""
        token = self._active.set(self)
        try:
            yield
        finally:
            self._active.reset(token)

    @classmethod
    def register(
        cls,
        kernel: VllmJitKernel[Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register a kernel with the active registry, if one exists."""
        registry = cls._active.get()
        if registry is not None:
            registry._add(kernel, args, kwargs)

    def _add(
        self,
        kernel: VllmJitKernel[Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        registrations = self._registrations.setdefault(kernel, [])
        # Every layer of a deep model appends an identical (args, kwargs) here
        # (e.g. a 61-layer DSA model registers the same pack (dtype, pad_value)
        # 61 times); each expands to the same compile keys, so tracing
        # get_warmup_keys once per distinct registration is sufficient. Dedup on
        # identity-or-equality -- identity short-circuits shared singletons like
        # vllm_config before any deep __eq__.
        if any(
            _same_registration(registered, (args, kwargs))
            for registered in registrations
        ):
            return
        registrations.append((args, kwargs))

    def __len__(self) -> int:
        return sum(len(registrations) for registrations in self._registrations.values())

    def warmup(self) -> None:
        """Expand registrations and compile each wrapper/key pair once."""
        from tqdm import tqdm

        from vllm.distributed import is_global_first_rank

        kernel_items: list[tuple[VllmJitKernel[Any], dict[Any, None]]] = []
        for kernel, registrations in self._registrations.items():
            compile_keys: dict[Any, None] = {}
            for args, kwargs in registrations:
                if (
                    not args
                    and not kwargs
                    and "vllm_config"
                    in inspect.signature(kernel.get_warmup_keys).parameters
                ):
                    kwargs = {"vllm_config": self.vllm_config}
                for compile_key in kernel.get_warmup_keys(*args, **kwargs):
                    compile_keys[compile_key] = None
            if compile_keys:
                kernel_items.append((kernel, compile_keys))

        if not kernel_items:
            return

        total_keys = sum(len(compile_keys) for _, compile_keys in kernel_items)
        with tqdm(
            kernel_items,
            desc=f"JIT kernel warmup ({total_keys} compile keys)",
            disable=not is_global_first_rank(),
            dynamic_ncols=True,
            unit="kernel",
        ) as progress:
            for kernel, compile_keys in progress:
                progress.set_postfix_str(
                    f"{kernel.__class__.__name__} ({len(compile_keys)} keys)",
                    refresh=False,
                )
                for compile_key in compile_keys:
                    kernel.compile(compile_key)
