# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import inspect
import itertools
import operator
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Generic, TypeVar

from vllm.config import VllmConfig

CompileKeyT = TypeVar("CompileKeyT")


@dataclass(frozen=True)
class WarmupIntRange:
    start: int
    stop: int
    step: int = 1


WarmupIntValues = int | WarmupIntRange | list[int] | tuple[int, ...]


def expand_warmup_int_values(values: WarmupIntValues) -> tuple[int, ...]:
    if isinstance(values, WarmupIntRange):
        return tuple(range(values.start, values.stop, values.step))
    if isinstance(values, int):
        return (values,)
    return tuple(values)


@dataclass(frozen=True)
class _DispatchTrace:
    field_exprs: tuple[tuple[str, ast.AST], ...]
    globals: Mapping[str, Any]
    input_names: frozenset[str]

    def compile_key(
        self,
        compile_key_type: type[CompileKeyT],
        kwargs: Mapping[str, Any],
    ) -> CompileKeyT:
        return compile_key_type(
            **{
                field: int(_eval_dispatch_expr(expr, kwargs, self.globals))
                for field, expr in self.field_exprs
            }
        )


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _full_attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _full_attr_name(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def _get_function_source_node(fn: Callable[..., Any]) -> ast.FunctionDef:
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
        full_name = _full_attr_name(node.func)
        args = [_eval_dispatch_expr(arg, kwargs, globals_) for arg in node.args]
        if full_name == "triton.next_power_of_2":
            if len(args) != 1 or node.keywords:
                raise ValueError("triton.next_power_of_2 dispatch call takes one arg")
            return _next_power_of_2(int(args[0]))

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


def trace_dispatch(fn: Callable[..., Any]) -> _DispatchTrace:
    source_fn = getattr(fn, "__func__", fn)
    globals_ = source_fn.__globals__
    function_def = _get_function_source_node(fn)

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
    candidate_names = set(signature.parameters)
    input_names: set[str] = set()
    for keyword in returns[0].keywords:
        if keyword.arg is None:
            raise ValueError(f"{fn.__name__} cannot use **kwargs in CompileKey")
        field_exprs.append((keyword.arg, keyword.value))
        input_names.update(_collect_input_names(keyword.value, candidate_names))

    return _DispatchTrace(tuple(field_exprs), globals_, frozenset(input_names))


def _literal_str_refs(node: ast.AST) -> tuple[str | int, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str | int):
        return (node.value,)
    if isinstance(node, ast.List | ast.Tuple):
        refs: list[str | int] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str | int):
                refs.append(elt.value)
            else:
                raise ValueError(
                    f"Unsupported Triton specialization ref: {ast.dump(elt)}"
                )
        return tuple(refs)
    raise ValueError(f"Unsupported Triton specialization refs: {ast.dump(node)}")


def _normalize_arg_refs(
    refs: tuple[str | int, ...],
    arg_names: tuple[str, ...],
) -> frozenset[str]:
    names: set[str] = set()
    for ref in refs:
        if isinstance(ref, int):
            names.add(arg_names[ref])
        else:
            names.add(ref)
    return frozenset(names)


def _decorator_keyword_refs(
    function_def: ast.FunctionDef,
    keyword_name: str,
) -> tuple[str | int, ...]:
    for decorator in function_def.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        decorator_name = _full_attr_name(decorator.func)
        if decorator_name not in ("triton.jit", "jit"):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == keyword_name:
                return _literal_str_refs(keyword.value)
    return ()


def _triton_do_not_specialize_args(
    kernel: Callable[..., Any],
    function_def: ast.FunctionDef,
    arg_names: tuple[str, ...],
) -> frozenset[str]:
    refs = getattr(kernel, "do_not_specialize", None)
    if refs is not None:
        return _normalize_arg_refs(tuple(refs), arg_names)
    return _normalize_arg_refs(
        _decorator_keyword_refs(function_def, "do_not_specialize"),
        arg_names,
    )


def _triton_constexpr_arg_names(
    kernel: Callable[..., Any],
    function_def: ast.FunctionDef,
    arg_names: tuple[str, ...],
) -> frozenset[str]:
    constexprs = getattr(kernel, "constexprs", None)
    if constexprs is not None:
        return frozenset(arg_names[index] for index in constexprs)

    names: set[str] = set()
    for arg in function_def.args.args + function_def.args.kwonlyargs:
        if arg.annotation is None:
            continue
        annotation = _full_attr_name(arg.annotation)
        if annotation in ("tl.constexpr", "triton.language.constexpr", "constexpr"):
            names.add(arg.arg)
    return frozenset(names)


def _leftmost_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BinOp):
        return _leftmost_name(node.left)
    return None


def _pointer_arg_names(
    function_def: ast.FunctionDef,
    arg_names: tuple[str, ...],
) -> frozenset[str]:
    candidate_names = set(arg_names)
    pointer_names = {name for name in arg_names if name.endswith("_ptr")}
    for node in ast.walk(function_def):
        if not isinstance(node, ast.Call):
            continue
        if _full_attr_name(node.func) not in ("tl.load", "tl.store"):
            continue
        if not node.args:
            continue
        name = _leftmost_name(node.args[0])
        if name in candidate_names:
            pointer_names.add(name)
    return frozenset(pointer_names)


def trace_triton_kernel_specialization_args(
    kernel: Callable[..., Any],
) -> tuple[str, ...]:
    function_def = _get_function_source_node(kernel)
    source_fn = getattr(kernel, "fn", kernel)
    arg_names = tuple(inspect.signature(source_fn).parameters)
    constexpr_args = _triton_constexpr_arg_names(kernel, function_def, arg_names)
    do_not_specialize_args = _triton_do_not_specialize_args(
        kernel, function_def, arg_names
    )
    pointer_args = _pointer_arg_names(function_def, arg_names)

    return tuple(
        name
        for name in arg_names
        if name in constexpr_args
        or (name not in pointer_args and name not in do_not_specialize_args)
    )


class VllmJitKernelWithWarmup(Generic[CompileKeyT], ABC):
    CompileKey: type[CompileKeyT]

    def __init__(self) -> None:
        self.traced_dispatch_args = trace_dispatch(self.dispatch)
        self.callables: dict[CompileKeyT, Callable[..., Any]] = {}

    def compile_key(self, kwargs: Mapping[str, Any]) -> CompileKeyT:
        return self.traced_dispatch_args.compile_key(self.CompileKey, kwargs)

    def assert_compile_key_matches_triton(
        self,
        kernel: Callable[..., Any],
    ) -> None:
        if not is_dataclass(self.CompileKey):
            raise TypeError(f"{type(self).__name__}.CompileKey must be a dataclass")

        compile_key_args = tuple(field.name for field in fields(self.CompileKey))
        triton_args = trace_triton_kernel_specialization_args(kernel)
        if compile_key_args != triton_args:
            raise ValueError(
                f"{type(self).__name__}.CompileKey fields {compile_key_args} "
                f"do not match Triton specialization args {triton_args}"
            )

    def _trace_dispatch(
        self, dispatch: Callable[..., CompileKeyT]
    ) -> Callable[..., list[CompileKeyT]]:
        traced_dispatch_args = trace_dispatch(dispatch)

        def traced(**kwargs: WarmupIntValues) -> list[CompileKeyT]:
            kwarg_names = tuple(
                name for name in kwargs if name in traced_dispatch_args.input_names
            )
            kwarg_values = tuple(
                expand_warmup_int_values(kwargs[name]) for name in kwarg_names
            )
            return list(
                dict.fromkeys(
                    traced_dispatch_args.compile_key(
                        self.CompileKey, dict(zip(kwarg_names, values))
                    )
                    for values in itertools.product(*kwarg_values)
                )
            )

        return traced

    @abstractmethod
    def dispatch(self, **kwargs: Any) -> CompileKeyT:
        raise NotImplementedError

    @abstractmethod
    def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKeyT]:
        raise NotImplementedError

    @abstractmethod
    def compile(self, compile_key: CompileKeyT) -> Callable[..., Any]:
        raise NotImplementedError

    def warmup(self, vllm_config: VllmConfig) -> None:
        for compile_key in self.get_warmup_keys(vllm_config):
            if compile_key not in self.callables:
                self.callables[compile_key] = self.compile(compile_key)

    def __call__(self, **kwargs: Any) -> Any:
        compile_key = self.compile_key(kwargs)
        fn = self.callables.get(compile_key)
        if fn is None:
            fn = self.compile(compile_key)
            self.callables[compile_key] = fn
        return fn(**kwargs)

    def saved_compile_keys(self) -> tuple[CompileKeyT, ...]:
        return tuple(self.callables)
