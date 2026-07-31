# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm.model_executor.warmup.jit_warmup import (
    get_ast_full_name,
    get_function_source_node,
)


@dataclass(frozen=True)
class TritonWarmupTensor:
    # Compile-only tensor descriptor for Triton pointer specialization.
    dtype: Any
    aligned: bool = True
    shape: tuple[int, ...] = (1,)

    def data_ptr(self) -> int:
        return 0 if self.aligned else 1

    def ptr_range(self) -> int:
        return 0

    def stride(self) -> tuple[int, ...]:
        strides: list[int] = []
        stride = 1
        for size in reversed(self.shape):
            strides.append(stride)
            stride *= size
        return tuple(reversed(strides))


@dataclass(frozen=True)
class TritonPointerInputVariant:
    # Named pointer-alignment variant for compile-only Triton warmup.
    alignments: tuple[tuple[str, bool], ...]

    @classmethod
    def from_alignment(cls, **aligned: bool) -> "TritonPointerInputVariant":
        return cls(tuple(aligned.items()))

    def is_aligned(self, name: str) -> bool:
        for alignment_name, aligned in self.alignments:
            if alignment_name == name:
                return aligned
        raise KeyError(f"Unknown Triton pointer input variant: {name}")

    def pointer(
        self,
        name: str,
        dtype: Any,
        shape: tuple[int, ...] = (1,),
    ) -> TritonWarmupTensor:
        return TritonWarmupTensor(dtype, aligned=self.is_aligned(name), shape=shape)


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
        decorator_name = get_ast_full_name(decorator.func)
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
        annotation = get_ast_full_name(arg.annotation)
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
        if get_ast_full_name(node.func) not in ("tl.load", "tl.store"):
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
    function_def = get_function_source_node(kernel)
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
