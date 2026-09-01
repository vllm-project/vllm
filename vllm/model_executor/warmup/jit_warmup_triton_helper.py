# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import inspect
from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property, wraps
from math import prod
from typing import Any, ClassVar, Generic, TypeVar, cast

import torch

from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    get_ast_full_name,
    get_function_source_node,
)

CompileKeyT = TypeVar("CompileKeyT")
TritonWarmupInputs = dict[str, Any] | tuple[tuple[Any, ...], dict[str, Any]]
LaunchSpec = tuple[Any | None, dict[str, Any]] | tuple[Any | None, dict[str, Any], Any]


def triton_scalar_specialization_rep(value: int) -> int:
    """Return an integer with the same default Triton JIT specialization.

    For an ordinary integer argument, Triton's cache key contains its inferred
    type (``i32``, ``i64``, or ``u64``) and one of three value classes:

    * ``1`` is specialized as the exact constant ``1``.
    * Multiples of 16 receive a ``tt.divisibility = 16`` attribute.
    * All other values have no value specialization.

    Warmup only needs one concrete value for each cache-key class. This helper
    returns ``1`` for the exact-one class and otherwise returns a divisible or
    generic representative while preserving the inferred integer type.

    This applies only to non-``constexpr`` integer arguments using Triton's
    default specialization. Do not use it for arguments listed in
    ``do_not_specialize`` or ``do_not_specialize_on_alignment``.
    """
    if value == 1:
        return 1

    if -(1 << 31) <= value < (1 << 31):
        divisible_rep = 16
        generic_rep = 2
    elif -(1 << 63) <= value < (1 << 63):
        divisible_rep = 1 << 31
        generic_rep = (1 << 31) + 1
    elif 0 <= value < (1 << 64):
        divisible_rep = 1 << 63
        generic_rep = (1 << 63) + 1
    else:
        raise OverflowError(f"Integer {value} is outside Triton's scalar range")

    return divisible_rep if value % 16 == 0 else generic_rep


@dataclass(frozen=True)
class TritonWarmupTensor:
    """Compile-only tensor metadata used by Triton warmup.

    ``strides=None`` represents compact row-major storage. Pass explicit strides
    whenever the runtime tensor can be padded, transposed, or otherwise strided.
    """

    dtype: Any
    aligned: bool = True
    shape: tuple[int, ...] = (1,)
    strides: tuple[int, ...] | None = None

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def data_ptr(self) -> int:
        return 0 if self.aligned else 1

    def ptr_range(self) -> int:
        return 0

    def stride(self, dim: int | None = None) -> int | tuple[int, ...]:
        if self.strides is None:
            strides: list[int] = []
            stride = 1
            for size in reversed(self.shape):
                strides.append(stride)
                stride *= size
            result = tuple(reversed(strides))
        else:
            result = self.strides
        return result if dim is None else result[dim]

    def size(self, dim: int | None = None) -> int | tuple[int, ...]:
        return self.shape if dim is None else self.shape[dim]

    def numel(self) -> int:
        return prod(self.shape)

    def element_size(self) -> int:
        return self.dtype.itemsize

    def is_contiguous(self) -> bool:
        if self.strides is None or self.numel() == 0:
            return True

        expected_stride = 1
        for size, stride in zip(
            reversed(self.shape), reversed(self.strides), strict=True
        ):
            if size > 1 and stride != expected_stride:
                return False
            expected_stride *= size
        return True

    def reshape(self, *shape: int) -> "TritonWarmupTensor":
        return TritonWarmupTensor(self.dtype, self.aligned, tuple(shape))

    def new_empty(
        self,
        shape: tuple[int, ...],
        *,
        dtype: Any | None = None,
    ) -> "TritonWarmupTensor":
        return TritonWarmupTensor(dtype or self.dtype, self.aligned, shape)

    def int(self) -> "TritonWarmupTensor":
        return TritonWarmupTensor(
            torch.int32,
            self.aligned,
            self.shape,
            self.strides,
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func is torch.empty_like:
            tensor = args[0]
            options = kwargs or {}
            return cls(
                options.get("dtype", tensor.dtype),
                tensor.aligned,
                tensor.shape,
                tensor.strides,
            )
        return NotImplemented


class VllmTritonJitKernel(VllmJitKernel[CompileKeyT], Generic[CompileKeyT]):
    """Triton owner whose runtime launch specification is reused for warmup."""

    kernel: ClassVar[Any]
    _warming = False

    @abstractmethod
    def warmup_inputs(self, compile_key: CompileKeyT) -> TritonWarmupInputs:
        """Return runtime-shaped inputs that reproduce one compile key."""
        raise NotImplementedError

    def compile(self, compile_key: CompileKeyT) -> None:
        inputs = self.warmup_inputs(compile_key)
        self._warming = True
        try:
            call = cast(Callable[..., Any], self)
            if isinstance(inputs, tuple):
                call(*inputs[0], **inputs[1])
            else:
                call(**inputs)
        finally:
            self._warming = False

    @cached_property
    def _kernel_param_names(self) -> frozenset[str]:
        return frozenset(self.kernel.arg_names)

    def launch(
        self,
        grid: Any,
        inputs: Mapping[str, Any],
        variadic_inputs: tuple[Any, ...] = (),
        /,
        **kwargs: Any,
    ) -> Any:
        forwarded: dict[str, Any] = {}
        for name, value in inputs.items():
            target = name if name in self._kernel_param_names else f"{name}_ptr"
            if target in self._kernel_param_names and target not in kwargs:
                forwarded[target] = value
        remaining_names = [
            name
            for name in self.kernel.arg_names
            if name not in kwargs and name not in forwarded
        ]
        if len(variadic_inputs) > len(remaining_names):
            raise TypeError("Too many positional Triton kernel arguments")
        forwarded.update(zip(remaining_names, variadic_inputs, strict=False))
        runtime_launcher = kwargs.pop("_runtime_launcher", None)
        runtime_launcher_arg_count = kwargs.pop("_runtime_launcher_arg_count", 0)
        if self._warming:
            warmup = getattr(self.kernel, "warmup", None)
            assert warmup is not None
            kwargs.update(forwarded)
            return warmup(grid=(1,), **kwargs)
        if runtime_launcher is not None:
            kwargs.update(forwarded)
            # Some launch contexts cache this positional runtime prefix.
            regular_args = [
                kwargs.pop(name)
                for name in self.kernel.arg_names[:runtime_launcher_arg_count]
            ]
            return runtime_launcher(self.kernel, grid, *regular_args, **kwargs)
        kwargs.update(forwarded)
        return self.kernel[grid](**kwargs)


def kernel_launcher(
    call_fn: Callable[..., LaunchSpec],
) -> Callable[..., Any]:
    """Launch a Triton kernel from a declarative ``__call__`` specification."""
    signature = inspect.signature(call_fn)

    @wraps(call_fn)
    def wrapper(
        self: VllmTritonJitKernel[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        launch_spec = call_fn(self, *args, **kwargs)
        grid, launch_kwargs = launch_spec[:2]
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        inputs: dict[str, Any] = {}
        variadic_inputs: tuple[Any, ...] = ()
        for name, value in bound.arguments.items():
            if name == "self":
                continue
            if signature.parameters[name].kind is inspect.Parameter.VAR_POSITIONAL:
                variadic_inputs = value
            else:
                inputs[name] = value
        if grid is not None:
            self.launch(grid, inputs, variadic_inputs, **launch_kwargs)
        return launch_spec[2] if len(launch_spec) == 3 else None

    return wrapper


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
    if not isinstance(function_def, ast.FunctionDef):
        raise ValueError("Expected Triton kernel to be defined as a function")
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
