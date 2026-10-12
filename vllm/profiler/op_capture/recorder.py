# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Records the operators a model dispatches, attributed to the issuing module."""

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.utils._python_dispatch import TorchDispatchMode

import vllm
from vllm.profiler.op_capture.meta_ops import (
    LEAF_NAMESPACES,
    NATIVE_PREFIXES,
    UnsupportedMetaOpError,
    placeholder_outputs,
)
from vllm.utils.torch_utils import DIRECT_REGISTERED_OPS

_DTYPE_ABBREVIATIONS = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.float8_e4m3fn: "f8e4m3",
    torch.float8_e5m2: "f8e5m2",
    torch.int8: "i8",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint8: "u8",
    torch.bool: "bool",
}


def vllm_location(exception: BaseException) -> str:
    """Innermost vLLM source line on `exception`'s traceback, outside this package.

    Returns:
        `"path:line in function"` relative to the repository root, or `""`.

    """
    vllm_root = Path(vllm.__file__).parent
    package = Path(__file__).parent
    location = ""
    for frame in traceback.extract_tb(exception.__traceback__):
        path = Path(frame.filename)
        if path.is_relative_to(vllm_root) and not path.is_relative_to(package):
            relative = path.relative_to(vllm_root.parent)
            location = f"{relative}:{frame.lineno} in {frame.name}"
    return location


def describe(value: Any) -> str:
    """Render one operator argument or result as a short shape/dtype string."""
    if isinstance(value, torch.Tensor):
        dtype = _DTYPE_ABBREVIATIONS.get(value.dtype, str(value.dtype)[6:])
        return f"{dtype}[{','.join(map(str, value.shape))}]"
    if isinstance(value, (list, tuple)):
        return f"[{', '.join(map(describe, value))}]"
    if isinstance(value, torch.dtype):
        return _DTYPE_ABBREVIATIONS.get(value, str(value)[6:])
    if isinstance(value, (bool, int, float, str)) or value is None:
        return repr(value)
    return type(value).__name__


_UNDEFINED = torch.empty(())
"""Stands in for an undefined tensor, shaped as a profiler renders one. Allocated
once, so recording a forward pass dispatches nothing of its own."""


def _marshalable(result: Any) -> Any:
    """Make a `Tensor[]` return value convertible back to its declared type.

    A compiled kernel may leave an entry of a `Tensor[]` return undefined --
    FlashAttention's `softmax_lse` when it was not asked for, say -- which
    reaches Python as `None`. PyTorch cannot convert that back into the op's
    declared return type, which only matters when the op is dispatched through a
    mode, so a placeholder stands in for the value the caller discards.
    """
    if not isinstance(result, list) or not any(item is None for item in result):
        return result
    return [_UNDEFINED if item is None else item for item in result]


@dataclass
class RecordedOp:
    """One dispatched operator, in execution order."""

    name: str
    """Qualified operator name, e.g. `"_C::rms_norm"`."""
    module: str
    """Dotted path of the innermost module that issued the op, `""` for none."""
    module_type: str
    """Class name of that module."""
    depth: int
    """Dispatch nesting depth: 0 when module code issued the op directly, 1 when
    another custom op's kernel issued it, and so on."""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    placeholder: bool = False
    """Whether the outputs are placeholders, the op's shapes being unknown."""
    body_error: str = ""
    """Why this custom op's own kernel stopped on the meta device, `""` if it
    did not. Its outputs then come from its fake kernel, or are placeholders,
    and the ops its kernel would have dispatched after the error are missing."""

    @property
    def is_custom(self) -> bool:
        """Whether this is a vLLM custom op rather than a native PyTorch one."""
        return not self.name.startswith(NATIVE_PREFIXES)

    @property
    def is_leaf_kernel(self) -> bool:
        """Whether this is a compiled kernel, which dispatches nothing further."""
        return self.name.partition("::")[0] in LEAF_NAMESPACES

    @property
    def signature(self) -> str:
        return f"{self.name}({', '.join(self.inputs)})"


class OpRecorder(TorchDispatchMode):
    """Dispatch mode that records every operator a forward pass executes.

    Operators are recorded before they run, so a custom op precedes the leaf
    kernels its own implementation dispatches. Module attribution comes from
    global `nn.Module` forward hooks, so it follows the real call stack rather
    than parameter ownership.

    Args:
        model: Model whose `named_modules()` supplies the dotted module paths.
        dispatch_key: Dispatch key the model's kernels are registered under,
            `"Meta"` for a meta-device capture.
        keep_going: Give an op that raises `UnsupportedMetaOpError`
            placeholder outputs and carry on, instead of propagating the error.
            On the meta device, likewise finish a custom op whose kernel fails
            partway -- reading a tensor's value, say -- with its fake kernel,
            or placeholders.

    """

    def __init__(
        self, model: nn.Module, dispatch_key: str = "Meta", keep_going: bool = False
    ):
        super().__init__()
        self.ops: list[RecordedOp] = []
        self.module_types: dict[str, str] = {}
        """Class name of every module entered, keyed by dotted path."""
        self._keyset = torch._C.DispatchKeySet(
            getattr(torch._C.DispatchKey, dispatch_key)
        )
        self._paths = {id(module): name for name, module in model.named_modules()}
        self._children: dict[int, dict[int, str]] = {}
        self._stack: list[tuple[nn.Module, str, str]] = []
        self._handles: list[Any] = []
        self._keep_going = keep_going
        self._on_meta = dispatch_key == "Meta"
        self._raised_in: list[tuple[BaseException, str]] = []
        self._entered = 0
        self._depth = 0

    def __enter__(self) -> "OpRecorder":
        # Re-entered once per nested custom op, so the module hooks are installed
        # and the attribution stack torn down only by the outermost entry.
        if self._entered == 0:
            self._handles = [
                register_module_forward_pre_hook(self._enter_module),
                register_module_forward_hook(self._exit_module, always_call=True),
            ]
        self._entered += 1
        super().__enter__()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        super().__exit__(*exc_info)
        self._entered -= 1
        if self._entered == 0:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._stack.clear()

    def _child_name(self, parent: nn.Module, module: nn.Module) -> str | None:
        names = self._children.get(id(parent))
        if names is None:
            names = {id(child): name for name, child in parent.named_children()}
            self._children[id(parent)] = names
        return names.get(id(module))

    def _enter_module(self, module: nn.Module, args: Any) -> None:
        if id(module) not in self._paths:
            return
        path = self._paths[id(module)]
        if self._stack:
            # A module shared between layers (a cached rotary embedding, say) has
            # one canonical name but several call sites, so name it through the
            # caller to keep each layer's ops in its own subtree.
            parent, parent_path, _ = self._stack[-1]
            name = self._child_name(parent, module)
            if name is not None:
                path = f"{parent_path}.{name}" if parent_path else name
        self.module_types[path] = type(module).__name__
        self._stack.append((module, path, type(module).__name__))

    def _exit_module(self, module: nn.Module, args: Any, output: Any) -> None:
        if self._stack and self._stack[-1][0] is module:
            _, path, _ = self._stack.pop()
            # Called while the exception is still being handled, innermost
            # module first.
            exception = sys.exc_info()[1]
            if exception is not None and all(
                seen is not exception for seen, _ in self._raised_in
            ):
                self._raised_in.append((exception, path))

    def module_raising(self, exception: BaseException) -> str | None:
        """Dotted path of the innermost module `exception` propagated out of."""
        return next((path for seen, path in self._raised_in if seen is exception), None)

    def record_launch(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> RecordedOp:
        """Record an operator at the current module and nesting depth.

        Dispatched operators are recorded automatically; call this for a kernel
        launched outside the dispatcher, such as a raw Triton kernel.

        Args:
            name: Qualified name to record it under.
            args: Its positional arguments.
            kwargs: Its keyword arguments.

        Returns:
            The record, already appended to `ops`.

        """
        _, path, module_type = self._stack[-1] if self._stack else (None, "", "")
        record = RecordedOp(
            name=name,
            module=path,
            module_type=module_type,
            depth=self._depth,
            inputs=tuple(map(describe, args))
            + tuple(f"{key}={describe(value)}" for key, value in kwargs.items()),
        )
        self.ops.append(record)
        return record

    def _finish_with_fake(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        exception: Exception,
    ) -> Any:
        """Outputs of a custom op whose kernel failed, from its fake kernel.

        Falls back to placeholder outputs for an op registered without one.

        Raises:
            Exception: `exception`, if neither can produce the outputs.

        """
        try:
            return func.redispatch(self._keyset, *args, **kwargs)
        except Exception:
            try:
                return placeholder_outputs(func._schema, args, kwargs)
            except UnsupportedMetaOpError:
                raise exception from None

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        record = self.record_launch(func.name(), args, kwargs)
        body = DIRECT_REGISTERED_OPS.get(record.name)
        if body is not None or (record.is_custom and not record.is_leaf_kernel):
            # vLLM's glue ops have Python kernels that dispatch to compiled
            # leaves, so stay installed while the kernel runs. Call a known body
            # directly -- a glue op with a `register_fake` would otherwise
            # short-circuit on the meta device, hiding the kernels it calls --
            # and otherwise enter below the Python key, so this same call is not
            # intercepted again.
            self._depth += 1
            try:
                with self:
                    if body is not None:
                        result = body(*args, **kwargs)
                    else:
                        result = func.redispatch(self._keyset, *args, **kwargs)
            except Exception as exception:
                if not (self._keep_going and self._on_meta):
                    raise
                result = self._finish_with_fake(func, args, kwargs, exception)
                record.body_error = f"{type(exception).__name__}: {exception}"
                if location := vllm_location(exception):
                    record.body_error += f" (at {location})"
            finally:
                self._depth -= 1
        else:
            try:
                result = func(*args, **kwargs)
            except UnsupportedMetaOpError:
                if not self._keep_going:
                    raise
                result = placeholder_outputs(func._schema, args, kwargs)
                record.placeholder = True
        result = _marshalable(result)
        record.outputs = () if result is None else (describe(result),)
        return result
