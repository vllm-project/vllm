# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry for overriding Triton kernels with platform implementations.

Non-CUDA platforms can replace Triton kernels defined in vLLM core
without modifying core code:

    from vllm.triton_utils.dispatcher import register_kernels

    register_kernels({
        "vllm.v1.sample.rejection_sampler.expand_kernel": my_expand_impl,
        "vllm.v1.worker.mamba_utils.batch_memcpy_kernel": my_memcpy_impl,
    })
"""

import importlib
import inspect
import sys
from collections.abc import Callable, Mapping
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_registry: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {}


def _kernel_arg_names(kernel: Callable[..., Any]) -> tuple[str, ...]:
    """Return the argument names of a Triton kernel or its fallback."""
    names = getattr(kernel, "arg_names", None)
    if names is None:
        # No real Triton (TritonPlaceholder): read the plain function.
        fn = getattr(kernel, "func", kernel)
        names = tuple(inspect.signature(fn).parameters)
    return tuple(names)


class KernelOverride:
    """Stands in for a Triton kernel and routes launches to ``impl``.

    The JIT warmup infrastructure launches kernels as
    ``kernel[grid](**kwargs)`` where ``kwargs`` are keyed by the original
    kernel's argument names. This wrapper forwards those launch arguments
    to the platform implementation: by keyword when the implementation's
    parameter names match the kernel's, otherwise positionally in the
    kernel's parameter order.
    """

    def __init__(self, kernel: Callable[..., Any], impl: Callable[..., Any]) -> None:
        self._impl = impl
        # Mirror the original kernel's names so launch binding and warmup
        # introspection keep working against this wrapper.
        self.arg_names = _kernel_arg_names(kernel)
        self.constexprs = getattr(kernel, "constexprs", None)
        self.func = impl
        self.__name__ = getattr(impl, "__name__", "kernel_override")
        self.__module__ = getattr(kernel, "__module__", "")
        self._forward_by_name = (
            tuple(inspect.signature(impl).parameters) == self.arg_names
        )

    def __getitem__(self, grid: Any) -> Callable[..., Any]:
        return self._launch

    def _launch(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            return self._impl(*args, **kwargs)
        if self._forward_by_name:
            return self._impl(**kwargs)
        unexpected = [name for name in kwargs if name not in self.arg_names]
        if unexpected:
            raise RuntimeError(
                f"Override for {self.__name__!r} received unexpected launch "
                f"arguments: {unexpected}"
            )
        return self._impl(*(kwargs[name] for name in self.arg_names if name in kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._impl(*args, **kwargs)


def _resolve_kernel(name: str) -> tuple[Any, str]:
    """Return the (host object, attribute) holding the kernel `name`."""
    module_name, attr = name.rsplit(".", 1)
    parts = module_name.split(".")
    if parts[-1][:1].isupper():
        class_module = importlib.import_module(".".join(parts[:-1]))
        return getattr(class_module, parts[-1]), attr
    importlib.import_module(module_name)
    module = sys.modules[module_name]
    if hasattr(module, attr):
        return module, attr
    raise ValueError(f"Kernel {name!r} not found for override")


def _rebind_kernels(
    overrides: list[tuple[Callable[..., Any], KernelOverride]],
) -> None:
    """Point every reference to each original kernel at its wrapper.

    A kernel object can be captured in two kinds of places besides its
    defining module, and both must be patched for the overrides to take
    effect everywhere:

    * plain attributes of other modules (``from mod import kernel``
      copies), rebound to the wrapper;
    * JIT warmup owners, whose ``kernel`` attribute is the kernel they
      launch; swapped, and their cached ``_kernel_arg_names`` invalidated
      so the launch binding re-derives from the wrapper.

    One pass over ``sys.modules`` handles all kernels in `overrides`.
    """
    originals = {original: wrapper for original, wrapper in overrides}

    def lookup(value: Any) -> KernelOverride | None:
        # Only compare by identity: this scan runs over every module
        # namespace, and some attribute values (e.g. PlaceholderModule
        # sentinels) trigger imports or errors on hash/eq.
        for original in originals:
            if value is original:
                return originals[original]
        return None

    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if not namespace:
            continue
        for attr, value in list(namespace.items()):
            wrapper = lookup(value)
            if wrapper is not None:
                setattr(module, attr, wrapper)
                continue
            # JIT warmup owners (e.g. _DecoratedTritonJitKernel) hold the
            # kernel as an instance attribute; read the instance dict directly
            # so module-level __getattr__ hooks are never triggered, and so
            # this scan needs no dependency on the warmup machinery.
            if hasattr(value, "__dict__"):
                wrapper = lookup(vars(value).get("kernel"))
            if wrapper is None:
                continue
            value.kernel = wrapper
            if not isinstance(value, type):
                value.__dict__.pop("_kernel_arg_names", None)


def register_kernels(overrides: Mapping[str, Callable[..., Any]]) -> None:
    """Register platform-specific implementations for Triton kernels.

    Args:
        overrides: Mapping from fully qualified kernel name to the
            platform implementation. Kernel names look like
            "vllm.v1.sample.rejection_sampler.expand_kernel"; for kernels
            owned by a JIT warmup class, include the attribute path, e.g.
            "vllm.v1.worker.block_table.ComputeSlotMappingKernel.kernel".
            Implementations are invoked like the kernels themselves, with
            the launch arguments in the original kernel's parameter order
            (or by keyword when their parameter names match the kernel's).

    """
    resolved: list[tuple[str, Any, str, Callable[..., Any], KernelOverride]] = []
    for name, impl in overrides.items():
        if name in _registry:
            logger.warning("Kernel %s is already registered. Overwriting.", name)
        host, attr = _resolve_kernel(name)
        original = getattr(host, attr)
        wrapper = KernelOverride(original, impl)
        _registry[name] = (original, impl)
        resolved.append((name, host, attr, original, wrapper))
    # Rebind other holders first: once the defining attributes are swapped
    # below, the scan could no longer find the original kernels.
    _rebind_kernels([(original, wrapper) for *_, original, wrapper in resolved])
    for name, host, attr, _, wrapper in resolved:
        setattr(host, attr, wrapper)
        logger.debug("Registered kernel override %s", name)
