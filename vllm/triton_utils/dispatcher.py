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

The same call covers Triton *device functions* (``@triton.jit`` helpers
called from inside other kernels and inlined at compile time): an
implementation that is itself a Triton kernel is swapped in directly, so
outer kernels keep compiling and launchable kernels run natively. Only
non-Triton implementations (plain callables, grid-remapping launchers)
are wrapped in a launch adapter.
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


def _is_triton_kernel(obj: Any) -> bool:
    """Whether ``obj`` is a real Triton kernel (JITFunction).

    Triton kernels (both launchable kernels and device functions called
    from inside other kernels) expose the JIT launch protocol; plain
    callables and grid-remapping launcher objects do not.
    """
    return callable(obj) and all(
        hasattr(obj, name) for name in ("__getitem__", "run", "warmup", "arg_names")
    )


class KernelOverride:
    """Stand in for a Triton kernel, routing launches to a non-Triton ``impl``.

    Used by :func:`register_kernels` for implementations that are not
    themselves Triton kernels (plain callables, grid-remapping launchers);
    Triton kernel implementations are swapped in directly and never wrapped.

    The JIT warmup infrastructure launches kernels as
    ``kernel[grid](**kwargs)`` where ``kwargs`` are keyed by the original
    kernel's argument names. This wrapper forwards those launch arguments
    to the platform implementation: by keyword when the implementation's
    parameter names match the kernel's, otherwise positionally in the
    kernel's parameter order. An implementation that reimplements the
    ``kernel[grid]`` protocol (a grid-remapping launcher) receives the
    caller's grid. Compile-only JIT warmup (``kernel.warmup(grid,
    **kwargs)``) is delegated to the implementation's own ``warmup`` when
    present, and is a no-op otherwise.
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
        try:
            impl_names = _kernel_arg_names(impl)
        except (TypeError, ValueError):
            # Kernel-like implementations (e.g. a grid-rewriting launcher)
            # are not introspectable; their launch arguments are forwarded
            # positionally in the original kernel's parameter order.
            impl_names = ()
        self._forward_by_name = impl_names == self.arg_names

    def __getitem__(self, grid: Any) -> Callable[..., Any]:
        """Bind a launch grid, mirroring Triton's ``kernel[grid]``.

        A platform override that is itself a Triton kernel (or any object
        implementing ``__getitem__``) takes over the launch natively:
        ``impl[grid](*args, **kwargs)`` binds against the override's own
        signature, so the caller's grid reaches the override unchanged.
        Such an override must mirror the original kernel's parameter names
        (or order), since launch sites key arguments by the original names.
        Plain-callable overrides are invoked with the launch arguments
        directly.
        """
        impl_getitem = getattr(self._impl, "__getitem__", None)
        if impl_getitem is not None:
            return impl_getitem(grid)
        return self._launch

    def warmup(self, grid: Any = None, **kwargs: Any) -> Any:
        """Delegate compile-only warmup to the implementation.

        The JIT warmup machinery compiles warmup-owned kernels as
        ``kernel.warmup(grid, **kwargs)``. Forward that to the
        implementation's own kernel so the override is what gets compiled.
        Implementations without a warmable Triton kernel (plain callables,
        grid-remapping launchers) have nothing to warm up and compile
        lazily on first launch.
        """
        impl_warmup = getattr(self._impl, "warmup", None)
        if impl_warmup is None:
            return None
        return impl_warmup(grid=grid, **kwargs)

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


def _rebind_impls(
    replacements: list[tuple[Callable[..., Any], Callable[..., Any]]],
) -> None:
    """Point every reference to each original object at its replacement.

    A kernel or device-function object can be captured in two kinds of
    places besides its defining module, and both must be patched for the
    overrides to take effect everywhere:

    * plain attributes of other modules (``from mod import kernel``
      copies), rebound to the replacement;
    * JIT warmup owners, whose ``kernel`` attribute is the kernel they
      launch; swapped, and their cached ``_kernel_arg_names`` invalidated
      so the launch binding re-derives from the replacement.

    One pass over ``sys.modules`` handles all objects in `replacements`.
    """
    replacements_by_original = {
        original: replacement for original, replacement in replacements
    }

    def lookup(value: Any) -> Callable[..., Any] | None:
        # Only compare by identity: this scan runs over every module
        # namespace, and some attribute values (e.g. PlaceholderModule
        # sentinels) trigger imports or errors on hash/eq.
        for original in replacements_by_original:
            if value is original:
                return replacements_by_original[original]
        return None

    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if not namespace:
            continue
        for attr, value in list(namespace.items()):
            replacement = lookup(value)
            if replacement is not None:
                setattr(module, attr, replacement)
                continue
            # JIT warmup owners (e.g. _DecoratedTritonJitKernel) hold the
            # kernel as an instance attribute; read the instance dict directly
            # so module-level __getattr__ hooks are never triggered, and so
            # this scan needs no dependency on the warmup machinery.
            if hasattr(value, "__dict__"):
                replacement = lookup(vars(value).get("kernel"))
            if replacement is None:
                continue
            value.kernel = replacement
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

            An implementation that is itself a Triton kernel (also the
            only kind that works for Triton *device functions*, which are
            inlined into other kernels at compile time) is swapped in
            directly and launches natively: ``impl[grid](*args, **kwargs)``
            binds against the implementation's own signature and the
            caller's grid reaches it unchanged. Such an implementation must
            mirror the original kernel's parameter names (or order).

            Any other implementation (a plain callable, or a launcher that
            rewrites the grid for a non-CUDA device) is invoked with the
            launch arguments in the original kernel's parameter order (or
            by keyword when its parameter names match the kernel's).

    """
    resolved: list[tuple[str, Any, str, Callable[..., Any], Callable[..., Any]]] = []
    for name, impl in overrides.items():
        if name in _registry:
            logger.warning("Kernel %s is already registered. Overwriting.", name)
        host, attr = _resolve_kernel(name)
        original = getattr(host, attr)
        # Triton kernel implementations are swapped in raw: they launch
        # natively at call sites and keep compiling when an outer kernel
        # inlines them as device functions. Wrapping such an object in
        # KernelOverride would break the outer kernel's compilation.
        replacement = (
            impl if _is_triton_kernel(impl) else KernelOverride(original, impl)
        )
        _registry[name] = (original, impl)
        resolved.append((name, host, attr, original, replacement))
    # Rebind other holders first: once the defining attributes are swapped
    # below, the scan could no longer find the original kernels.
    _rebind_impls([(original, replacement) for *_, original, replacement in resolved])
    for name, host, attr, _, replacement in resolved:
        setattr(host, attr, replacement)
        logger.debug("Registered kernel override %s", name)
