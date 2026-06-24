# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton Kernel Dispatcher for multi-platform support.

This module provides a mechanism to dispatch Triton kernel calls to
platform-specific implementations while maintaining the standard
`kernel[grid](...)` invocation syntax.

Usage:
    1. In vLLM core, define and register the default Triton kernel using decorator,
    take `expand_kernel` in `vllm.v1.sample.rejection_sampler` as an example:
       ```python
       from vllm.model_executor.triton_dispatcher import pluggable_kernel

       @pluggable_kernel
       @triton.jit
       def expand_kernel(...):
           ...
       ```

    2. In an out-of-tree or non-default platform, register a platform-specific
       implementation:
       ```python
       from vllm.model_executor.triton_dispatcher import register_kernel


       @register_kernel("vllm.v1.sample.rejection_sampler.expand_kernel")
       def my_expand_kernel(*args, grid=None, **kwargs):
           # Platform-specific implementation.
           ...
       ```
       or
       ```python
       from vllm.model_executor.triton_dispatcher import register_kernel

       register_kernel("vllm.v1.sample.rejection_sampler.expand_kernel")(
           my_expand_kernel
       )
       ```
"""

from collections.abc import Callable
from typing import Any, Protocol

from vllm.logger import init_logger


class SubscriptableCallable(Protocol):
    """
    Protocol for callables that support subscript notation (e.g., Triton kernels).

    Triton kernels decorated with @triton.jit support `kernel[grid]` syntax
    which returns a launcher object. This protocol represents that behavior.
    """

    __name__: str

    def __getitem__(self, grid: Any) -> Callable: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


logger = init_logger(__name__)

# Global registry: { kernel_name: implementation_fn }
_kernel_registry: dict[str, Callable] = {}


def _get_kernel_impl(name: str) -> Callable | None:
    """
    Retrieve the registered kernel implementation.
    Returns None if no custom implementation is registered (fallback to default).
    """
    return _kernel_registry.get(name)


class _KernelLauncher:
    """
    Internal class that holds the grid configuration and kernel implementation.
    Returned by `KernelDispatcher.__getitem__(grid)` for custom implementations.
    """

    __slots__ = ("_impl", "_grid")

    def __init__(self, impl: Callable, grid: Any):
        self._impl = impl
        self._grid = grid

    def __call__(self, *args, **kwargs):
        """
        Invoke the custom kernel implementation.
        The `grid` is passed as a keyword argument to the implementation.
        """
        return self._impl(*args, grid=self._grid, **kwargs)


class KernelDispatcher:
    """
    A dispatcher that wraps a default Triton kernel and allows
    platform-specific overrides via `register_kernel`.

    Maintains the standard `kernel[grid](...)` syntax for seamless
    integration with existing vLLM code.

    Args:
        name: The unique name of the kernel for registry lookup.
        default_impl: The default Triton kernel implementation (usually CUDA).
    """

    def __init__(self, name: str, default_impl: SubscriptableCallable):
        self.name = name
        self.default_impl = default_impl
        self.__name__ = default_impl.__name__
        self.__module__ = default_impl.__module__

    def __call__(self, *args, **kwargs):
        impl = _get_kernel_impl(self.name)
        if impl is not None:
            return impl(*args, **kwargs)
        return self.default_impl(*args, **kwargs)

    def __getitem__(self, grid):
        """
        Called when using `dispatcher[grid]` syntax.
        Returns a launcher object that can be called with kernel arguments.
        """
        impl = _get_kernel_impl(self.name)
        if impl is not None:
            # Custom implementation registered: wrap in launcher to pass grid as kwarg
            return _KernelLauncher(impl, grid)

        # No custom implementation: fall back to the default Triton kernel.
        # Triton kernels natively support the [grid] syntax, so we return it directly
        # to avoid passing `grid` as a kwarg (which would cause a TypeError).
        return self.default_impl[grid]


def pluggable_kernel(jit_decorated_func: SubscriptableCallable) -> "KernelDispatcher":
    """
    Decorator to automatically register a Triton kernel with the dispatcher.

    This decorator should be applied **after** `@triton.jit`:

        @pluggable_kernel
        @triton.jit
        def my_kernel(...):
            pass

    It automatically extracts the fully qualified name (module + function name)
    to ensure global uniqueness, eliminating the need for manual registration.

    Args:
        jit_decorated_func: The Triton JIT-compiled kernel function.

    Returns:
        A KernelDispatcher instance wrapping the kernel.
    """
    kernel_name = f"{jit_decorated_func.__module__}.{jit_decorated_func.__name__}"
    logger.debug("Auto-registered kernel %s", kernel_name)
    return KernelDispatcher(kernel_name, jit_decorated_func)


def register_kernel(name: str) -> Callable:
    """
    Decorator to register a platform-specific kernel implementation.

    Args:
        name: The fully qualified name of the kernel (e.g.,
            "vllm.v1.worker.mamba_utils.batch_memcpy_kernel").
    """

    def decorator(func: Callable) -> Callable:
        if name in _kernel_registry:
            logger.warning("Kernel %s is already registered. Overwriting.", name)
        _kernel_registry[name] = func
        logger.debug("Registered kernel %s", name)
        return func

    return decorator
