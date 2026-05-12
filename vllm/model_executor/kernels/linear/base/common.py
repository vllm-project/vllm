# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import torch
from torch.library import Library

# Each Composite registers a dispatch op under torch.ops.composed_kernel.
composed_kernel_lib = Library("composed_kernel", "FRAGMENT")  # noqa


@dataclass
class Config:
    """Base config; subclasses (e.g. ``base/w16a16.py``) add scheme-specific fields."""


ConfigT = TypeVar("ConfigT", bound=Config)


class Kernel(ABC, Generic[ConfigT]):
    """A linear GEMM implementation. Subclasses pick one hardware/algorithm
    path and stay fixed for the life of the layer."""

    config: ConfigT

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    @classmethod
    def get_name(cls) -> str:
        module = cls.__module__.removeprefix("vllm.model_executor.kernels.")
        return f"{module}.{cls.__name__}"

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]: ...

    @classmethod
    def can_implement(cls, config: ConfigT) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    @abstractmethod
    def apply_weights(self, *args, **kwargs) -> torch.Tensor:
        """Run the kernel for the layer. Delegate to ``type(self).apply`` so
        ``PredicateKernel`` subclasses can override ``apply`` without
        overriding this method."""
        ...

    @staticmethod
    @abstractmethod
    def apply(*args, **kwargs) -> torch.Tensor:
        """Pure-function GEMM. Static so the signature is inspectable at
        class-definition time (used for op registration / tracing)."""
        ...


class PredicateKernel(Kernel[ConfigT]):
    """Kernel with a predicate. Override ``predicate`` with a check matching
    ``apply``'s args."""

    @staticmethod
    @abstractmethod
    def predicate(*args, **kwargs) -> bool: ...


def _hash_chain_name(kernel_names: list[str]) -> str:
    """Hash an ordered list of kernel identity strings to a short 8-hex digest."""
    return hashlib.sha1(".".join(kernel_names).encode()).hexdigest()[:8]


class Composite(Kernel[ConfigT]):
    """Dispatching Kernel built from a chain of inner kernels.

    Subclasses set:
        _scheme_tag: op-name prefix
        _chain: PredicateKernels, optionally ending in a plain Kernel terminal
        _dispatcher_fn: wraps (predicate, primary, fallback) into a closure
        _native_impl: safe-default impl; runtime fallback and fake_impl for tracing
    """

    _scheme_tag: ClassVar[str]
    _chain: ClassVar[list[type[Kernel]]]
    _dispatcher_fn: ClassVar[Callable]
    _native_impl: ClassVar[Callable]

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        if any(k.is_supported(compute_capability)[0] for k in cls._chain):
            return True, None
        return False, "no inner kernel supported on this platform"

    @classmethod
    def can_implement(cls, config: ConfigT) -> tuple[bool, str | None]:
        for k in cls._chain:
            if k.is_supported()[0] and k.can_implement(config)[0]:
                return True, None
        return False, "no inner kernel viable for this config"

    def __init__(self, config: ConfigT) -> None:
        super().__init__(config)
        last = self._chain[-1]
        if issubclass(last, PredicateKernel):
            predicated, terminal_apply = self._chain, self._native_impl
        else:
            predicated = self._chain[:-1]
            if last.is_supported()[0] and last.can_implement(config)[0]:
                terminal_apply = last.apply
            else:
                terminal_apply = self._native_impl

        viable = [
            k for k in predicated if k.is_supported()[0] and k.can_implement(config)[0]
        ]

        dispatch_fn = terminal_apply
        for primary in reversed(viable):
            if not issubclass(primary, PredicateKernel):
                raise TypeError(
                    f"{primary.__name__} must be a PredicateKernel; only the last "
                    f"entry of {type(self).__name__}._chain may be a plain Kernel."
                )
            dispatch_fn = self._dispatcher_fn(
                primary.predicate,
                primary,
                dispatch_fn,
            )

        kernel_names = [k.get_name() for k in viable]
        kernel_names.append(
            f"{terminal_apply.__module__}.{terminal_apply.__qualname__}"
        )
        self._op_name = f"{self._scheme_tag}_{_hash_chain_name(kernel_names)}"
        self._dispatch_fn = dispatch_fn

        # Skip re-registration if this op is already in the library.
        if not hasattr(torch.ops.composed_kernel, self._op_name):
            from vllm.utils.torch_utils import direct_register_custom_op

            direct_register_custom_op(
                self._op_name,
                dispatch_fn,
                fake_impl=self._native_impl,
                target_lib=composed_kernel_lib,
            )

        # Cache to avoid getattr on every forward.
        self._op = getattr(torch.ops.composed_kernel, self._op_name)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._op(x, layer.processed_weight, bias)

    @staticmethod
    def apply(*args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
