# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

import torch
from torch.library import Library

# Library for composed-kernel ops. Each Composite subclass registers its
# per-layer dispatch op under torch.ops.composed_kernel.<op_name>.
composed_kernel_lib = Library("composed_kernel", "FRAGMENT")  # noqa


@dataclass
class Config:
    """Base class for kernel layer configurations.

    Scheme-specific bases (e.g. ``base/w16a16.py``) subclass this to define
    the concrete fields required by their kernels. Passing a typed
    ``Config`` subclass through ``Kernel.__init__`` and ``can_implement``
    ensures that configuration validation is scheme-aware at the call site.
    """


ConfigT = TypeVar("ConfigT", bound=Config)


class Kernel(ABC, Generic[ConfigT]):
    """Abstract base class defining the interface contract for all linear GEMM kernels.

    A ``Kernel`` encapsulates a single hardware- or algorithm-specific
    implementation of a linear projection. Concrete subclasses are selected
    at layer-initialisation time based on platform support and weight
    configuration, and are expected to remain fixed for the lifetime of the
    layer.

    The class is structured around two complementary entry points,
    ``apply`` and ``apply_weights``, whose separation is load-bearing for
    compile-time correctness.
    """

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

    def _get_layer_params(self, layer: torch.nn.Module) -> Any:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    @abstractmethod
    def apply_weights(self, *args, **kwargs) -> torch.Tensor:
        """Execute the kernel for the given layer and inputs.

        This is the primary runtime entry point. Implementations must extract
        the relevant parameters from ``layer`` via ``_get_layer_params`` and
        delegate to ``type(self).apply``. Routing through ``type(self)``
        rather than a direct call to ``apply`` is essential: it ensures that a
        subclass override of ``apply`` is honoured without requiring any
        corresponding override of ``apply_weights`` in the predicated subclass.

        Scheme-specific bases narrow the signature to the concrete arguments
        required by their calling convention.
        """
        ...

    @staticmethod
    @abstractmethod
    def apply(*args, **kwargs) -> torch.Tensor:
        """Define the pure tensor semantics of this kernel.

        ``apply`` is a *static* method so that its signature is fully
        inspectable at class-definition time, independent of any instance
        state. ``make_predicated`` overrides ``apply`` on the generated
        ``_PredicatedKernel`` subclass to redirect calls through the registered
        op, thereby avoiding Python-level conditionals that would otherwise
        cause graph breaks under ``torch.compile`` and CUDA graph capture.

        Scheme-specific bases narrow the signature and provide a safe default
        implementation (e.g. ``torch.nn.functional.linear`` for w16a16) that
        serves as the terminal fallback in the dispatch chain.
        """
        ...


class PredicateKernel(Kernel[ConfigT]):
    """A Kernel that participates in a predicated chain.

    Subclasses must override ``predicate`` with  check
    whose signature mirrors ``apply`` for the scheme.
    """

    @staticmethod
    @abstractmethod
    def predicate(*args, **kwargs) -> bool: ...


def _hash_chain_name(kernel_names: list[str]) -> str:
    """Hash an ordered list of kernel identity strings to a short 8-hex
    digest."""
    return hashlib.sha1(".".join(kernel_names).encode()).hexdigest()[:8]


class Composite(Kernel[ConfigT]):
    """Generic dispatching Kernel built from a list of inner kernels.

    Subclasses declare:
        _scheme_tag    : str — short tag baked into the registered op name
        _chain         : list[type[Kernel]] — PredicateKernels followed by an
                         optional plain-Kernel terminal candidate
        _dispatcher_fn : Callable — scheme-specific dispatch factory; takes
                         (predicate, primary, fallback_fn) and returns a
                         typed closure suitable for direct_register_custom_op
        _native_impl   : Callable — scheme-specific safe-default impl. Used as
                         (a) the runtime safety net when no PredicateKernel
                         matches and the plain terminal candidate (if any)
                         isn't viable for this config, and (b) the fake_impl
                         passed to direct_register_custom_op for tracing.
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

        # native_impl is always a safe runtime fallback, so dispatch_fn is
        # guaranteed non-None even if no PredicateKernel matches at apply time.
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

        # Cache the op handle so apply_weights doesn't re-resolve via getattr
        # on every forward.
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
