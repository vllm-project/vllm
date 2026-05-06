# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, TypeVar

import torch


class Config(ABC):
    """Abstract base class for kernel layer configurations.

    Scheme-specific bases (e.g. ``base/w16a16.py``) subclass this to define
    the concrete fields required by their kernels. Passing a typed
    ``Config`` subclass through ``Kernel.__init__`` and ``can_implement``
    ensures that configuration validation is scheme-aware at the call site.
    """
    ...


class Kernel(ABC):
    """Abstract base class defining the interface contract for all linear GEMM kernels.

    A ``Kernel`` encapsulates a single hardware- or algorithm-specific
    implementation of a linear projection. Concrete subclasses are selected
    at layer-initialisation time based on platform support and weight
    configuration, and are expected to remain fixed for the lifetime of the
    layer.

    The class is structured around two complementary entry points,
    ``apply`` and ``apply_weights``, whose separation is load-bearing for
    compile-time correctness — see their respective docstrings.
    """

    def __init__(self, config: Config) -> None:
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
    def can_implement(cls, config: Config) -> tuple[bool, str | None]:
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


K = TypeVar('K', bound=Kernel)


def _resolve_dispatch_fn(cls: type[K], config: Config) -> Callable:
    """Recursively walk the fallback chain to find a supported dispatch callable.

    Starting from ``cls``, checks ``is_supported`` and ``can_implement`` in
    turn. If either fails, recurses into ``cls._fallback``. If the chain is
    exhausted, returns ``cls.apply`` directly as the terminal callable.
    Otherwise instantiates ``cls`` and returns its ``_dispatch_fn`` if present
    (set by ``make_predicated``), falling back to ``cls.apply``.
    """
    ok, _ = cls.is_supported()
    if ok:
        ok, _ = cls.can_implement(config)
    if not ok:
        fallback = getattr(cls, '_fallback', None)
        if fallback is not None:
            return _resolve_dispatch_fn(fallback, config)
        return cls.apply
    inst = cls(config)
    return getattr(inst, '_dispatch_fn', cls.apply)


def make_predicated(
    name: str,
    predicate: Callable,
    fallback: type[K],
    dispatcher_fn: Callable,
    fake_impl: Callable,
) -> Callable[[type[K]], type[K]]:
    """Decorator factory that wraps a kernel class with predicated dispatch.

    Returns a decorator that, when applied to a kernel class ``primary``,
    produces a ``_PredicatedKernel`` subclass. At layer-initialisation time
    the subclass resolves the full fallback chain into a single flat dispatch
    callable and registers it as a ``torch.ops.vllm`` custom op. Subsequent
    calls to ``apply`` are routed through this op, eliminating Python-level
    conditionals that would otherwise cause graph breaks under
    ``torch.compile`` / CUDA graph capture.

    Args:
        name: The op name to register under ``torch.ops.vllm``.
        predicate: A callable that returns ``True`` when ``primary`` should
            be invoked. Forwarded to ``dispatcher_fn``.
        fallback: The kernel class to fall back to when the predicate is
            ``False``. May itself be a predicated kernel, in which case the
            chain is resolved recursively and collapsed into a single op.
        dispatcher_fn: A scheme-specific factory that, given
            ``(predicate, primary, fallback_fn)``, returns the typed dispatch
            closure to register. Must carry concrete type annotations so that
            ``infer_schema`` can derive the ATen op schema.
        fake_impl: A callable used as the abstract interpretation of the op
            during tracing (e.g. ``torch.nn.functional.linear`` for w16a16).
    """
    def decorator(primary: type[K]) -> type[K]:
        if primary.apply is fake_impl:
            raise TypeError(
                f"{primary.__qualname__} does not override 'apply'. "
                f"Kernels wrapped with make_predicated must define a static "
                f"'apply' method."
            )

        class _PredicatedKernel(primary):
            _registered: ClassVar[bool] = False
            _fallback: ClassVar[type] = fallback

            def __init__(self, config: Config) -> None:
                fallback_fn = _resolve_dispatch_fn(fallback, config)
                dispatch_fn = dispatcher_fn(predicate, primary, fallback_fn)
                self._dispatch_fn = dispatch_fn
                if not _PredicatedKernel._registered:
                    from vllm.utils.torch_utils import direct_register_custom_op
                    direct_register_custom_op(
                        name, dispatch_fn, fake_impl=fake_impl,
                    )
                    _PredicatedKernel._registered = True

            @staticmethod
            def apply(*args, **kwargs) -> torch.Tensor:
                return getattr(torch.ops.vllm, name)(*args, **kwargs)

        _PredicatedKernel.__name__ = primary.__name__
        _PredicatedKernel.__qualname__ = primary.__qualname__
        _PredicatedKernel.__module__ = primary.__module__
        return _PredicatedKernel

    return decorator
