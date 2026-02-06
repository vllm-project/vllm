# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Callable
from typing import Any, ClassVar, overload

import torch
from torch.library import Library, infer_schema

from vllm.logger import init_logger
from vllm.logging_utils import lazy, tensors_str_no_data

vllm_ir_lib = Library("vllm_ir", "FRAGMENT")

logger = init_logger(__name__)

RESERVED_PROVIDERS = ["native", "unfused"]
"""Providers that are reserved and cannot be used for custom implementations."""


# 0-param decorator overload
@overload
def register_op(f: Callable[..., Any]) -> "IrOp": ...


# parametrized decorator overload
@overload
def register_op(
    *,
    name: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
) -> Callable[[Callable[..., Any]], "IrOp"]: ...


def register_op(
    f: Callable | None = None,
    *,
    name: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
) -> "IrOp | Callable[[Callable], IrOp]":
    """
    Register a new vLLM IR op.

    :param f: the native implementation of the op
    :param name: the name of the op, defaults to the function name
    :param tags: any additional torch tags for the op
    :return: the IrOp object if f is provided, otherwise a decorator

    Example usage:
    ```python
    @vllm.ir.register_op
    def my_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


    @vllm.ir.register_op(name="custom_mul")
    def multiply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y"""

    def decorator(_f: Callable):
        op_name = _f.__name__ if name is None else name
        return IrOp(op_name, _f, tags)

    if f is not None:
        return decorator(f)

    return decorator


class IrOp:
    registry: ClassVar[dict[str, "IrOp"]] = {}

    def __init__(
        self, name: str, native_impl: Callable, tags: tuple[torch.Tag, ...] = ()
    ):
        self.name = name
        self.impls: dict[str, IrOpImpl] = {}
        self._priority_impls: list[IrOpImpl] = []
        self._schema_str = infer_schema(native_impl, mutates_args=[])
        self.native_fn = native_impl

        # native implementation, constructor also registers into impls
        self._native_impl = IrOpImpl(
            self, "native", native_impl, supported=True, supports_args=None
        )

        self._fake_fn = native_impl

        # torch registration
        vllm_ir_lib.define(self.name + self._schema_str, tags=tags)
        vllm_ir_lib.impl(self.name, self._inner_call, dispatch_key="CUDA")
        vllm_ir_lib.impl(self.name, self._inner_call, dispatch_key="CPU")
        vllm_ir_lib._register_fake(self.name, self._fake_call)
        assert hasattr(torch.ops.vllm_ir, name)
        self.torch_op = getattr(torch.ops.vllm_ir, name).default

        assert name not in self.registry
        self.registry[name] = self

    def register_fake(self, fn: Callable) -> Callable:
        """
        Register a fake impl for the torch custom op. If this method is not called,
        the native implementation is used directly for the fake implementation.
        """
        self._fake_fn = fn
        return fn

    def _fake_call(self, *args, **kwargs) -> Any:
        """
        Call to the fake implementation of the op. We use indirection because we want
        users to be able to register fake later but also want it to fall back to native
        directly by default, instead of going through the dispatching mechanism.
        """
        return self._fake_fn(*args, **kwargs)

    def register_impl(
        self,
        provider: str,
        *,
        supported: bool = True,
        supports_args: Callable[..., bool] | None = None,
    ):
        """
        Register an implementation for this custom op.
        :param provider: The name of the provider, must be unique.
        :param supported: Static support check, use this to check platform support.
        :param supports_args: Dynamic arg support check.
        :return: A decorator that registers the implementation.

        The decorated function must have the same semantics and signature as
        the native implementation.

        The provider name must be unique and not one of the RESERVED_PROVIDERS.
        The supported and supports_args parameters should not be used to implement
        custom enablement logic based on global state (e.g. environment variables).
        Instead, supported param should only be used to check for platform support
        (e.g. whether a specific hardware or library is available).
        supports_args should be used to check whether the provided arguments are
        compatible with the implementation.
        For custom enablement logic, set op impl priority.

        Example:
        ```python
        @my_op.register_impl("my_provider", supported=torch.cuda.is_available())
        def my_provider_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
        ```

        """
        assert provider not in RESERVED_PROVIDERS, (
            f"Provider name {provider} is reserved."
        )

        def _register_impl(f: Callable):
            return IrOpImpl(self, provider, f, supported, supports_args)

        return _register_impl

    def _inner_call(self, *args, **kwargs) -> Any:
        """Direct call to torch op, could also skip the torch layer if eager?"""
        impl = self.dispatch(*args, **kwargs)
        return impl.impl_fn(*args, **kwargs)

    def dispatch(self, *args, **kwargs) -> "IrOpImpl":
        """
        Dispatch to the appropriate implementation based on current priority
        and argument support checks. Returns the selected IrOpImpl.
        """
        if not self._priority_impls:
            logger.warning_once(
                "Priority not set for op %s, using native implementation.", self.name
            )
            return self.impls["native"]

        for impl in self._priority_impls:
            assert impl.supported, (
                "All implementations in priority list must be supported."
            )
            if impl.supports_args is None or impl.supports_args(*args, **kwargs):
                return impl

            logger.debug(
                "Skipping provider %s because it does not support "
                "%s with args=%s kwargs=%s",
                impl.provider,
                self.name,
                lazy(lambda: tensors_str_no_data(args)),
                lazy(lambda: tensors_str_no_data(kwargs)),
            )

        raise RuntimeError(
            "Priority set incorrectly: the last implementation must "
            "support all args (can be native). This is likely an internal bug"
        )

    def __call__(self, *args, **kwargs) -> Any:
        return self.torch_op(*args, **kwargs)

    def get_priority(self) -> list[str]:
        """Get the current dispatch priority for implementations for this op."""
        return [p.provider for p in self._priority_impls]

    @contextlib.contextmanager
    def set_priority(self, priority: list[str]):
        """
        Context manager to set the dispatch priority for implementations for this op.
        """
        assert all(p in self.impls for p in priority), (
            "All providers in priority must be registered implementations."
        )

        def filter_priority_impls(p_list: list[str]) -> list[IrOpImpl]:
            filtered_impls = []
            for p in p_list:
                impl = self.impls[p]
                if not impl.supported:
                    continue

                # Skip unsupported implementations
                filtered_impls.append(impl)

                # If all args are supported, skip other implementations
                if impl.supports_args is None:
                    return filtered_impls

            logger.warning_once(
                "Op %s: No implementation in priority list supports all args, "
                "execution fallback to native is possible. To silence this warning, "
                "explicitly add 'native' to the end of the priority list",
                self.name,
            )
            filtered_impls.append(self.impls["native"])
            return filtered_impls

        # Temporarily set priority
        old_priority_impls = self._priority_impls
        try:
            self._priority_impls = filter_priority_impls(priority)
            yield
        finally:
            self._priority_impls = old_priority_impls

    def supported_providers(self) -> list[str]:
        return [p.provider for p in self.impls.values() if p.supported]


class IrOpImpl:
    def __init__(
        self,
        op: IrOp,
        provider: str,
        impl_fn: Callable,
        supported: bool,
        supports_args: Callable[..., bool] | None,
    ):
        assert provider not in op.impls, (
            f"Implementation for provider {provider} already registered."
        )
        # Native also uses this path, so we allow it here.
        assert provider == "native" or provider not in RESERVED_PROVIDERS

        self.op = op
        self.provider = provider
        self.impl_fn = impl_fn
        self.supported = supported
        self.supports_args = supports_args

        op.impls[provider] = self

        if op.get_priority():
            logger.warning(
                "Warning: registering new impl %s for op %s while priority is set.",
                provider,
                op.name,
            )
