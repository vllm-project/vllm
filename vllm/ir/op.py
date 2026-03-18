# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import inspect
from collections.abc import Callable
from typing import Any, ClassVar, overload

import torch
from torch.library import Library, infer_schema

from vllm.ir.util import hash_source, weak_cache
from vllm.logger import init_logger
from vllm.logging_utils import lazy, tensors_str_no_data

vllm_ir_lib = Library("vllm_ir", "FRAGMENT")

logger = init_logger(__name__)

RESERVED_PROVIDERS = ["native", "unfused"]
"""Providers that are reserved and cannot be used for custom implementations."""

_DIRECT_DISPATCH: bool = False
"""Global override flag to skip the torch op layer."""


@contextlib.contextmanager
def direct_dispatch(direct: bool = True):
    """
    Context manager to set direct dispatch mode for vLLM IR ops.
    When direct dispatch is enabled, the torch custom op layer is skipped
    and IR ops dispatch directly to the implementation.
    Helpful for avoiding torch dispatch overhead in eager mode
    and avoiding the need for lowering for platforms not using Inductor.
    """

    global _DIRECT_DISPATCH
    old = _DIRECT_DISPATCH
    _DIRECT_DISPATCH = direct
    yield
    _DIRECT_DISPATCH = old


# 0-param decorator overload
@overload
def register_op(f: Callable[..., Any]) -> "IrOp": ...


# parametrized decorator overload
@overload
def register_op(
    *,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], "IrOp"]: ...


def register_op(
    f: Callable | None = None,
    *,
    name: str | None = None,
) -> "IrOp | Callable[[Callable], IrOp]":
    """
    Register a new vLLM IR op.

    :param f: the native implementation of the op
    :param name: the name of the op, defaults to the function name
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
        op_name: str = _f.__name__ if name is None else name
        assert op_name not in IrOp.registry
        op = IrOp(op_name, _f)
        IrOp.registry[op_name] = op
        return op

    if f is not None:
        return decorator(f)

    return decorator


class IrOp:
    registry: ClassVar[dict[str, "IrOp"]] = {}

    name: str
    impls: dict[str, "IrOpImpl"]

    def __init__(self, name: str, native_impl: Callable):
        self._py_signature = inspect.signature(native_impl)
        if any(
            p.kind == inspect.Parameter.KEYWORD_ONLY
            for p in self._py_signature.parameters.values()
        ):
            raise ValueError(
                f"Op {name} has keyword-only arguments which are not currently "
                f"supported. That's because kwargs are not allowed during lowering."
            )

        self.name = name
        self.impls: dict[str, IrOpImpl] = {}
        self._priority_impls: list[IrOpImpl] = []
        self._schema_str = infer_schema(native_impl, mutates_args=[])

        # native implementation
        self.impls["native"] = IrOpImpl(
            self, "native", native_impl, supported=True, supports_args=None
        )

        # By default, fake routes directly to native,
        # can be overridden by register_fake
        self._fake_fn = native_impl

        # torch registration
        vllm_ir_lib.define(self.name + self._schema_str)
        # CompositeExplicitAutograd is not decomposed
        # by ATen IR normalization in AOTAutograd
        vllm_ir_lib.impl(
            self.name, self._inner_call, dispatch_key="CompositeExplicitAutograd"
        )
        vllm_ir_lib._register_fake(self.name, self._fake_call)
        assert hasattr(torch.ops.vllm_ir, name)
        self.torch_op: torch._ops.OpOverload = getattr(torch.ops.vllm_ir, name).default

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
        :param supports_args: Dynamic arg support check, used for types and shapes.
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
            impl = IrOpImpl(self, provider, f, supported, supports_args)
            self.impls[provider] = impl

            if self.get_priority():
                logger.warning(
                    "Warning: registering new impl %s for op %s while priority is set.",
                    provider,
                    self.name,
                )

            return impl

        return _register_impl

    def _inner_call(self, *args, **kwargs) -> Any:
        """
        Eager call to torch op lands here. In the future, direct dispatch might
        route straight here instead of going through torch op dispatching.
        """
        impl = self.dispatch(*args, **kwargs)
        return impl.impl_fn(*args, **kwargs)

    def apply_arg_defaults(self, args) -> tuple:
        """
        Return args with default values applied.
        Defaults are taken from the native implementation signature.

        SHOULD NOT BE USED IN THE DISPATCH PATH (SLOW).
        Only for Inductor lowering.
        """
        bound_args = self._py_signature.bind(*args)
        bound_args.apply_defaults()
        return bound_args.args

    def dispatch(self, *args, **kwargs) -> "IrOpImpl":
        """
        Dispatch to the appropriate implementation based on current priority
        and argument support checks. Returns the selected IrOpImpl.

        THIS FUNCTION IS ON THE HOT PATH (OP DISPATCH), MUST BE FAST.
        """
        if not self._priority_impls:
            if not torch.compiler.is_compiling():
                # Logging not compatible with Dynamo tracing
                # (this code is exposed when direct_dispatch is enabled)
                logger.warning_once(
                    "Priority not set for op %s, using native implementation.",
                    self.name,
                )
            return self.impls["native"]

        for impl in self._priority_impls:
            if not impl.supported:
                raise ValueError(
                    f"Implementation {impl.provider} for op {self.name} not supported. "
                    f"All implementations in priority list must be supported."
                )
            if impl.supports_args(*args, **kwargs):
                return impl

            if not torch.compiler.is_compiling():
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
        if _DIRECT_DISPATCH:
            return self._inner_call(*args, **kwargs)

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
                    # Skip unsupported implementations
                    continue

                filtered_impls.append(impl)

                # If all args are supported, skip other implementations
                if impl.supports_all_args:
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

        # Enforce the exact same schema as the native implementation.
        # This takes care of names, types, and defaults.
        schema = infer_schema(impl_fn, mutates_args=[])
        if schema != op._schema_str:
            raise ValueError(
                f"Implementation for provider {provider} has schema '{schema}' which "
                f"does not match native schema '{op._schema_str}' for op {op.name}."
            )

        if supports_args is not None:
            if not callable(supports_args):
                raise ValueError(
                    f"supports_args for provider {provider} must be a callable"
                )

            # We also manually validate the supports_args signature.
            # Matching signatures allow faster dispatch on the hotpath.

            # Check that supports_args does not have keyword-only parameters
            supports_args_signature = inspect.signature(supports_args)
            params = supports_args_signature.parameters
            if any(p.kind == inspect.Parameter.KEYWORD_ONLY for p in params.values()):
                raise ValueError(
                    f"supports_args for provider {provider} "
                    f"cannot have keyword-only parameters"
                )

            # Check that supports_args has the same total number of parameters
            op_params = op._py_signature.parameters
            if len(params) != len(op_params):
                raise ValueError(
                    f"supports_args for provider {provider} must have the same number "
                    f"of parameters ({len(params)}) as the native implementation "
                    f"({len(op_params)})"
                )

            # Check that names and defaults match for supports_args
            for p, op_p in zip(params.values(), op_params.values()):
                if p.name != op_p.name:
                    raise ValueError(
                        f"supports_args for provider {provider} has parameter "
                        f"'{p.name}' which does not match native parameter "
                        f"'{op_p.name}'"
                    )
                if p.default != op_p.default:
                    raise ValueError(
                        f"supports_args for provider {provider} has parameter "
                        f"'{p.name}' with default {p.default} which does not match "
                        f"native default {op_p.default}'"
                    )

        self.op = op
        self.provider = provider
        self.impl_fn = impl_fn
        self.supported = supported
        self._supports_args = supports_args

    @property
    def supports_all_args(self) -> bool:
        """Check if this implementation supports all args unconditionally."""
        return self._supports_args is None

    def supports_args(self, *args, **kwargs) -> bool:
        if self._supports_args is None:
            return True

        return self._supports_args(*args, **kwargs)

    @weak_cache
    def uuid(self):
        """
        Compile-time hash to uniquely determine whether the implementation has changed.
        Used by vllm-compile hash mechanism and torch.compile lowering pass uuid to
        control the vLLM compile cache and AOTAutograd/Inductor caches respectively.

        impl_fn and _supports_args should not change after construction, so we cache it.
        """
        return hash_source(self.impl_fn, self._supports_args)
