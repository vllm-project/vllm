# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import inspect
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
    activations: list[str] | None = None,
    allow_inplace: bool = False,
) -> Callable[[Callable[..., Any]], "IrOp"]: ...


def register_op(
    f: Callable | None = None,
    *,
    name: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
    activations: list[str] | None = None,
    allow_inplace: bool = False,
) -> "IrOp | Callable[[Callable], IrOp]":
    """
    Register a new vLLM IR op.

    :param f: the native implementation of the op
    :param name: the name of the op, defaults to the function name
    :param tags: any additional torch tags for the op
    :param activations: list of activation params, defaults to params starting with 'x'
    :param allow_inplace: add a maybe_inplace overload that allows inplace impls
    :return: the IrOp object if f is provided, otherwise a decorator

    Example usage:
    ```python
    @vllm.ir.register_op
    def my_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


    @vllm.ir.register_op(name="custom_mul")
    def multiply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y"""

    def decorator(_f: Callable):
        op_name = _f.__name__ if name is None else name
        return IrOp(op_name, _f, tags, activations, allow_inplace)

    if f is not None:
        return decorator(f)

    return decorator


class IrOp:
    registry: ClassVar[dict[str, "IrOp"]] = {}

    def __init__(
        self,
        name: str,
        native_impl: Callable,
        tags: tuple[torch.Tag, ...] = (),
        activations: list[str] | None = None,
        allow_inplace: bool = False,
    ):
        signature = inspect.signature(native_impl)
        if any(
            p.kind == inspect.Parameter.KEYWORD_ONLY
            for p in signature.parameters.values()
        ):
            raise ValueError(
                f"Op {name} has keyword-only arguments which are not currently "
                f"supported. That's because kwargs are not allowed during lowering."
            )

        # By convention, we consider parameters starting with 'x' as activations.
        if activations is None:
            activations = [
                p.name for p in signature.parameters.values() if p.name.startswith("x")
            ]

        self.name = name
        self.activations = activations
        self.activation_indices = [
            i
            for i, p in enumerate(signature.parameters.values())
            if p.name in activations
        ]
        self.impls: dict[str, IrOpImpl] = {}
        self._priority_impls: list[IrOpImpl] = []
        self._schema_str = infer_schema(native_impl, mutates_args=[])
        self.native_fn = native_impl
        self.allow_inplace = allow_inplace

        # native implementation, constructor also registers into impls
        self._native_impl = IrOpImpl(
            self, "native", native_impl, supported=True, supports_args=None
        )

        self._fake_fn = native_impl
        self._signature = signature

        # torch registration
        vllm_ir_lib.define(self.name + self._schema_str, tags=tags)
        vllm_ir_lib.impl(self.name, self._inner_call, dispatch_key="CUDA")
        vllm_ir_lib.impl(self.name, self._inner_call, dispatch_key="CPU")
        vllm_ir_lib._register_fake(self.name, self._fake_call)
        assert hasattr(torch.ops.vllm_ir, name)
        self.torch_op = getattr(torch.ops.vllm_ir, name).default

        if self.allow_inplace:
            self.maybe_inplace = IrOpInplaceOverload(self)

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
        inplace: bool = False,
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
            return IrOpImpl(self, provider, f, supported, supports_args, inplace)

        return _register_impl

    def _inner_call(self, *args, **kwargs) -> Any:
        """Direct call to torch op, TODO could also skip the torch layer if eager?"""
        impl = self.dispatch(*args, **kwargs)

        # Default overload must be functional,
        # use safe_impl_fn to correctly handle inplace impls.
        return impl.safe_impl_fn(*args, **kwargs)

    def apply_arg_defaults(self, *args, **kwargs) -> tuple[tuple, dict]:
        """
        Return (args, kwargs) with default values applied.
        Defaults are taken from the native implementation signature.
        """
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return bound_args.args, bound_args.kwargs

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

        args, kwargs = self.apply_arg_defaults(*args, **kwargs)
        for impl in self._priority_impls:
            assert impl.supported, (
                "All implementations in priority list must be supported."
            )
            if impl.supports_args(*args, **kwargs):
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


class IrOpInplaceOverload:
    def __init__(self, op: IrOp):
        params, returns = op._schema_str.split(" -> ")
        n_outputs = returns.count("Tensor")

        assert returns.count("Tensor") == len(op.activations), (
            "Inplace overload requires the same number of outputs as activations."
        )

        assert returns.count(",") == n_outputs - 1, (
            "Inplace overload only supports Tensor outputs for now."
        )

        self.op = op
        self.name = f"{op.name}.maybe_inplace"
        self._schema_str = infer_schema(op.native_fn, mutates_args=op.activations)

        # torch registration
        vllm_ir_lib.define(self.name + self._schema_str)
        vllm_ir_lib.impl(self.name, self._inner_call, dispatch_key="CUDA")
        vllm_ir_lib.impl(self.name, self._inner_call, dispatch_key="CPU")
        # fake goes to default overload for now
        vllm_ir_lib._register_fake(self.name, self.op._fake_call)
        assert hasattr(getattr(torch.ops.vllm_ir, self.op.name), "maybe_inplace")
        self.torch_op = getattr(torch.ops.vllm_ir, self.op.name).maybe_inplace

    def __call__(self, *args, **kwargs) -> Any:
        return self.torch_op(*args, **kwargs)

    def _inner_call(self, *args, **kwargs) -> Any:
        # Calling the maybe_inplace overload means we can use inplace impls directly.
        impl = self.op.dispatch(*args, **kwargs)
        return impl.impl_fn(*args, **kwargs)


class IrOpImpl:
    def __init__(
        self,
        op: IrOp,
        provider: str,
        impl_fn: Callable,
        supported: bool,
        supports_args: Callable[..., bool] | None,
        inplace: bool = False,
    ):
        assert provider not in op.impls, (
            f"Implementation for provider {provider} already registered."
        )
        # Native also uses this path, so we allow it here.
        assert provider == "native" or provider not in RESERVED_PROVIDERS

        # check schema matches native impl
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
            # Check that supports_args does not have keyword-only parameters
            supports_args_signature = inspect.signature(supports_args)
            params = supports_args_signature.parameters
            if any(p.kind == inspect.Parameter.KEYWORD_ONLY for p in params.values()):
                raise ValueError(
                    f"supports_args for provider {provider} "
                    f"cannot have keyword-only parameters"
                )

            # Check that supports_args has the same total number of parameters
            if len(params) != len(op._signature.parameters):
                raise ValueError(
                    f"supports_args for provider {provider} must have the same number "
                    f"of parameters ({len(params)}) as the native implementation "
                    f"({len(op._signature.parameters)})"
                )

        if inplace:
            assert op.allow_inplace, (
                f"Inplace implementation cannot be registered for op {op.name}"
                f" that does not allow inplace."
            )

        self.op = op
        self.provider = provider
        self.impl_fn = impl_fn
        self.supported = supported
        self._supports_args = supports_args
        self.inplace = inplace

        op.impls[provider] = self

        if op.get_priority():
            logger.warning(
                "Warning: registering new impl %s for op %s while priority is set.",
                provider,
                op.name,
            )

    @property
    def supports_all_args(self) -> bool:
        """Check if this implementation supports all args unconditionally."""
        return self._supports_args is None

    def supports_args(self, *args, **kwargs) -> bool:
        if self._supports_args is None:
            return True

        args, kwargs = self.op.apply_arg_defaults(*args, **kwargs)
        return self._supports_args(*args, **kwargs)

    def safe_impl_fn(self, *args, **kwargs) -> Any:
        """
        Copy any inputs in activations if this is an inplace impl,
        to ensure functional semantics.
        """
        if not self.inplace:
            return self.impl_fn(*args, **kwargs)

        # copy activations to ensure functional semantics
        new_args = list(args)
        for i in self.op.activation_indices:
            assert isinstance(args[i], torch.Tensor)
            new_args[i] = args[i].clone()

        return self.impl_fn(*new_args, **kwargs)
