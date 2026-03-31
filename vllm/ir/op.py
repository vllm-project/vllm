# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import inspect
from collections.abc import Callable
from pathlib import Path
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

_ENABLE_TORCH_WRAP: bool = True
"""Global override flag to control torch op layer wrapping."""


@contextlib.contextmanager
def enable_torch_wrap(enable: bool = True):
    """
    Context manager to enable/disable torch custom op wrapping for vLLM IR ops.
    When torch wrapping is disabled, the torch custom op layer is skipped
    and IR ops dispatch directly to the implementation.
    Helpful for avoiding torch dispatch overhead in eager mode
    and avoiding the need for lowering for platforms not using Inductor.
    """

    global _ENABLE_TORCH_WRAP
    old = _ENABLE_TORCH_WRAP
    try:
        _ENABLE_TORCH_WRAP = enable
        yield
    finally:
        _ENABLE_TORCH_WRAP = old


# 0-param decorator overload
@overload
def register_op(f: Callable[..., Any]) -> "IrOp": ...


# parametrized decorator overload
@overload
def register_op(
    *,
    name: str | None = None,
    has_reduction: bool = False,
    activations: list[str] | None = None,
    allow_inplace: bool = False,
) -> Callable[[Callable[..., Any]], "IrOp"]: ...


def register_op(
    f: Callable | None = None,
    *,
    name: str | None = None,
    has_reduction: bool = False,
    activations: list[str] | None = None,
    allow_inplace: bool = False,
) -> "IrOp | Callable[[Callable], IrOp]":
    """
    Register a new vLLM IR op.

    :param f: the native implementation of the op
    :param name: the name of the op, defaults to the function name
    :param has_reduction: is this op is a reduction op, which affects batch-invariance
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
        op_name: str = _f.__name__ if name is None else name
        assert op_name not in IrOp.registry
        op = IrOp(op_name, _f, has_reduction, activations, allow_inplace)
        IrOp.registry[op_name] = op
        return op

    if f is not None:
        return decorator(f)

    return decorator


class IrOp:
    registry: ClassVar[dict[str, "IrOp"]] = {}

    name: str
    has_reduction: bool
    impls: dict[str, "IrOpImpl"]
    maybe_inplace: "IrOpInplaceOverload | None"

    def __init__(
        self,
        name: str,
        native_impl: Callable,
        has_reduction: bool,
        activations: list[str] | None = None,
        allow_inplace: bool = False,
    ):
        self._py_signature = inspect.signature(native_impl)
        if any(
            p.kind == inspect.Parameter.KEYWORD_ONLY
            for p in self._py_signature.parameters.values()
        ):
            raise ValueError(
                f"Op {name} has keyword-only arguments which are not currently "
                f"supported. That's because kwargs are not allowed during lowering."
            )

        # By convention, we consider parameters starting with 'x' as activations.
        if activations is None:
            activations = [
                p.name
                for p in self._py_signature.parameters.values()
                if p.name.startswith("x")
            ]

        self.name = name
        self.has_reduction = has_reduction
        self.activations = activations
        self.activation_indices = [
            i
            for i, p in enumerate(self._py_signature.parameters.values())
            if p.name in activations
        ]
        self.impls: dict[str, IrOpImpl] = {}
        self._priority_impls: list[IrOpImpl] = []
        self._schema_str = infer_schema(native_impl, mutates_args=[])
        self.allow_inplace = allow_inplace

        # native implementation
        self.impls["native"] = IrOpImpl(
            self,
            "native",
            native_impl,
            # always supported
            supported=True,
            supports_args=None,
            # Native implementation is always batch-invariant
            # (batch invariance is controlled at the torch level)
            batch_invariant=True,
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

        if self.allow_inplace:
            self.maybe_inplace = IrOpInplaceOverload(self)
        else:
            self.maybe_inplace = None

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
        batch_invariant: bool | None = None,
        inplace: bool = False,
    ):
        """
        Register an implementation for this custom op.
        :param provider: The name of the provider, must be unique.
        :param supported: Static support check, use this to check platform support.
        :param supports_args: Dynamic arg support check, used for types and shapes.
        :param batch_invariant: is this implementation is batch-invariant.
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

        Default behavior of batch_invariant depends on op.has_reduction:
        - op.has_reduction == True: batch_invariant = False
        - op.has_reduction == False: batch_invariant = True

        This is because ops without reductions are always batch-invariant.
        Ops with reductions have to opt-in, as they are not batch-invariant by default.

        """
        assert provider not in RESERVED_PROVIDERS, (
            f"Provider name {provider} is reserved."
        )

        if batch_invariant is None:
            batch_invariant = not self.has_reduction

        def _register_impl(f: Callable):
            impl = IrOpImpl(
                self, provider, f, supported, supports_args, batch_invariant, inplace
            )
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
        Eager call to torch op lands here. When torch wrapping is disabled,
        __call__ routes straight here instead of going through torch op dispatching.
        """
        impl = self.dispatch(*args, **kwargs)

        # Default overload must be functional,
        # use func_impl_fn to correctly handle inplace impls.
        return impl.func_impl_fn(*args, **kwargs)

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
                # (this code is exposed when torch wrapping is disabled)
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
        if not _ENABLE_TORCH_WRAP:
            return self._inner_call(*args, **kwargs)

        return self.torch_op(*args, **kwargs)

    def get_priority(self) -> list[str]:
        """Get the current dispatch priority for implementations for this op."""
        return [p.provider for p in self._priority_impls]

    @contextlib.contextmanager
    def set_priority(self, priority: list[str], *, batch_invariant_only: bool = False):
        """
        Context manager to set the dispatch priority for implementations for this op.
        """
        assert all(p in self.impls for p in priority), (
            f"All providers in priority must be registered implementations, missing "
            f"{','.join(p for p in priority if p not in self.impls)}"
        )

        def filter_priority_impls(p_list: list[str]) -> list[IrOpImpl]:
            filtered_impls = []
            for p in p_list:
                impl = self.impls[p]
                if not impl.supported:
                    # Skip unsupported implementations
                    continue

                if batch_invariant_only and not impl.batch_invariant:
                    # Skip batch invariant implementations
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
            logger.debug(
                "Priority for vllm.ir.%s set to %s",
                self.name,
                lazy(lambda: [p.provider for p in self._priority_impls]),
            )
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
        self._schema_str = infer_schema(
            op.impls["native"].impl_fn, mutates_args=op.activations
        )

        # torch registration
        vllm_ir_lib.define(self.name + self._schema_str)
        vllm_ir_lib.impl(
            self.name, self._inner_call, dispatch_key="CompositeExplicitAutograd"
        )
        # fake goes to default overload for now
        vllm_ir_lib._register_fake(self.name, self.op._fake_call)

        assert hasattr(getattr(torch.ops.vllm_ir, self.op.name), "maybe_inplace")
        self.torch_op = getattr(torch.ops.vllm_ir, self.op.name).maybe_inplace

    def __call__(self, *args, **kwargs) -> Any:
        if not _ENABLE_TORCH_WRAP:
            return self._inner_call(*args, **kwargs)

        return self.torch_op(*args, **kwargs)

    def _inner_call(self, *args, **kwargs) -> Any:
        # Calling the maybe_inplace overload means we can use inplace impls directly.
        impl = self.op.dispatch(*args, **kwargs)
        return impl.impl_fn(*args, **kwargs)


class IrOpImpl:
    op: IrOp
    provider: str
    impl_fn: Callable
    supported: bool
    batch_invariant: bool

    def __init__(
        self,
        op: IrOp,
        provider: str,
        impl_fn: Callable,
        supported: bool,
        supports_args: Callable[..., bool] | None,
        batch_invariant: bool,
        inplace: bool = False,
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
        self.batch_invariant = batch_invariant
        self.inplace = inplace

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

        Source file contents do not change so we cache uuid.
        TODO(luka): Cache the file hash as multiple impls are likely in the same file.
        """
        sources = [Path(inspect.getfile(self.impl_fn))]
        return hash_source(*sources)

    def func_impl_fn(self, *args, **kwargs) -> Any:
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
