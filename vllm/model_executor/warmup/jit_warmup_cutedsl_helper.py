# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, ClassVar, Generic, TypeAlias, TypeVar

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel

DEFAULT_CUTEDSL_COMPILE_OPTIONS = "--enable-tvm-ffi"

CompileKeyT = TypeVar("CompileKeyT")
CuTeDSLLaunchSpec: TypeAlias = tuple[
    CompileKeyT,
    tuple[Any, ...],
    Mapping[str, Any] | None,
]


def cutedsl_fake_stream(*, use_tvm_ffi_env_stream: bool = True) -> Any:
    import cutlass.cute as cute

    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=use_tvm_ffi_env_stream)


def compile_cutedsl(
    entry: Callable[..., Any],
    *args: Any,
    options: str = DEFAULT_CUTEDSL_COMPILE_OPTIONS,
    use_tvm_ffi_env_stream: bool = True,
) -> Any:
    import cutlass.cute as cute

    return cute.compile(
        entry,
        *args,
        cutedsl_fake_stream(use_tvm_ffi_env_stream=use_tvm_ffi_env_stream),
        options=options,
    )


class VllmCuTeDSLJitKernel(VllmJitKernel[CompileKeyT], Generic[CompileKeyT]):
    """CuTeDSL owner whose compiled executor is shared by warmup and runtime."""

    kernel: ClassVar[Any]

    @abstractmethod
    def warmup_inputs(self, compile_key: CompileKeyT) -> tuple[Any, ...]:
        """Return fake arguments that compile one executor specialization."""
        raise NotImplementedError

    def compile(self, compile_key: CompileKeyT) -> None:
        if compile_key in self._compiled_cache:
            return
        self._compiled_cache[compile_key] = compile_cutedsl(
            self.kernel(compile_key),
            *self.warmup_inputs(compile_key),
        )


def cutedsl_kernel_launcher(
    call_fn: Callable[..., CuTeDSLLaunchSpec[CompileKeyT]],
) -> Callable[..., Any]:
    """Invoke a cached CuTeDSL executor from a declarative ``__call__``."""

    @wraps(call_fn)
    def wrapper(
        self: VllmCuTeDSLJitKernel[CompileKeyT],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        compile_key, launch_args, runtime_context = call_fn(self, *args, **kwargs)
        executor = self._get_or_compile(
            compile_key,
            runtime_context=runtime_context,
        )
        return executor(*launch_args)

    return wrapper
