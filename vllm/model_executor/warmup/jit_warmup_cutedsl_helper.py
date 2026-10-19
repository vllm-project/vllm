# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Generic, TypeAlias, TypeVar

from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
)

DEFAULT_CUTEDSL_COMPILE_OPTIONS = "--enable-tvm-ffi"

CompileKeyT = TypeVar("CompileKeyT")
CuTeDSLLaunchSpec: TypeAlias = (
    tuple[CompileKeyT, tuple[Any, ...]]
    | tuple[CompileKeyT, tuple[Any, ...], Any]
    | tuple[
        CompileKeyT,
        tuple[Any, ...],
        Any,
        Callable[[], Any],
    ]
)


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
    bind_launch_inputs = False

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

    def launch(
        self,
        launch_spec: CuTeDSLLaunchSpec[CompileKeyT],
        _inputs: Mapping[str, Any],
    ) -> Any:
        # (compile_key, args), optionally followed by output and epilogue.
        compile_key, launch_args = launch_spec[:2]
        executor = self._get_or_compile(compile_key)
        result = executor(*launch_args)
        if len(launch_spec) == 4:
            return launch_spec[3]()
        return launch_spec[2] if len(launch_spec) == 3 else result
