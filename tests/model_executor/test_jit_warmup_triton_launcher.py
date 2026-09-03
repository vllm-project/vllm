# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    VllmTritonJitKernel,
    kernel_launcher,
)


class _FakeTritonKernel:
    arg_names = ("first", "second", "CONST")

    def __init__(self) -> None:
        self.warmup_calls: list[dict[str, Any]] = []

    def warmup(self, **kwargs: Any) -> None:
        self.warmup_calls.append(kwargs)


class _TestTritonKernel(VllmTritonJitKernel["_TestTritonKernel.CompileKey"]):
    kernel = _FakeTritonKernel()

    @dataclass(frozen=True)
    class CompileKey:
        value: int

    def dispatch(self, *, value: int) -> CompileKey:
        return self.CompileKey(value=value)

    def get_warmup_keys(self) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(value=1)

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(first="warmup", second=compile_key.value, runtime_launcher=None)

    @kernel_launcher
    def __call__(
        self,
        first: str,
        second: int,
        runtime_launcher: Any,
    ) -> LaunchSpec:
        return (2,), dict(
            CONST=7,
            _runtime_launcher=runtime_launcher,
            _runtime_launcher_arg_count=2,
        )


def test_triton_launcher_supports_compile_and_runtime_adapters() -> None:
    owner = _TestTritonKernel()
    owner.kernel.warmup_calls.clear()

    owner.compile(owner.CompileKey(value=1))
    assert owner.kernel.warmup_calls == [
        {"grid": (1,), "first": "warmup", "second": 1, "CONST": 7}
    ]

    runtime_calls: list[tuple[Any, ...]] = []

    def runtime_launcher(kernel: Any, grid: Any, *args: Any, **kwargs: Any) -> None:
        runtime_calls.append((kernel, grid, args, kwargs))

    owner("runtime", 2, runtime_launcher)
    assert runtime_calls == [(owner.kernel, (2,), ("runtime", 2), {"CONST": 7})]
