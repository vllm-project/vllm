# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

from vllm.model_executor.warmup import jit_warmup_cutedsl_helper
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)


class _TestCuTeDSLKernel(VllmCuTeDSLJitKernel["_TestCuTeDSLKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        variant: int

    @staticmethod
    def kernel(compile_key: CompileKey) -> tuple[str, int]:
        return "entry", compile_key.variant

    def dispatch(self, *, variant: int) -> CompileKey:
        return self.CompileKey(variant=variant)

    def get_warmup_keys(self) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(variant=(1, 2))

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        return (f"fake-{compile_key.variant}",)

    @kernel_launcher
    def __call__(
        self,
        payload: str,
        *,
        variant: int,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        compile_key = self.dispatch(variant=variant)
        return compile_key, (payload,)


def test_cutedsl_launcher_reuses_compiled_executor(monkeypatch) -> None:
    compile_calls: list[tuple[Any, tuple[Any, ...]]] = []
    launch_calls: list[tuple[int, tuple[Any, ...]]] = []

    def compile_cutedsl(entry: tuple[str, int], *args: Any) -> Any:
        compile_calls.append((entry, args))
        variant = entry[1]

        def executor(*runtime_args: Any) -> tuple[int, tuple[Any, ...]]:
            launch_calls.append((variant, runtime_args))
            return variant, runtime_args

        return executor

    monkeypatch.setattr(
        jit_warmup_cutedsl_helper,
        "compile_cutedsl",
        compile_cutedsl,
    )
    kernel = _TestCuTeDSLKernel()
    compile_key = kernel.CompileKey(variant=1)

    kernel.compile(compile_key)
    kernel.compile(compile_key)
    assert compile_calls == [(("entry", 1), ("fake-1",))]

    assert kernel("runtime", variant=1) == (1, ("runtime",))
    assert kernel("other", variant=2) == (2, ("other",))
    assert compile_calls == [
        (("entry", 1), ("fake-1",)),
        (("entry", 2), ("fake-2",)),
    ]
    assert launch_calls == [(1, ("runtime",)), (2, ("other",))]


def test_cutedsl_launcher_runs_post_launch_finalizer(monkeypatch) -> None:
    events: list[str] = []

    def compile_cutedsl(entry: tuple[str, int], *args: Any) -> Any:
        def executor(*runtime_args: Any) -> None:
            events.append("launch")

        return executor

    class FinalizedKernel(_TestCuTeDSLKernel):
        @kernel_launcher
        def __call__(
            self,
            payload: str,
            *,
            variant: int,
        ) -> CuTeDSLLaunchSpec[_TestCuTeDSLKernel.CompileKey]:
            compile_key = self.dispatch(variant=variant)

            def finalize() -> str:
                events.append("finalize")
                return "output"

            return compile_key, (payload,), "unused", finalize

    monkeypatch.setattr(
        jit_warmup_cutedsl_helper,
        "compile_cutedsl",
        compile_cutedsl,
    )

    assert FinalizedKernel()("runtime", variant=1) == "output"
    assert events == ["launch", "finalize"]
