# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupInputs,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
)


class _RecordingTritonKernel:
    arg_names: tuple[str, ...] = ("ptr", "value", "BLOCK_SIZE")

    def __init__(self) -> None:
        self.launches: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        self.warmups: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.fail_warmup = False

    def __getitem__(self, grid: Any):
        def launch(*args: Any, **kwargs: Any) -> None:
            self.launches.append((grid, args, kwargs))

        return launch

    def warmup(self, *args: Any, **kwargs: Any) -> None:
        if self.fail_warmup:
            raise RuntimeError("compile failed")
        self.warmups.append((args, kwargs))


class _RecordingOwner(VllmTritonJitKernel["_RecordingOwner.CompileKey"]):
    kernel = _RecordingTritonKernel()

    @dataclass(frozen=True)
    class CompileKey:
        value: int

    def dispatch(self, *, value: int) -> CompileKey:  # type: ignore[override]
        return self.CompileKey(value)

    def get_warmup_keys(self) -> list[CompileKey]:
        return []

    def warmup_inputs(self, compile_key: CompileKey) -> TritonWarmupInputs:
        return {
            "ptr": TritonWarmupTensor(torch.int32),
            "value": compile_key.value,
        }

    @kernel_launcher
    def __call__(self, ptr: Any, value: int) -> LaunchSpec:
        return (value,), {"BLOCK_SIZE": value * 2}


def test_triton_owner_reuses_runtime_launch_for_warmup() -> None:
    owner = _RecordingOwner()
    owner.kernel = _RecordingTritonKernel()

    owner.compile(owner.CompileKey(8))

    assert owner.kernel.warmups == [
        (
            (),
            {
                "ptr": TritonWarmupTensor(torch.int32),
                "value": 8,
                "BLOCK_SIZE": 16,
                "grid": (1,),
            },
        )
    ]
    assert owner.kernel.launches == []


def test_triton_owner_runtime_uses_real_grid() -> None:
    owner = _RecordingOwner()
    owner.kernel = _RecordingTritonKernel()
    ptr = object()

    owner(ptr, 8)

    assert owner.kernel.launches == [
        ((8,), (), {"ptr": ptr, "value": 8, "BLOCK_SIZE": 16})
    ]
    assert owner.kernel.warmups == []


def test_triton_owner_restores_runtime_mode_after_compile_error() -> None:
    owner = _RecordingOwner()
    owner.kernel = _RecordingTritonKernel()
    owner.kernel.fail_warmup = True

    with pytest.raises(RuntimeError, match="compile failed"):
        owner.compile(owner.CompileKey(8))

    owner.kernel.fail_warmup = False
    owner(object(), 4)
    assert len(owner.kernel.launches) == 1


def test_triton_warmup_tensor_runtime_metadata() -> None:
    padded = TritonWarmupTensor(
        torch.int32,
        shape=(2, 3, 4),
        strides=(16, 4, 1),
    )

    assert padded.ndim == 3
    assert padded.size() == (2, 3, 4)
    assert padded.size(1) == 3
    assert padded.numel() == 24
    assert padded.element_size() == 4
    assert padded.stride() == (16, 4, 1)
    assert padded.stride(0) == 16
    assert not padded.is_contiguous()
    assert padded.int().dtype == torch.int32
    assert padded.new_empty((5, 6), dtype=torch.float32).shape == (5, 6)

    contiguous = TritonWarmupTensor(torch.int32, shape=(2, 3, 4))
    assert contiguous.stride() == (12, 4, 1)
    assert contiguous.is_contiguous()

    transposed = TritonWarmupTensor(
        torch.int32,
        shape=(2, 3),
        strides=(1, 2),
    )
    assert not transposed.is_contiguous()


def test_triton_owner_supports_positional_warmup_inputs() -> None:
    class _PositionalOwner(_RecordingOwner):
        def warmup_inputs(
            self, compile_key: _RecordingOwner.CompileKey
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return (TritonWarmupTensor(torch.int32), compile_key.value), {}

    owner = _PositionalOwner()
    owner.kernel = _RecordingTritonKernel()
    owner.compile(owner.CompileKey(3))

    assert owner.kernel.warmups[0][1]["value"] == 3


def test_triton_owner_forwards_variadic_inputs() -> None:
    class _VariadicOwner(_RecordingOwner):
        @kernel_launcher
        def __call__(self, ptr: Any, value: int, *args: Any) -> LaunchSpec:
            return (value,), {"BLOCK_SIZE": value * 2}

        def warmup_inputs(
            self, compile_key: _RecordingOwner.CompileKey
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return (
                TritonWarmupTensor(torch.int32),
                compile_key.value,
                compile_key.value * 3,
            ), {}

    owner = _VariadicOwner()
    owner.kernel = _RecordingTritonKernel()
    owner.kernel.arg_names = ("ptr", "value", "extra", "BLOCK_SIZE")
    owner.compile(owner.CompileKey(3))

    assert owner.kernel.warmups[0][1]["extra"] == 9


def test_triton_launcher_preserves_result_and_skipped_launch() -> None:
    class _ResultOwner(_RecordingOwner):
        @kernel_launcher
        def __call__(self, ptr: Any, value: int) -> LaunchSpec:
            result = value * 3
            if value == 0:
                return None, {}, result
            return (value,), {"BLOCK_SIZE": value * 2}, result

    owner = _ResultOwner()
    owner.kernel = _RecordingTritonKernel()

    assert owner(object(), 4) == 12
    assert len(owner.kernel.launches) == 1
    assert owner(object(), 0) == 0
    assert len(owner.kernel.launches) == 1


def test_triton_launcher_preserves_custom_runtime_arguments() -> None:
    calls: list[tuple[Any, ...]] = []

    def runtime_launcher(
        kernel: Any,
        grid: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        calls.append((kernel, grid, args, kwargs))

    class _CustomLauncherOwner(_RecordingOwner):
        @kernel_launcher
        def __call__(self, ptr: Any, value: int) -> LaunchSpec:
            return (value,), {
                "BLOCK_SIZE": value * 2,
                "_runtime_launcher": runtime_launcher,
                "_runtime_launcher_arg_count": 2,
            }

    owner = _CustomLauncherOwner()
    owner.kernel = _RecordingTritonKernel()
    ptr = object()
    owner(ptr, 4)

    assert calls == [(owner.kernel, (4,), (ptr, 4), {"BLOCK_SIZE": 8})]
