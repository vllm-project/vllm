# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    VllmTritonJitKernel,
)


class _RecordingTritonKernel:
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

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return {
            "ptr": TritonWarmupTensor(torch.int32),
            "value": compile_key.value,
        }

    def __call__(self, ptr: Any, value: int) -> None:
        self._launch((value,), ptr, value, BLOCK_SIZE=value * 2)


def test_triton_owner_reuses_runtime_launch_for_warmup() -> None:
    owner = _RecordingOwner()
    owner.kernel = _RecordingTritonKernel()

    owner.compile(owner.CompileKey(8))

    assert owner.kernel.warmups == [
        (
            (TritonWarmupTensor(torch.int32), 8),
            {"BLOCK_SIZE": 16, "grid": (1,)},
        )
    ]
    assert owner.kernel.launches == []


def test_triton_owner_runtime_uses_real_grid() -> None:
    owner = _RecordingOwner()
    owner.kernel = _RecordingTritonKernel()
    ptr = object()

    owner(ptr, 8)

    assert owner.kernel.launches == [
        ((8,), (ptr, 8), {"BLOCK_SIZE": 16})
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
    tensor = TritonWarmupTensor(
        torch.int32,
        shape=(2, 3, 4),
        strides=(16, 4, 1),
    )

    assert tensor.ndim == 3
    assert tensor.size() == (2, 3, 4)
    assert tensor.size(1) == 3
    assert tensor.numel() == 24
    assert tensor.element_size() == 4
    assert tensor.stride() == (16, 4, 1)
    assert tensor.stride(0) == 16
    assert tensor.int().dtype == torch.int32
    assert tensor.new_empty((5, 6), dtype=torch.float32).shape == (5, 6)


def test_triton_owner_supports_positional_warmup_inputs() -> None:
    class _PositionalOwner(_RecordingOwner):
        def warmup_inputs(
            self, compile_key: _RecordingOwner.CompileKey
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return (TritonWarmupTensor(torch.int32), compile_key.value), {}

    owner = _PositionalOwner()
    owner.kernel = _RecordingTritonKernel()
    owner.compile(owner.CompileKey(3))

    assert owner.kernel.warmups[0][0][1] == 3
