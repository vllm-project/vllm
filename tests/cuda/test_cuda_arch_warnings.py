# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from vllm.platforms.interface import DeviceCapability


@pytest.fixture
def cuda_platform_base(monkeypatch: pytest.MonkeyPatch) -> Any:
    stable_libtorch_module = ModuleType("vllm._C_stable_libtorch")
    monkeypatch.setitem(
        sys.modules,
        "vllm._C_stable_libtorch",
        stable_libtorch_module,
    )
    from vllm.platforms.cuda import CudaPlatformBase

    return CudaPlatformBase


def test_compiled_arch_covers_device(cuda_platform_base: Any) -> None:
    assert cuda_platform_base._compiled_arch_covers_device(
        "12.1a", DeviceCapability(12, 1)
    )
    assert not cuda_platform_base._compiled_arch_covers_device(
        "12.0a", DeviceCapability(12, 1)
    )

    assert cuda_platform_base._compiled_arch_covers_device(
        "12.0f", DeviceCapability(12, 1)
    )
    assert not cuda_platform_base._compiled_arch_covers_device(
        "10.0f", DeviceCapability(12, 1)
    )


def test_warn_if_device_arch_not_compiled(
    monkeypatch: pytest.MonkeyPatch, cuda_platform_base: Any
) -> None:
    def device_count(cls: type[Any]) -> int:
        return 2

    def get_device_capability(
        cls: type[Any], device_id: int = 0
    ) -> DeviceCapability | None:
        capabilities = {
            0: DeviceCapability(12, 1),
            1: DeviceCapability(10, 3),
        }
        return capabilities[device_id]

    def get_device_name(cls: type[Any], device_id: int = 0) -> str:
        return f"GPU {device_id}"

    warnings: list[tuple[str, str, str]] = []

    def warning_once(message: str, compiled_archs: str, devices: str) -> None:
        warnings.append((message, compiled_archs, devices))

    monkeypatch.setattr(cuda_platform_base, "device_count", classmethod(device_count))
    monkeypatch.setattr(
        cuda_platform_base,
        "get_device_capability",
        classmethod(get_device_capability),
    )
    monkeypatch.setattr(
        cuda_platform_base, "get_device_name", classmethod(get_device_name)
    )
    monkeypatch.setattr(
        "vllm.platforms.cuda.torch.ops",
        SimpleNamespace(
            _C=SimpleNamespace(get_compiled_cuda_archs=lambda: "12.0f,10.0a")
        ),
    )
    monkeypatch.setattr(
        "vllm.platforms.cuda.logger",
        SimpleNamespace(warning_once=warning_once),
    )

    cuda_platform_base._warn_if_device_arch_not_compiled()

    assert len(warnings) == 1
    assert warnings[0][1] == "12.0f, 10.0a"
    assert "1: GPU 1 (compute capability 10.3)" in warnings[0][2]


def test_warn_if_device_arch_not_compiled_no_warning(
    monkeypatch: pytest.MonkeyPatch, cuda_platform_base: Any
) -> None:
    def device_count(cls: type[Any]) -> int:
        return 1

    def get_device_capability(
        cls: type[Any], device_id: int = 0
    ) -> DeviceCapability | None:
        return DeviceCapability(10, 3)

    def get_device_name(cls: type[Any], device_id: int = 0) -> str:
        raise AssertionError("get_device_name should not be called for covered devices")

    warnings: list[tuple[str, str, str]] = []

    def warning_once(message: str, compiled_archs: str, devices: str) -> None:
        warnings.append((message, compiled_archs, devices))

    monkeypatch.setattr(cuda_platform_base, "device_count", classmethod(device_count))
    monkeypatch.setattr(
        cuda_platform_base,
        "get_device_capability",
        classmethod(get_device_capability),
    )
    monkeypatch.setattr(
        cuda_platform_base, "get_device_name", classmethod(get_device_name)
    )
    monkeypatch.setattr(
        "vllm.platforms.cuda.torch.ops",
        SimpleNamespace(_C=SimpleNamespace(get_compiled_cuda_archs=lambda: "10.0f")),
    )
    monkeypatch.setattr(
        "vllm.platforms.cuda.logger",
        SimpleNamespace(warning_once=warning_once),
    )

    cuda_platform_base._warn_if_device_arch_not_compiled()

    assert warnings == []
