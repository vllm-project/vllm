# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.platforms.interface import DeviceCapability


@pytest.fixture
def cuda_platform_base():
    pytest.importorskip("vllm._C_stable_libtorch")
    from vllm.platforms.cuda import CudaPlatformBase

    return CudaPlatformBase


def test_compiled_arch_covers_device(cuda_platform_base):
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


def test_warn_if_device_arch_not_compiled(monkeypatch, cuda_platform_base):
    class MockCudaPlatform(cuda_platform_base):
        @classmethod
        def device_count(cls) -> int:
            return 2

        @classmethod
        def get_device_capability(
            cls, device_id: int = 0
        ) -> DeviceCapability | None:
            capabilities = {
                0: DeviceCapability(12, 1),
                1: DeviceCapability(10, 3),
            }
            return capabilities[device_id]

        @classmethod
        def get_device_name(cls, device_id: int = 0) -> str:
            return f"GPU {device_id}"

    warnings: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        "vllm.platforms.cuda.torch.ops",
        SimpleNamespace(
            _C=SimpleNamespace(get_compiled_cuda_archs=lambda: "12.0f,10.0a")
        ),
    )
    monkeypatch.setattr(
        "vllm.platforms.cuda.logger",
        SimpleNamespace(warning_once=lambda *args: warnings.append(args)),
    )

    MockCudaPlatform._warn_if_device_arch_not_compiled()

    assert len(warnings) == 1
    assert warnings[0][1] == "12.0f, 10.0a"
    assert "1: GPU 1 (compute capability 10.3)" in warnings[0][2]


def test_warn_if_device_arch_not_compiled_no_warning(monkeypatch, cuda_platform_base):
    class MockCudaPlatform(cuda_platform_base):
        @classmethod
        def device_count(cls) -> int:
            return 1

        @classmethod
        def get_device_capability(
            cls, device_id: int = 0
        ) -> DeviceCapability | None:
            return DeviceCapability(10, 3)

        @classmethod
        def get_device_name(cls, device_id: int = 0) -> str:
            pytest.fail("get_device_name should not be called for covered devices")

    warnings: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        "vllm.platforms.cuda.torch.ops",
        SimpleNamespace(_C=SimpleNamespace(get_compiled_cuda_archs=lambda: "10.0f")),
    )
    monkeypatch.setattr(
        "vllm.platforms.cuda.logger",
        SimpleNamespace(warning_once=lambda *args: warnings.append(args)),
    )

    MockCudaPlatform._warn_if_device_arch_not_compiled()

    assert warnings == []
