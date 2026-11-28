# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the CPU offload runtime factory."""

import pytest

import vllm.v1.kv_offload.cpu.platforms.factory as platform_module
from vllm.platforms import PlatformEnum, current_platform
from vllm.v1.kv_offload.cpu.engines.factory import CopyEngineFactory
from vllm.v1.kv_offload.cpu.engines.torch import TorchCopyEngine
from vllm.v1.kv_offload.cpu.platforms.base import RuntimeCapabilities
from vllm.v1.kv_offload.cpu.platforms.cuda import CudaRuntime
from vllm.v1.kv_offload.cpu.platforms.factory import UnsupportedRuntime
from vllm.v1.kv_offload.cpu.platforms.rocm import RocmRuntime
from vllm.v1.kv_offload.cpu.platforms.xpu import XPURuntime

PAGE = 0x100


@pytest.mark.parametrize(
    "platform,expected,caps",
    [
        (
            PlatformEnum.CUDA,
            CudaRuntime,
            RuntimeCapabilities(cpu_offload=True, shared_region=True, async_init=True),
        ),
        (
            PlatformEnum.ROCM,
            RocmRuntime,
            RuntimeCapabilities(cpu_offload=True),
        ),
        (
            PlatformEnum.XPU,
            XPURuntime,
            RuntimeCapabilities(cpu_offload=True, shared_region=True),
        ),
        # TPU is deliberately not registered: CPU offloading is unsupported.
        (PlatformEnum.TPU, UnsupportedRuntime, RuntimeCapabilities()),
        (PlatformEnum.CPU, UnsupportedRuntime, RuntimeCapabilities()),
    ],
    ids=["cuda", "rocm", "xpu", "tpu", "cpu"],
)
def test_runtime_selection_and_capabilities(
    monkeypatch, platform, expected, caps
) -> None:
    monkeypatch.setattr(current_platform, "_enum", platform)

    runtime = platform_module.get_device_runtime()

    assert isinstance(runtime, expected)
    assert runtime.capabilities == caps


def test_copy_engine_factory_selects_torch() -> None:
    assert CopyEngineFactory.get_engine("torch") is TorchCopyEngine
    with pytest.raises(ValueError, match="Unknown copy mechanism"):
        CopyEngineFactory.get_engine("missing")


def test_unsupported_runtime_rejects_copy_engine() -> None:
    with pytest.raises(RuntimeError, match="not implemented"):
        UnsupportedRuntime().create_copy_engine(
            device_to_cpu=True,
            host_memory_is_pinned=True,
            page_sizes=[PAGE],
        )


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="descriptor buffers need an accelerator",
)
def test_descriptor_buffers_use_platform_pointer_dtype() -> None:
    """XPU's DMA op requires uint64 for addresses and sizes; CUDA uses int64."""
    engine = platform_module.get_device_runtime().create_copy_engine(
        device_to_cpu=False,
        host_memory_is_pinned=True,
        page_sizes=[PAGE],
    )
    assert isinstance(engine, TorchCopyEngine)
    src, dst, sizes = engine.new_batch(4).payload
    assert src.dtype == engine.ptr_dtype
    assert dst.dtype == engine.ptr_dtype
    assert sizes.dtype == engine.ptr_dtype
