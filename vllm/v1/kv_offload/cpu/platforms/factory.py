# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory selecting the platform runtime for the CPU offload worker."""

import importlib
from collections.abc import Callable, Sequence

from vllm.platforms import PlatformEnum, current_platform
from vllm.v1.kv_offload.cpu.engines.base import CopyEngine
from vllm.v1.kv_offload.cpu.platforms.base import DeviceRuntime


class UnsupportedRuntime(DeviceRuntime):
    """Placeholder for platforms without a CPU offload implementation."""

    def create_copy_engine(
        self,
        *,
        device_to_cpu: bool,
        host_memory_is_pinned: bool,
        page_sizes: Sequence[int],
    ) -> CopyEngine:
        raise RuntimeError(
            f"CPU offloading is not implemented for {current_platform.device_name}"
        )


class DeviceRuntimeFactory:
    """Registry mapping vLLM platforms to their CPU offload runtime.

    Registrations hold an import path rather than a class, so selecting a
    runtime never imports a platform module (or touches a device) until it is
    used.
    """

    _registry: dict[PlatformEnum, Callable[[], type[DeviceRuntime]]] = {}

    @classmethod
    def register(
        cls, platform: PlatformEnum, module_path: str, class_name: str
    ) -> None:
        if platform in cls._registry:
            raise ValueError(f"Platform {platform} already has a device runtime.")

        def load() -> type[DeviceRuntime]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[platform] = load

    @classmethod
    def get_runtime(cls, platform: PlatformEnum) -> DeviceRuntime:
        load = cls._registry.get(platform)
        if load is None:
            return UnsupportedRuntime()
        return load()()


DeviceRuntimeFactory.register(
    PlatformEnum.CUDA,
    "vllm.v1.kv_offload.cpu.platforms.cuda",
    "CudaRuntime",
)
DeviceRuntimeFactory.register(
    PlatformEnum.ROCM,
    "vllm.v1.kv_offload.cpu.platforms.rocm",
    "RocmRuntime",
)
DeviceRuntimeFactory.register(
    PlatformEnum.XPU,
    "vllm.v1.kv_offload.cpu.platforms.xpu",
    "XPURuntime",
)


def get_device_runtime() -> DeviceRuntime:
    """Return the runtime registered for the current platform.

    Intentionally uncached: platform imports stay lazy, so this is safe to call
    from the scheduler process, and tests can patch `current_platform`.
    """
    return DeviceRuntimeFactory.get_runtime(current_platform._enum)
