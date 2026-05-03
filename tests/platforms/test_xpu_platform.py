# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import os
import sys
import types
from enum import Enum, auto
from pathlib import Path

import pytest

XPU_PLATFORM_PATH = (
    Path(__file__).resolve().parents[2] / "vllm" / "platforms" / "xpu.py"
)


class _FakeLogger:

    def warning(self, *args, **kwargs) -> None:
        pass

    def warning_once(self, *args, **kwargs) -> None:
        pass

    def info(self, *args, **kwargs) -> None:
        pass

    def info_once(self, *args, **kwargs) -> None:
        pass


def _install_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    module: types.ModuleType,
) -> None:
    monkeypatch.setitem(sys.modules, name, module)


def _load_xpu_platform(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_triton_awq: bool,
):
    for package_name in (
        "vllm",
        "vllm.platforms",
        "vllm.utils",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm_xpu_kernels",
    ):
        package = types.ModuleType(package_name)
        package.__path__ = []
        _install_module(monkeypatch, package_name, package)

    torch_module = types.ModuleType("torch")
    torch_module.Tensor = type("Tensor", (), {})
    torch_module.device = type("device", (), {})
    torch_module.dtype = type("dtype", (), {})
    torch_module.types = types.SimpleNamespace(Device=object)
    torch_module.float32 = object()
    torch_module.float16 = object()
    torch_module.bfloat16 = object()
    torch_module.float8_e4m3fn = object()
    torch_module.no_grad = lambda: None
    torch_module.xpu = types.SimpleNamespace(
        set_device=lambda device: None,
        manual_seed_all=lambda seed: None,
        get_device_name=lambda device_id=0: "fake-xpu",
        get_device_properties=lambda device_id=0: types.SimpleNamespace(
            total_memory=0,
            max_compute_units=0,
        ),
        empty_cache=lambda: None,
        reset_peak_memory_stats=lambda device=None: None,
        max_memory_allocated=lambda device=None: 0,
        device_count=lambda: 0,
    )
    _install_module(monkeypatch, "torch", torch_module)

    envs_module = types.ModuleType("vllm.envs")
    envs_module.VLLM_USE_TRITON_AWQ = use_triton_awq
    envs_module.VLLM_XPU_ENABLE_XPU_GRAPH = False
    _install_module(monkeypatch, "vllm.envs", envs_module)

    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: _FakeLogger()
    _install_module(monkeypatch, "vllm.logger", logger_module)

    torch_utils_module = types.ModuleType("vllm.utils.torch_utils")
    torch_utils_module.supports_xpu_graph = lambda: False
    _install_module(monkeypatch, "vllm.utils.torch_utils", torch_utils_module)

    registry_module = types.ModuleType("vllm.v1.attention.backends.registry")
    registry_module.AttentionBackendEnum = type("AttentionBackendEnum", (), {})
    _install_module(
        monkeypatch,
        "vllm.v1.attention.backends.registry",
        registry_module,
    )

    interface_module = types.ModuleType("vllm.platforms.interface")

    class PlatformEnum(Enum):
        XPU = auto()

    class DeviceCapability(tuple):
        pass

    class Platform:
        supported_quantization: list[str] = []
        device_name = "xpu"

        @classmethod
        def verify_quantization(cls, quant: str) -> None:
            if cls.supported_quantization and quant not in cls.supported_quantization:
                raise ValueError(
                    f"{quant} quantization is currently not supported in "
                    f"{cls.device_name}."
                )

    interface_module.DeviceCapability = DeviceCapability
    interface_module.Platform = Platform
    interface_module.PlatformEnum = PlatformEnum
    _install_module(monkeypatch, "vllm.platforms.interface", interface_module)

    for suffix in ("_C", "_moe_C", "_xpu_C"):
        _install_module(
            monkeypatch,
            f"vllm_xpu_kernels.{suffix}",
            types.ModuleType(f"vllm_xpu_kernels.{suffix}"),
        )

    monkeypatch.delitem(sys.modules, "vllm.platforms.xpu", raising=False)
    spec = importlib.util.spec_from_file_location(
        "vllm.platforms.xpu",
        XPU_PLATFORM_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    _install_module(monkeypatch, "vllm.platforms.xpu", module)
    spec.loader.exec_module(module)
    return module.XPUPlatform


def test_xpu_awq_forces_triton_awq(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_TRITON_AWQ", "0")

    xpu_platform = _load_xpu_platform(monkeypatch, use_triton_awq=False)
    xpu_platform.verify_quantization("awq")

    assert os.environ["VLLM_USE_TRITON_AWQ"] == "1"


def test_xpu_non_awq_does_not_force_triton_awq(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_TRITON_AWQ", "0")

    xpu_platform = _load_xpu_platform(monkeypatch, use_triton_awq=False)
    xpu_platform.verify_quantization("gptq")

    assert os.environ["VLLM_USE_TRITON_AWQ"] == "0"
