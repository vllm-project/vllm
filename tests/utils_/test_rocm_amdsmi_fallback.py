# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import types

import pytest
import torch

GPU_NAME = "AMD Radeon RX 9060 XT"


class _DeviceProperties:
    gcnArchName = "gfx1200"


def _amdsmi_init_fails() -> None:
    # What amdsmi raises on WSL2 (https://github.com/vllm-project/vllm/issues/64664).
    raise RuntimeError("AmdSmiLibraryException(34)")


@pytest.fixture
def rocm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    # Private copy, because the import-time GPU arch query is stubbed and must not
    # leak into sys.modules for other tests.
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *_: _DeviceProperties()
    )
    spec = importlib.util.find_spec("vllm.platforms.rocm")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # raising=False because amdsmi_init does not exist on hosts without amdsmi.
    monkeypatch.setattr(module, "amdsmi_init", _amdsmi_init_fails, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: GPU_NAME)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return module


def test_get_device_name_falls_back_to_torch(rocm: types.ModuleType) -> None:
    assert rocm.RocmPlatform.get_device_name(0) == GPU_NAME


@pytest.mark.parametrize(
    ("visible_devices", "expected"), [(None, "rocm:0"), ("1", "rocm:1")]
)
def test_get_device_uuid_falls_back_to_physical_id(
    rocm: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    visible_devices: str | None,
    expected: str,
) -> None:
    if visible_devices is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    assert rocm.RocmPlatform.get_device_uuid(0) == expected


def test_is_fully_connected_is_false_without_amdsmi(rocm: types.ModuleType) -> None:
    assert not rocm.RocmPlatform.is_fully_connected([0, 1])
