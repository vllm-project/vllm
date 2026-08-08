# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_device_diagnostics_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "vllm"
        / "config"
        / "device_diagnostics.py"
    )
    spec = spec_from_file_location("device_diagnostics_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_failed_device_type_message_includes_runtime_summary():
    module = _load_device_diagnostics_module()

    message = module.build_failed_device_type_message(
        None,
        None,
        environment_looks_like_rocm=False,
    )

    assert "Failed to infer device type." in message
    assert "torch.version.cuda=None, torch.version.hip=None" in message
    assert "VLLM_LOGGING_LEVEL=DEBUG" in message


def test_build_failed_device_type_message_flags_cuda_runtime_on_rocm():
    module = _load_device_diagnostics_module()

    message = module.build_failed_device_type_message(
        "12.8",
        None,
        environment_looks_like_rocm=True,
    )

    assert "torch.version.cuda=12.8, torch.version.hip=None" in message
    assert "reports CUDA instead of HIP" in message
    assert "CUDA wheel was installed" in message


def test_environment_looks_like_rocm_accepts_env_or_amdsmi_signal():
    module = _load_device_diagnostics_module()

    assert module.environment_looks_like_rocm(
        {"ROCM_PATH": "/opt/rocm"},
        amdsmi_present=False,
    )
    assert module.environment_looks_like_rocm({}, amdsmi_present=True)
    assert not module.environment_looks_like_rocm({}, amdsmi_present=False)
