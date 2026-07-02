# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the consumer-Blackwell CUDA toolchain/driver PTX mismatch guard.

`_vllm_fa2_C` ships no native cubin for sm_120/121 and falls back to PTX
JIT, which fails if vLLM was built with a newer CUDA toolkit than the
installed driver supports. `_check_flash_attn_ptx_compat` detects this
ahead of time; see https://github.com/vllm-project/vllm/issues/47397.
"""

from unittest.mock import patch

import pytest

import vllm.platforms.cuda as cuda_platform
from vllm.platforms.cuda import (
    _build_cuda_version,
    _check_flash_attn_ptx_compat,
    _driver_max_cuda_version,
)
from vllm.platforms.interface import DeviceCapability


def test_no_error_on_non_blackwell_consumer_gpu():
    """The guard only applies to sm_120/121; other capabilities are untouched
    even if build/driver versions would otherwise mismatch."""
    with (
        patch.object(cuda_platform, "_build_cuda_version", return_value=(13, 3)),
        patch.object(cuda_platform, "_driver_max_cuda_version", return_value=(13, 0)),
    ):
        _check_flash_attn_ptx_compat(DeviceCapability(9, 0))


@pytest.mark.parametrize(
    "device_capability", [DeviceCapability(12, 0), DeviceCapability(12, 1)]
)
def test_raises_on_build_newer_than_driver(device_capability):
    with (
        patch.object(cuda_platform, "_build_cuda_version", return_value=(13, 3)),
        patch.object(cuda_platform, "_driver_max_cuda_version", return_value=(13, 0)),
        pytest.raises(RuntimeError, match="cudaErrorUnsupportedPtxVersion"),
    ):
        _check_flash_attn_ptx_compat(device_capability)


def test_no_error_when_driver_covers_build():
    with (
        patch.object(cuda_platform, "_build_cuda_version", return_value=(13, 0)),
        patch.object(cuda_platform, "_driver_max_cuda_version", return_value=(13, 3)),
    ):
        _check_flash_attn_ptx_compat(DeviceCapability(12, 1))


def test_no_error_when_versions_match():
    with (
        patch.object(cuda_platform, "_build_cuda_version", return_value=(13, 0)),
        patch.object(cuda_platform, "_driver_max_cuda_version", return_value=(13, 0)),
    ):
        _check_flash_attn_ptx_compat(DeviceCapability(12, 1))


def test_no_error_when_build_version_undetermined():
    """If we can't tell what CUDA version compiled the extensions, don't
    guess -- skip the check rather than risk a false positive."""
    with (
        patch.object(cuda_platform, "_build_cuda_version", return_value=None),
        patch.object(cuda_platform, "_driver_max_cuda_version", return_value=(13, 0)),
    ):
        _check_flash_attn_ptx_compat(DeviceCapability(12, 1))


def test_no_error_when_driver_version_undetermined():
    with (
        patch.object(cuda_platform, "_build_cuda_version", return_value=(13, 3)),
        patch.object(cuda_platform, "_driver_max_cuda_version", return_value=None),
    ):
        _check_flash_attn_ptx_compat(DeviceCapability(12, 1))


def test_build_cuda_version_parses_local_version_suffix():
    with patch(
        "importlib.metadata.version",
        return_value="0.1.dev18154+g491f07501.cu133",
    ):
        assert _build_cuda_version() == (13, 3)


def test_build_cuda_version_falls_back_to_main_cuda_version():
    """setup.py's get_vllm_version() only appends a `cuXYZ` suffix when the
    build CUDA version differs from VLLM_MAIN_CUDA_VERSION -- a build against
    exactly the main version has no suffix at all."""
    with (
        patch("importlib.metadata.version", return_value="0.1.dev18154+g491f07501"),
        patch.object(cuda_platform.envs, "VLLM_MAIN_CUDA_VERSION", "13.0"),
    ):
        assert _build_cuda_version() == (13, 0)


def test_driver_max_cuda_version_handles_nvml_init_failure():
    """nvmlInit() failing (e.g. NVML unavailable) must degrade to None, not
    propagate -- this used to crash via the shared @with_nvml_context
    decorator, which doesn't guard nvmlInit() itself."""
    with patch.object(
        cuda_platform.pynvml,
        "nvmlInit",
        side_effect=cuda_platform.pynvml.NVMLError(999),
    ):
        assert _driver_max_cuda_version() is None


def test_driver_max_cuda_version_shuts_down_nvml_on_query_failure():
    with (
        patch.object(cuda_platform.pynvml, "nvmlInit") as mock_init,
        patch.object(
            cuda_platform.pynvml,
            "nvmlSystemGetCudaDriverVersion_v2",
            side_effect=cuda_platform.pynvml.NVMLError(999),
        ),
        patch.object(cuda_platform.pynvml, "nvmlShutdown") as mock_shutdown,
    ):
        assert _driver_max_cuda_version() is None
        mock_init.assert_called_once()
        mock_shutdown.assert_called_once()
