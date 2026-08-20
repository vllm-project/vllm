# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for error handling in the CUDA attention backend probe.

Environment-shaped probe failures (missing packages, unreadable caches,
broken driver installs) must mark the backend unavailable rather than
terminating engine init; programming errors must still propagate.
See https://github.com/vllm-project/vllm/issues/51658.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-specific tests"
)

SELECTOR_CONFIG = AttentionSelectorConfig(
    head_size=64,
    dtype=torch.float16,
    kv_cache_dtype=None,
    block_size=16,
)

SM90 = DeviceCapability(major=9, minor=0)


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("flashinfer is not installed"),
        PermissionError(13, "Permission denied", "/root/.cache/flashinfer"),
        OSError("libcuda.so.1: cannot open shared object file"),
    ],
)
def test_get_valid_backends_records_environment_failure(exc):
    with patch("vllm.platforms.cuda._get_attn_backend_class", side_effect=exc):
        valid, invalid_reasons = CudaPlatform.get_valid_backends(
            device_capability=SM90,
            attn_selector_config=SELECTOR_CONFIG,
            num_heads=32,
        )
    assert valid == []
    assert invalid_reasons
    for _priority, reasons in invalid_reasons.values():
        assert reasons == [f"{type(exc).__name__}: {exc}"]


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("CUDA error: no kernel image is available"),
        AttributeError("module has no attribute 'get_builder_cls'"),
        KeyboardInterrupt(),
    ],
)
def test_get_valid_backends_propagates_unexpected_errors(exc):
    with (
        patch("vllm.platforms.cuda._get_attn_backend_class", side_effect=exc),
        pytest.raises(type(exc)),
    ):
        CudaPlatform.get_valid_backends(
            device_capability=SM90,
            attn_selector_config=SELECTOR_CONFIG,
            num_heads=32,
        )


def test_get_valid_backends_keeps_probing_after_failure():
    healthy = MagicMock()
    healthy.validate_configuration.return_value = []
    side_effects = [OSError("probe failed")] + [healthy] * 32

    with patch("vllm.platforms.cuda._get_attn_backend_class", side_effect=side_effects):
        valid, invalid_reasons = CudaPlatform.get_valid_backends(
            device_capability=SM90,
            attn_selector_config=SELECTOR_CONFIG,
            num_heads=32,
        )
    assert len(invalid_reasons) == 1
    assert valid


def test_selected_backend_probe_failure_raises_value_error_with_cause():
    exc = OSError("libcuda.so.1: cannot open shared object file")
    with (
        patch("vllm.platforms.cuda._get_attn_backend_class", side_effect=exc),
        patch.object(CudaPlatform, "get_device_capability", return_value=SM90),
        pytest.raises(ValueError, match="OSError") as excinfo,
    ):
        CudaPlatform.get_attn_backend_cls(
            selected_backend=AttentionBackendEnum.FLASH_ATTN,
            attn_selector_config=SELECTOR_CONFIG,
            num_heads=32,
        )
    assert excinfo.value.__cause__ is exc
