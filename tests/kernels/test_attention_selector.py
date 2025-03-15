# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
import torch

from tests.kernels.utils import override_backend_env_variable
from vllm.attention.selector import _cached_get_attn_backend, get_attn_backend
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.openvino import OpenVinoPlatform
from vllm.platforms.rocm import RocmPlatform
from vllm.utils import STR_FLASH_ATTN_VAL, STR_INVALID_VAL


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching.
    """
    _cached_get_attn_backend.cache_clear()


@pytest.mark.parametrize(
    "name", ["TORCH_SDPA", "ROCM_FLASH", "XFORMERS", "FLASHINFER", "OPENVINO"])
@pytest.mark.parametrize("use_v1", [True, False])
@pytest.mark.parametrize("device", ["cpu", "openvino", "hip", "cuda"])
def test_env(name: str, use_v1: bool, device: str, monkeypatch):
    """Test that the attention selector can be set via environment variable.
    Note that we do not test FlashAttn because it is the default backend.
    """

    monkeypatch.setenv("VLLM_USE_V1", "1" if use_v1 else "0")
    override_backend_env_variable(monkeypatch, name)

    if device == "cpu":
        with patch("vllm.attention.selector.current_platform", CpuPlatform()):
            backend = get_attn_backend(16, torch.float16, torch.float16, 16,
                                       False)
        assert backend.get_name() == "TORCH_SDPA"
    elif device == "hip":
        with patch("vllm.attention.selector.current_platform", RocmPlatform()):
            backend = get_attn_backend(16, torch.float16, torch.float16, 16,
                                       False)
        EXPECTED = "ROCM_ATTN_VLLM_V1" if use_v1 else "ROCM_FLASH"
        assert backend.get_name() == EXPECTED
    elif device == "openvino":
        with patch("vllm.attention.selector.current_platform",
                   OpenVinoPlatform()), patch.dict('sys.modules',
                                                   {'openvino': Mock()}):
            backend = get_attn_backend(16, torch.float16, torch.float16, 16,
                                       False)
        assert backend.get_name() == "OPENVINO"
    else:
        if name in ["XFORMERS", "FLASHINFER"]:
            with patch("vllm.attention.selector.current_platform",
                       CudaPlatform()):
                backend = get_attn_backend(16, torch.float16, torch.float16,
                                           16, False)
            EXPECTED = "FLASH_ATTN_VLLM_V1" if use_v1 else name
            assert backend.get_name() == EXPECTED


def test_flash_attn(monkeypatch):
    """Test FlashAttn validation."""
    # TODO: When testing for v1, pipe in `use_v1` as an argument to
    # get_attn_backend

    override_backend_env_variable(monkeypatch, STR_FLASH_ATTN_VAL)

    # Unsupported CUDA arch
    with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
        backend = get_attn_backend(16, torch.float16, None, 16, False)
        assert backend.get_name() != STR_FLASH_ATTN_VAL

    # Unsupported data type
    backend = get_attn_backend(16, torch.float8_e4m3fn, None, 16, False)
    assert backend.get_name() != STR_FLASH_ATTN_VAL

    # Unsupported kv cache data type
    backend = get_attn_backend(16, torch.float16, "fp8", 16, False)
    assert backend.get_name() != STR_FLASH_ATTN_VAL

    # Unsupported block size
    backend = get_attn_backend(16, torch.float16, None, 8, False)
    assert backend.get_name() != STR_FLASH_ATTN_VAL

    # flash-attn is not installed
    with patch.dict('sys.modules', {'vllm_flash_attn': None}):
        backend = get_attn_backend(16, torch.float16, None, 16, False)
        assert backend.get_name() != STR_FLASH_ATTN_VAL

    # Unsupported head size
    backend = get_attn_backend(17, torch.float16, None, 16, False)
    assert backend.get_name() != STR_FLASH_ATTN_VAL

    # Attention-free models should bypass env and use PlaceholderAttention
    backend = get_attn_backend(16, torch.float16, torch.float16, 16, True)
    assert backend.get_name() != STR_FLASH_ATTN_VAL


@pytest.mark.parametrize("use_v1", [True, False])
def test_invalid_env(use_v1: bool, monkeypatch):
    """Ignore the invalid env variable if it is set."""
    monkeypatch.setenv("VLLM_USE_V1", "1" if use_v1 else "0")
    override_backend_env_variable(monkeypatch, STR_INVALID_VAL)

    with patch("vllm.attention.selector.current_platform", CudaPlatform()):
        backend = get_attn_backend(32, torch.float16, None, 16, False)
        EXPECTED = "FLASH_ATTN_VLLM_V1" if use_v1 else "FLASH_ATTN"
        assert backend.get_name() == EXPECTED

        # when block size == 16, backend will fall back to XFORMERS
        # this behavior is not yet supported on V1.
        if use_v1:
            # TODO: support fallback on V1!
            # https://github.com/vllm-project/vllm/issues/14524
            pass
        else:
            backend = get_attn_backend(16, torch.float16, None, 16, False)
            assert backend.get_name() == "XFORMERS"
