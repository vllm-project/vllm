# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import Mock

import pytest
import torch

from vllm.attention.selector import _cached_get_attn_backend, get_attn_backend
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.openvino import OpenVinoPlatform
from vllm.platforms.rocm import RocmPlatform
from vllm.utils import STR_BACKEND_ENV_VAR, STR_FLASH_ATTN_VAL, STR_INVALID_VAL


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching.
    """
    _cached_get_attn_backend.cache_clear()


@pytest.mark.parametrize(
    "name", ["TORCH_SDPA", "ROCM_FLASH", "XFORMERS", "FLASHINFER", "OPENVINO"])
@pytest.mark.parametrize("use_v1", [True, False])
@pytest.mark.parametrize("device", ["cpu", "openvino", "hip", "cuda"])
def test_env(
    name: str,
    use_v1: bool,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the attention selector can be set via environment variable.
    Note that we do not test FlashAttn because it is the default backend.
    """

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1" if use_v1 else "0")
        m.setenv(STR_BACKEND_ENV_VAR, name)

        # Set up the platform based on device type
        platform_map = {
            "cpu": CpuPlatform(),
            "hip": RocmPlatform(),
            "openvino": OpenVinoPlatform(),
            "cuda": CudaPlatform()
        }

        # Set the current platform using monkeypatch
        m.setattr("vllm.attention.selector", "current_platform",
                  platform_map[device])

        # Mock openvino module if needed
        if device == "openvino":
            m.setitem(sys.modules, 'openvino', Mock())

        # Get the backend
        backend = get_attn_backend(16, torch.float16, torch.float16, 16, False)

        # Determine expected backend
        if device == "cpu":
            expected = "TORCH_SDPA"
        elif device == "hip":
            expected = "ROCM_ATTN_VLLM_V1" if use_v1 else "ROCM_FLASH"
        elif device == "openvino":
            expected = "OPENVINO"
        elif device == "cuda" and name in ["XFORMERS", "FLASHINFER"]:
            expected = "FLASH_ATTN_VLLM_V1" if use_v1 else name
        else:
            # Skip test for other combinations
            pytest.skip(f"Skipping test for device={device}, name={name}")

        # Assert the expected backend
        assert backend.get_name() == expected


def test_flash_attn(monkeypatch: pytest.MonkeyPatch):
    """Test FlashAttn validation."""
    # TODO: When testing for v1, pipe in `use_v1` as an argument to
    # get_attn_backend

    with monkeypatch.context() as m:
        m.setenv(STR_BACKEND_ENV_VAR, STR_FLASH_ATTN_VAL)

        # Unsupported CUDA arch
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda:
                            (7, 5))
        backend = get_attn_backend(16, torch.float16, None, 16, False)
        assert backend.get_name() != STR_FLASH_ATTN_VAL

        # Reset the monkeypatch for subsequent tests
        monkeypatch.undo()

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
        import sys
        original_module = sys.modules.get('vllm_flash_attn')
        monkeypatch.setitem(sys.modules, 'vllm_flash_attn', None)
        backend = get_attn_backend(16, torch.float16, None, 16, False)
        assert backend.get_name() != STR_FLASH_ATTN_VAL

        # Restore the original module if it existed
        if original_module is not None:
            monkeypatch.setitem(sys.modules, 'vllm_flash_attn',
                                original_module)
        else:
            monkeypatch.delitem(sys.modules, 'vllm_flash_attn', raising=False)

        # Unsupported head size
        backend = get_attn_backend(17, torch.float16, None, 16, False)
        assert backend.get_name() != STR_FLASH_ATTN_VAL

        # Attention-free models should bypass env and use PlaceholderAttention
        backend = get_attn_backend(16, torch.float16, torch.float16, 16, True)
        assert backend.get_name() != STR_FLASH_ATTN_VAL


@pytest.mark.parametrize("use_v1", [True, False])
def test_invalid_env(use_v1: bool, monkeypatch: pytest.MonkeyPatch):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1" if use_v1 else "0")
        m.setenv(STR_BACKEND_ENV_VAR, STR_INVALID_VAL)

        # Set the current platform to CUDA using monkeypatch
        monkeypatch.setattr("vllm.attention.selector", "current_platform",
                            CudaPlatform())

        # Test with head size 32
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
