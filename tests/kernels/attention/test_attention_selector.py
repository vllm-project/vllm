# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm.attention.selector import _cached_get_attn_backend, get_attn_backend
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.rocm import RocmPlatform
from vllm.utils import STR_BACKEND_ENV_VAR, STR_FLASH_ATTN_VAL, STR_INVALID_VAL


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching.
    """
    _cached_get_attn_backend.cache_clear()


# Define MLA and non-MLA backends separately
DEVICE_MLA_BACKENDS = {
    "cuda": [
        "TRITON_MLA", "FLASHMLA", "FLASHINFER_MLA", "FLASH_ATTN_MLA",
        "CUTLASS_MLA"
    ],
    "hip": ["TRITON_MLA", "ROCM_AITER_MLA"],
    "cpu": [],
}

DEVICE_REGULAR_ATTN_BACKENDS = {
    "cuda": ["XFORMERS", "FLASHINFER"],
    "hip": ["ROCM_FLASH"],
    "cpu": ["TORCH_SDPA"],
}

DEVICE_MLA_BLOCK_SIZES = {
    "cuda": [16, 64],  # CUDA supports both standard and extended block sizes
    "hip": [16, 1],  # HIP requires special handling for block_size=1
    # "cpu": [16]  # CPU uses fixed block size from test cases
    "cpu": []  # FIXME(woosuk): Temporarily disable CPU tests
}


def generate_params():
    params = []
    for use_mla in [True, False]:
        for device in ["cuda", "hip", "cpu"]:
            backends = DEVICE_MLA_BACKENDS[
                device] if use_mla else DEVICE_REGULAR_ATTN_BACKENDS[device]
            for name in backends:
                block_sizes = DEVICE_MLA_BLOCK_SIZES[device] if use_mla else [
                    16
                ]
                for block_size in block_sizes:
                    params.append(
                        pytest.param(
                            device,
                            name,
                            use_mla,
                            block_size,
                            id=
                            f"{device}_{name}_mla_{str(use_mla)[0]}_blks{block_size}"
                        ))
    return params


@pytest.mark.skip(reason="Skipped for now. Should be revisited.")
@pytest.mark.parametrize("device, name, use_mla, block_size",
                         generate_params())
def test_env(
    device: str,
    name: str,
    use_mla: bool,
    block_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test attention backend selection with valid device-backend pairs."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv(STR_BACKEND_ENV_VAR, name)
        m.setenv("VLLM_MLA_DISABLE", "1" if use_mla else "0")

        if device == "cpu":
            with patch("vllm.attention.selector.current_platform",
                       CpuPlatform()):
                backend = get_attn_backend(16, torch.float16, None, block_size,
                                           False)
            assert backend.get_name() == "TORCH_SDPA_VLLM_V1"

        elif device == "hip":
            with patch("vllm.attention.selector.current_platform",
                       RocmPlatform()):
                if use_mla:
                    # ROCm MLA backend logic:
                    # - TRITON_MLA: supported when block_size != 1
                    # - ROCM_AITER_MLA: supported when block_size == 1
                    # If backend is forced but doesn't match block_size,
                    # should raise ValueError

                    if name == "TRITON_MLA" and block_size == 1:
                        # TRITON_MLA doesn't support block_size == 1
                        with pytest.raises(ValueError) as exc_info:
                            get_attn_backend(16,
                                             torch.float16,
                                             None,
                                             block_size,
                                             False,
                                             use_mla=use_mla)
                        assert f"The selected backend, {name}" in str(
                            exc_info.value)
                    elif name == "ROCM_AITER_MLA" and block_size != 1:
                        # ROCM_AITER_MLA only supports block_size == 1
                        with pytest.raises(ValueError) as exc_info:
                            get_attn_backend(16,
                                             torch.float16,
                                             None,
                                             block_size,
                                             False,
                                             use_mla=use_mla)
                        assert f"The selected backend, {name}" in str(
                            exc_info.value)
                    else:
                        # Valid backend-block_size combination
                        backend = get_attn_backend(16,
                                                   torch.float16,
                                                   None,
                                                   block_size,
                                                   False,
                                                   use_mla=use_mla)
                        expected = f"{name}_VLLM_V1"
                        assert backend.get_name() == expected
                else:
                    backend = get_attn_backend(16,
                                               torch.float16,
                                               None,
                                               block_size,
                                               False,
                                               use_mla=use_mla)
                    expected = "TRITON_ATTN_VLLM_V1"
                    assert backend.get_name() == expected

        elif device == "cuda":
            with patch("vllm.attention.selector.current_platform",
                       CudaPlatform()):
                if use_mla:
                    # CUDA MLA backend logic:
                    # - CUTLASS_MLA: only supported with block_size == 128
                    #   and Blackwell GPUs (SM 10.0), V1 only
                    # - FLASHINFER_MLA: only supported on Blackwell GPUs
                    #   (SM 10.0+), V1 only
                    # - FLASHMLA: only supported with block_size == 64
                    # - FLASH_ATTN_MLA: V1 only
                    # - TRITON_MLA: fallback for other cases

                    if name == "CUTLASS_MLA":
                        if block_size != 128:
                            # CUTLASS_MLA only supports block_size == 128
                            pytest.skip(
                                "CUTLASS_MLA only supports block_size 128")
                        else:
                            backend = get_attn_backend(16,
                                                       torch.float16,
                                                       None,
                                                       block_size,
                                                       False,
                                                       use_mla=use_mla)
                            expected = "CUTLASS_MLA_VLLM_V1"
                            assert backend.get_name() == expected
                    elif name == "FLASHINFER_MLA":
                        if block_size not in [32, 64]:
                            # FlashInfer MLA only supports block_size 32 or 64
                            pytest.skip(
                                "FlashInfer MLA only supports block_size 32 "
                                "or 64")
                        else:
                            backend = get_attn_backend(16,
                                                       torch.float16,
                                                       None,
                                                       block_size,
                                                       False,
                                                       use_mla=use_mla)
                            expected = "FLASHINFER_MLA"
                            assert backend.get_name() == expected
                    elif name == "FLASHMLA":
                        if block_size != 64:
                            # FlashMLA only supports block_size == 64
                            pytest.skip("FlashMLA only supports block_size 64")
                        else:
                            from vllm.attention.backends.flashmla import (
                                is_flashmla_supported)
                            is_supported, _ = is_flashmla_supported()
                            if not is_supported:
                                pytest.skip(
                                    "FlashMLA not supported on this platform")
                            else:
                                backend = get_attn_backend(16,
                                                           torch.float16,
                                                           None,
                                                           block_size,
                                                           False,
                                                           use_mla=use_mla)
                                expected = f"{name}_VLLM_V1"
                                assert backend.get_name() == expected
                    elif name == "FLASH_ATTN_MLA":
                        backend = get_attn_backend(16,
                                                   torch.float16,
                                                   None,
                                                   block_size,
                                                   False,
                                                   use_mla=use_mla)
                        expected = "FLASH_ATTN_MLA"
                        assert backend.get_name() == expected
                    else:
                        # TRITON_MLA or other fallback
                        backend = get_attn_backend(16,
                                                   torch.float16,
                                                   None,
                                                   block_size,
                                                   False,
                                                   use_mla=use_mla)
                        expected = "TRITON_MLA_VLLM_V1"
                        assert backend.get_name() == expected
                elif name == "FLASHINFER":
                    backend = get_attn_backend(16,
                                               torch.float16,
                                               None,
                                               block_size,
                                               False,
                                               use_mla=use_mla)
                    expected = "FLASHINFER_VLLM_V1"
                    assert backend.get_name() == expected
                else:
                    backend = get_attn_backend(32,
                                               torch.float16,
                                               None,
                                               block_size,
                                               False,
                                               use_mla=use_mla)
                    expected = "FLASH_ATTN_VLLM_V1"
                    assert backend.get_name() == expected

                    backend = get_attn_backend(16,
                                               torch.float16,
                                               None,
                                               block_size,
                                               False,
                                               use_mla=use_mla)
                    assert backend.get_name() == "FLEX_ATTENTION", (
                        "Should fallback to FlexAttention if head size is "
                        "not supported by FlashAttention")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp32_fallback(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test attention backend selection with fp32."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        if device == "cpu":
            with patch("vllm.attention.selector.current_platform",
                       CpuPlatform()):
                backend = get_attn_backend(16, torch.float32, None, 16, False)
            assert backend.get_name() == "TORCH_SDPA_VLLM_V1"

        elif device == "cuda":
            with patch("vllm.attention.selector.current_platform",
                       CudaPlatform()):
                backend = get_attn_backend(16, torch.float32, None, 16, False)
            assert backend.get_name() == "FLEX_ATTENTION"


def test_flash_attn(monkeypatch: pytest.MonkeyPatch):
    """Test FlashAttn validation."""
    # TODO: When testing for v1, pipe in `use_v1` as an argument to
    # get_attn_backend

    with monkeypatch.context() as m:
        m.setenv(STR_BACKEND_ENV_VAR, STR_FLASH_ATTN_VAL)

        # Unsupported CUDA arch
        monkeypatch.setattr(torch.cuda,
                            "get_device_capability",
                            lambda _=None: (7, 5))
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
        backend = get_attn_backend(16, torch.float16, None, 16, True)
        assert backend.get_name() != STR_FLASH_ATTN_VAL


def test_invalid_env(monkeypatch: pytest.MonkeyPatch):
    """Test that invalid attention backend names raise ValueError."""
    with monkeypatch.context() as m, patch(
            "vllm.attention.selector.current_platform", CudaPlatform()):
        m.setenv("VLLM_USE_V1", "1")
        m.setenv(STR_BACKEND_ENV_VAR, STR_INVALID_VAL)

        # Should raise ValueError for invalid backend
        with pytest.raises(ValueError) as exc_info:
            get_attn_backend(32, torch.float16, None, 16, False)
        assert "Invalid value 'INVALID'" in str(exc_info.value)
