# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.rocm import RocmPlatform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import _cached_get_attn_backend, get_attn_backend


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching."""
    _cached_get_attn_backend.cache_clear()


# Define MLA and non-MLA backends separately
DEVICE_MLA_BACKENDS = {
    "cuda": [
        "TRITON_MLA",
        "FLASHMLA",
        "FLASHINFER_MLA",
        "FLASH_ATTN_MLA",
        "CUTLASS_MLA",
    ],
    "hip": ["TRITON_MLA", "ROCM_AITER_MLA"],
    "cpu": [],
}

DEVICE_REGULAR_ATTN_BACKENDS = {
    "cuda": ["FLASHINFER", "FLASH_ATTN"],
    "hip": ["ROCM_ATTN"],
    "cpu": ["CPU_ATTN"],
}

DEVICE_MLA_BLOCK_SIZES = {
    "cuda": [16, 64],  # CUDA supports both standard and extended block sizes
    "hip": [16, 1],  # HIP requires special handling for block_size=1
    # "cpu": [16]  # CPU uses fixed block size from test cases
    "cpu": [],  # FIXME(woosuk): Temporarily disable CPU tests
}


def generate_params():
    is_rocm = current_platform.is_rocm()
    params = []
    device_list = ["cuda", "cpu"] if not is_rocm else ["hip", "cpu"]
    for use_mla in [True, False]:
        for device in device_list:
            backends = (
                DEVICE_MLA_BACKENDS[device]
                if use_mla
                else DEVICE_REGULAR_ATTN_BACKENDS[device]
            )
            for name in backends:
                block_sizes = DEVICE_MLA_BLOCK_SIZES[device] if use_mla else [16]
                for block_size in block_sizes:
                    params.append(
                        pytest.param(
                            device,
                            name,
                            use_mla,
                            block_size,
                            id=f"{device}_{name}_mla_{str(use_mla)[0]}_blks{block_size}",
                        )
                    )
    return params


@pytest.mark.parametrize("device, name, use_mla, block_size", generate_params())
def test_backend_selection(
    device: str,
    name: str,
    use_mla: bool,
    block_size: int,
):
    """Test attention backend selection with valid device-backend pairs."""
    # Create AttentionConfig with the specified backend
    attention_config = AttentionConfig(backend=AttentionBackendEnum[name])
    vllm_config = VllmConfig(attention_config=attention_config)

    with set_current_vllm_config(vllm_config):
        if device == "cpu":
            with patch("vllm.platforms.current_platform", CpuPlatform()):
                backend = get_attn_backend(16, torch.float16, None, block_size)
            assert backend.get_name() == "CPU_ATTN"

        elif device == "hip":
            with patch("vllm.platforms.current_platform", RocmPlatform()):
                if use_mla:
                    # ROCm MLA backend logic:
                    # - TRITON_MLA: supported when block_size != 1
                    # - ROCM_AITER_MLA: supported when block_size == 1
                    # If backend is forced but doesn't match block_size,
                    # should raise ValueError

                    if name == "TRITON_MLA" and block_size == 1:
                        # TRITON_MLA doesn't support block_size == 1
                        with pytest.raises(ValueError) as exc_info:
                            get_attn_backend(
                                16, torch.float16, None, block_size, use_mla=use_mla
                            )
                        assert f"The selected backend, {name}" in str(exc_info.value)
                    else:
                        # Valid backend-block_size combination
                        backend = get_attn_backend(
                            16, torch.float16, None, block_size, use_mla=use_mla
                        )
                        expected = name
                        assert backend.get_name() == expected
                else:
                    backend = get_attn_backend(
                        16, torch.float16, None, block_size, use_mla=use_mla
                    )
                    expected = "ROCM_ATTN"
                    assert backend.get_name() == expected

        elif device == "cuda":
            with patch("vllm.platforms.current_platform", CudaPlatform()):
                capability = torch.cuda.get_device_capability()
                if use_mla:
                    # CUDA MLA backend logic:
                    # - CUTLASS_MLA: only supported with block_size == 128
                    #   and Blackwell GPUs (SM 10.x), V1 only
                    # - FLASHINFER_MLA: only supported on Blackwell GPUs
                    #   (SM 10.x), V1 only
                    # - FLASHMLA: only supported with block_size == 64
                    # - FLASH_ATTN_MLA: V1 only
                    # - TRITON_MLA: fallback for other cases

                    if name == "CUTLASS_MLA":
                        if block_size != 128:
                            # CUTLASS_MLA only supports block_size == 128
                            pytest.skip("CUTLASS_MLA only supports block_size 128")
                        if capability[0] != 10:
                            pytest.skip("CUTLASS MLA is not supported on this platform")
                        backend = get_attn_backend(
                            576, torch.float16, None, block_size, use_mla=use_mla
                        )
                        expected = "CUTLASS_MLA"
                        assert backend.get_name() == expected
                    elif name == "FLASHINFER_MLA":
                        if capability[0] != 10:
                            pytest.skip(
                                "FlashInfer MLA is not supported on this platform"
                            )
                        if block_size not in [32, 64]:
                            # FlashInfer MLA only supports block_size 32 or 64
                            pytest.skip(
                                "FlashInfer MLA only supports block_size 32 or 64"
                            )
                        backend = get_attn_backend(
                            576, torch.float16, None, block_size, use_mla=use_mla
                        )
                        expected = "FLASHINFER_MLA"
                        assert backend.get_name() == expected
                    elif name == "FLASHMLA":
                        if block_size != 64:
                            # FlashMLA only supports block_size == 64
                            pytest.skip("FlashMLA only supports block_size 64")
                        from vllm.v1.attention.backends.mla.flashmla import (
                            is_flashmla_dense_supported,
                        )

                        is_supported, _ = is_flashmla_dense_supported()
                        if not is_supported:
                            pytest.skip("FlashMLA not supported on this platform")
                        backend = get_attn_backend(
                            576,
                            torch.float16,
                            None,
                            block_size,
                            use_mla=use_mla,
                        )
                        expected = name
                        assert backend.get_name() == expected
                    elif name == "FLASH_ATTN_MLA":
                        from vllm.v1.attention.backends.fa_utils import (
                            flash_attn_supports_mla,
                        )

                        if not flash_attn_supports_mla():
                            pytest.skip(
                                "FlashAttention MLA not supported on this platform"
                            )
                        backend = get_attn_backend(
                            576, torch.float16, None, block_size, use_mla=use_mla
                        )
                        expected = "FLASH_ATTN_MLA"
                        assert backend.get_name() == expected
                    else:
                        # TRITON_MLA or other fallback
                        backend = get_attn_backend(
                            576, torch.float16, None, block_size, use_mla=use_mla
                        )
                        expected = "TRITON_MLA"
                        assert backend.get_name() == expected
                elif name == "FLASHINFER":
                    backend = get_attn_backend(
                        64, torch.float16, None, block_size, use_mla=use_mla
                    )
                    expected = "FLASHINFER"
                    assert backend.get_name() == expected
                elif name == "FLASH_ATTN":
                    backend = get_attn_backend(
                        32, torch.float16, None, block_size, use_mla=use_mla
                    )
                    expected = "FLASH_ATTN"
                    assert backend.get_name() == expected


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp32_fallback(device: str):
    """Test attention backend selection with fp32."""
    # Use default config (no backend specified)
    vllm_config = VllmConfig()

    with set_current_vllm_config(vllm_config):
        if device == "cpu":
            with patch("vllm.platforms.current_platform", CpuPlatform()):
                backend = get_attn_backend(16, torch.float32, None, 16)
            assert backend.get_name() == "CPU_ATTN"

        elif device == "cuda":
            with patch("vllm.platforms.current_platform", CudaPlatform()):
                backend = get_attn_backend(16, torch.float32, None, 16)
            assert backend.get_name() == "FLEX_ATTENTION"


def test_flash_attn(monkeypatch: pytest.MonkeyPatch):
    """Test FlashAttn validation."""
    pytest.skip(
        "Skipping as current backend selector does not "
        "handle fallbacks when a backend is explicitly set."
    )

    attention_config = AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN)
    vllm_config = VllmConfig(attention_config=attention_config)

    with set_current_vllm_config(vllm_config):
        # Unsupported CUDA arch
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _=None: (7, 5))
        backend = get_attn_backend(16, torch.float16, None, 16)
        assert backend.get_name() != "FLASH_ATTN"

        # Reset the monkeypatch for subsequent tests
        monkeypatch.undo()

        # Unsupported data type
        backend = get_attn_backend(16, torch.float8_e4m3fn, None, 16)
        assert backend.get_name() != "FLASH_ATTN"

        # Unsupported kv cache data type
        backend = get_attn_backend(16, torch.float16, "fp8", 16)
        assert backend.get_name() != "FLASH_ATTN"

        # Unsupported block size
        backend = get_attn_backend(16, torch.float16, None, 8)
        assert backend.get_name() != "FLASH_ATTN"

        # flash-attn is not installed
        import sys

        original_module = sys.modules.get("vllm_flash_attn")
        monkeypatch.setitem(sys.modules, "vllm_flash_attn", None)
        backend = get_attn_backend(16, torch.float16, None, 16)
        assert backend.get_name() != "FLASH_ATTN"

        # Restore the original module if it existed
        if original_module is not None:
            monkeypatch.setitem(sys.modules, "vllm_flash_attn", original_module)
        else:
            monkeypatch.delitem(sys.modules, "vllm_flash_attn", raising=False)

        # Unsupported head size
        backend = get_attn_backend(17, torch.float16, None, 16)
        assert backend.get_name() != "FLASH_ATTN"


def test_invalid_backend():
    """Test that invalid attention backend names raise ValueError."""
    with (
        pytest.raises(ValueError),
    ):
        # Invalid backend name should raise ValueError when creating enum
        AttentionConfig(backend=AttentionBackendEnum["INVALID"])
