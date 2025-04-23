# SPDX-License-Identifier: Apache-2.0

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
    "cuda": ["TRITON_MLA", "FLASHMLA"],
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
    "cpu": [16]  # CPU uses fixed block size from test cases
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


@pytest.mark.parametrize("device, name, use_mla, block_size",
                         generate_params())
@pytest.mark.parametrize("use_v1", [True, False])
def test_env(
    device: str,
    name: str,
    use_mla: bool,
    block_size: int,
    use_v1: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test attention backend selection with valid device-backend pairs."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1" if use_v1 else "0")
        m.setenv(STR_BACKEND_ENV_VAR, name)
        m.setenv("VLLM_MLA_DISABLE", "1" if use_mla else "0")

        if device == "cpu":
            with patch("vllm.attention.selector.current_platform",
                       CpuPlatform()):
                backend = get_attn_backend(16, torch.float16, torch.float16,
                                           block_size, False)
            assert backend.get_name() == "TORCH_SDPA"

        elif device == "hip":
            with patch("vllm.attention.selector.current_platform",
                       RocmPlatform()):
                if use_mla:
                    # Validate HIP MLA backend-block_size combinations
                    valid_combination = (
                        (name == "TRITON_MLA" and block_size != 1)
                        or (name == "ROCM_AITER_MLA" and block_size == 1))

                    if valid_combination:
                        backend = get_attn_backend(16,
                                                   torch.float16,
                                                   torch.float16,
                                                   block_size,
                                                   False,
                                                   use_mla=use_mla)
                        assert backend.get_name() == name
                    else:
                        with pytest.raises(ValueError) as exc_info:
                            get_attn_backend(16,
                                             torch.float16,
                                             torch.float16,
                                             block_size,
                                             False,
                                             use_mla=use_mla)
                        assert f"The selected backend, {name}" in str(
                            exc_info.value)
                else:
                    backend = get_attn_backend(16,
                                               torch.float16,
                                               torch.float16,
                                               block_size,
                                               False,
                                               use_mla=use_mla)
                    expected = "TRITON_ATTN_VLLM_V1" if use_v1 else "ROCM_FLASH"
                    assert backend.get_name() == expected

        elif device == "cuda":
            with patch("vllm.attention.selector.current_platform",
                       CudaPlatform()):
                if use_mla:
                    if name == "FLASHMLA" and block_size == 64:
                        from vllm.attention.backends.flashmla import (
                            is_flashmla_supported)

                        # only on cuda platforms with specific capability.
                        is_supported, _ = is_flashmla_supported()

                        if not is_supported:
                            # if platform is not supported then skip this case.
                            pytest.skip()
                        else:
                            backend = get_attn_backend(16,
                                                       torch.float16,
                                                       torch.float16,
                                                       block_size,
                                                       False,
                                                       use_mla=use_mla)
                            expected = f"{name}_VLLM_V1" if use_v1 else name
                            assert backend.get_name() == expected
                    else:
                        backend = get_attn_backend(16,
                                                   torch.float16,
                                                   torch.float16,
                                                   block_size,
                                                   False,
                                                   use_mla=use_mla)
                        expected = ("TRITON_MLA_VLLM_V1"
                                    if use_v1 else "TRITON_MLA")
                        assert backend.get_name() == expected
                elif name == "FLASHINFER":
                    backend = get_attn_backend(16,
                                               torch.float16,
                                               torch.float16,
                                               block_size,
                                               False,
                                               use_mla=use_mla)
                    expected = "FLASHINFER_VLLM_V1" if use_v1 else name
                    assert backend.get_name() == expected
                else:
                    backend = get_attn_backend(16,
                                               torch.float16,
                                               torch.float16,
                                               block_size,
                                               False,
                                               use_mla=use_mla)
                    expected = "FLASH_ATTN_VLLM_V1" if use_v1 else name
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

    with monkeypatch.context() as m, patch(
            "vllm.attention.selector.current_platform", CudaPlatform()):
        m.setenv("VLLM_USE_V1", "1" if use_v1 else "0")
        m.setenv(STR_BACKEND_ENV_VAR, STR_INVALID_VAL)

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
