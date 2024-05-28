import os
from unittest.mock import patch

import pytest
import torch

from vllm.attention.selector import which_attn_to_use


@pytest.mark.parametrize(
    "name", ["TORCH_SDPA", "ROCM_FLASH", "XFORMERS", "FLASHINFER"])
@pytest.mark.parametrize("device", ["cpu", "hip", "cuda"])
def test_env(name: str, device: str):
    """Test that the attention selector can be set via environment variable.
    Note that we do not test FlashAttn because it is the default backend.
    """
    name_backup = os.environ.get("VLLM_ATTENTION_BACKEND", None)
    os.environ["VLLM_ATTENTION_BACKEND"] = name

    if device == "cpu":
        with patch("vllm.attention.selector.is_cpu", return_value=True):
            backend = which_attn_to_use(8, 16, 8, None, torch.float16,
                                        torch.float16, 16)
        assert backend.name == "TORCH_SDPA"
    elif device == "hip":
        with patch("vllm.attention.selector.is_hip", return_value=True):
            backend = which_attn_to_use(8, 16, 8, None, torch.float16,
                                        torch.float16, 16)
        assert backend.name == "ROCM_FLASH"
    else:
        backend = which_attn_to_use(8, 16, 8, None, torch.float16,
                                    torch.float16, 16)
        assert backend.name == name

    if name_backup is not None:
        os.environ["VLLM_ATTENTION_BACKEND"] = name_backup


def test_flash_attn():
    """Test FlashAttn validation."""
    name_backup = os.environ.get("VLLM_ATTENTION_BACKEND", None)
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

    # Unsupported CUDA arch
    with patch("torch.cuda.get_device_capability", return_value=[7, 5]):
        backend = which_attn_to_use(8, 16, 8, None, torch.float16, None, 16)
        assert backend.name != "FLASH_ATTN"

    # Unsupported data type
    backend = which_attn_to_use(8, 16, 8, None, torch.float8_e4m3fn, None, 16)
    assert backend.name != "FLASH_ATTN"

    # Unsupported kv cache data type
    backend = which_attn_to_use(8, 16, 8, None, torch.float16, "fp8", 16)
    assert backend.name != "FLASH_ATTN"

    # Unsupported block size
    backend = which_attn_to_use(8, 16, 8, None, torch.float16, None, 8)
    assert backend.name != "FLASH_ATTN"

    # Unsupported sliding window
    backend = which_attn_to_use(8, 16, 8, 1, torch.float16, None, 16)
    assert backend.name != "FLASH_ATTN"

    # flash-attn is not installed
    with patch.dict('sys.modules', {'vllm_flash_attn': None}):
        backend = which_attn_to_use(8, 16, 8, None, torch.float16, None, 16)
        assert backend.name != "FLASH_ATTN"

    # Unsupported head size
    backend = which_attn_to_use(8, 17, 8, None, torch.float16, None, 16)
    assert backend.name != "FLASH_ATTN"

    if name_backup is not None:
        os.environ["VLLM_ATTENTION_BACKEND"] = name_backup


def test_invalid_env():
    """Throw an exception if the backend name is invalid."""
    name_backup = os.environ.get("VLLM_ATTENTION_BACKEND", None)
    os.environ["VLLM_ATTENTION_BACKEND"] = "INVALID"
    with pytest.raises(ValueError):
        which_attn_to_use(8, 16, 8, None, torch.float16, None, 16)
    os.environ["VLLM_ATTENTION_BACKEND"] = name_backup
