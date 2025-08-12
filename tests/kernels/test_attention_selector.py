from unittest.mock import patch

import pytest
import torch

from tests.kernels.utils import (STR_FLASH_ATTN_VAL, STR_INVALID_VAL,
                                 override_backend_env_variable)
from vllm.attention.selector import which_attn_to_use


@pytest.mark.parametrize(
    "name", ["TORCH_SDPA", "ROCM_FLASH", "XFORMERS", "FLASHINFER", "OPENVINO"])
@pytest.mark.parametrize("device", ["cpu", "openvino", "hip", "cuda"])
def test_env(name: str, device: str, monkeypatch):
    """Test that the attention selector can be set via environment variable.
    Note that we do not test FlashAttn because it is the default backend.
    """

    override_backend_env_variable(monkeypatch, name)

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
    elif device == "openvino":
        with patch("vllm.attention.selector.is_openvino", return_value=True):
            backend = which_attn_to_use(8, 16, 8, None, torch.float16,
                                        torch.float16, 16)
        assert backend.name == "OPENVINO"
    else:
        backend = which_attn_to_use(8, 16, 8, None, torch.float16,
                                    torch.float16, 16)
        assert backend.name == name


def test_flash_attn(monkeypatch):
    """Test FlashAttn validation."""

    override_backend_env_variable(monkeypatch, STR_FLASH_ATTN_VAL)

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


def test_invalid_env(monkeypatch):
    """Throw an exception if the backend name is invalid."""
    override_backend_env_variable(monkeypatch, STR_INVALID_VAL)
    with pytest.raises(ValueError):
        which_attn_to_use(8, 16, 8, None, torch.float16, None, 16)
