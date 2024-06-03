from unittest.mock import patch

import pytest
import torch

from tests.kernels.utils import backend_override_fixture
from vllm.attention.selector import which_attn_to_use


@pytest.mark.parametrize(
    "name", ["TORCH_SDPA", "ROCM_FLASH", "XFORMERS", "FLASHINFER"])
@pytest.mark.parametrize("device", ["cpu", "hip", "cuda"])
def test_env(name: str, device: str):
    """Test that the attention selector can be set via environment variable.
    Note that we do not test FlashAttn because it is the default backend.
    """

    with backend_override_fixture(name):

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


def test_flash_attn():
    """Test FlashAttn validation."""

    with backend_override_fixture("FLASH_ATTN"):

        # Unsupported CUDA arch
        with patch("torch.cuda.get_device_capability", return_value=[7, 5]):
            backend = which_attn_to_use(8, 16, 8, None, torch.float16, None,
                                        16)
            assert backend.name != "FLASH_ATTN"

        # Unsupported data type
        backend = which_attn_to_use(8, 16, 8, None, torch.float8_e4m3fn, None,
                                    16)
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
            backend = which_attn_to_use(8, 16, 8, None, torch.float16, None,
                                        16)
            assert backend.name != "FLASH_ATTN"

        # Unsupported head size
        backend = which_attn_to_use(8, 17, 8, None, torch.float16, None, 16)
        assert backend.name != "FLASH_ATTN"


def test_invalid_env():
    """Throw an exception if the backend name is invalid."""

    with backend_override_fixture("INVALID"), pytest.raises(ValueError):
        which_attn_to_use(8, 16, 8, None, torch.float16, None, 16)
