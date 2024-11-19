from unittest.mock import patch

import pytest
import torch

from tests.kernels.utils import override_backend_env_variable
from vllm.attention.selector import which_attn_to_use
from vllm.platforms import cpu, cuda, openvino, rocm
from vllm.utils import STR_FLASH_ATTN_VAL, STR_INVALID_VAL


@pytest.mark.parametrize(
    "name", ["TORCH_SDPA", "ROCM_FLASH", "XFORMERS", "FLASHINFER", "OPENVINO"])
@pytest.mark.parametrize("device", ["cpu", "openvino", "hip", "cuda"])
def test_env(name: str, device: str, monkeypatch):
    """Test that the attention selector can be set via environment variable.
    Note that we do not test FlashAttn because it is the default backend.
    """

    override_backend_env_variable(monkeypatch, name)

    if device == "cpu":
        with patch("vllm.attention.selector.current_platform",
                   cpu.CpuPlatform()):
            backend = which_attn_to_use(16, torch.float16, torch.float16, 16,
                                        False)
        assert backend.name == "TORCH_SDPA"
    elif device == "hip":
        with patch("vllm.attention.selector.current_platform",
                   rocm.RocmPlatform()):
            backend = which_attn_to_use(16, torch.float16, torch.float16, 16,
                                        False)
        assert backend.name == "ROCM_FLASH"
    elif device == "openvino":
        with patch("vllm.attention.selector.current_platform",
                   openvino.OpenVinoPlatform()):
            backend = which_attn_to_use(16, torch.float16, torch.float16, 16,
                                        False)
        assert backend.name == "OPENVINO"
    else:
        with patch("vllm.attention.selector.current_platform",
                   cuda.CudaPlatform()):
            backend = which_attn_to_use(16, torch.float16, torch.float16, 16,
                                        False)
        assert backend.name == name


def test_flash_attn(monkeypatch):
    """Test FlashAttn validation."""
    # TODO: When testing for v1, pipe in `use_v1` as an argument to
    # which_attn_to_use

    override_backend_env_variable(monkeypatch, STR_FLASH_ATTN_VAL)

    # Unsupported CUDA arch
    with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
        backend = which_attn_to_use(16, torch.float16, None, 16, False)
        assert backend.name != STR_FLASH_ATTN_VAL

    # Unsupported data type
    backend = which_attn_to_use(16, torch.float8_e4m3fn, None, 16, False)
    assert backend.name != STR_FLASH_ATTN_VAL

    # Unsupported kv cache data type
    backend = which_attn_to_use(16, torch.float16, "fp8", 16, False)
    assert backend.name != STR_FLASH_ATTN_VAL

    # Unsupported block size
    backend = which_attn_to_use(16, torch.float16, None, 8, False)
    assert backend.name != STR_FLASH_ATTN_VAL

    # flash-attn is not installed
    with patch.dict('sys.modules', {'vllm_flash_attn': None}):
        backend = which_attn_to_use(16, torch.float16, None, 16, False)
        assert backend.name != STR_FLASH_ATTN_VAL

    # Unsupported head size
    backend = which_attn_to_use(17, torch.float16, None, 16, False)
    assert backend.name != STR_FLASH_ATTN_VAL

    # Attention-free models should bypass env and use PlaceholderAttention
    backend = which_attn_to_use(16, torch.float16, torch.float16, 16, True)
    assert backend.name != STR_FLASH_ATTN_VAL


def test_invalid_env(monkeypatch):
    """Throw an exception if the backend name is invalid."""
    override_backend_env_variable(monkeypatch, STR_INVALID_VAL)
    with pytest.raises(ValueError):
        which_attn_to_use(16, torch.float16, None, 16, False)
