from unittest.mock import patch

import pytest
import torch

from tests.kernels.utils import override_backend_env_variable
from vllm.attention.selector import which_attn_to_use
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.openvino import OpenVinoPlatform
from vllm.platforms.rocm import RocmPlatform
from vllm.utils import STR_FLASH_ATTN_VAL, STR_INVALID_VAL


@pytest.mark.parametrize(
    "name", ["TORCH_SDPA", "ROCM_FLASH", "XFORMERS", "FLASHINFER", "OPENVINO"])
@pytest.mark.parametrize("device", ["cpu", "openvino", "hip", "cuda"])
def test_env(name: str, device: str, monkeypatch):
    """Test that the attention selector can be set via environment variable."""

    override_backend_env_variable(monkeypatch, name)

    if device == "cpu":
        with patch("vllm.attention.selector.current_platform", CpuPlatform()):
            backend = which_attn_to_use(16, torch.float16, torch.float16, 16,
                                        False)
        assert backend == "vllm.attention.backends.torch_sdpa.TorchSDPABackend"
    elif device == "hip":
        with patch("vllm.attention.selector.current_platform", RocmPlatform()):
            backend = which_attn_to_use(16, torch.float16, torch.float16, 16,
                                        False)
        assert backend == "vllm.attention.backends.rocm_flash_attn.ROCmFlashAttentionBackend"  # noqa: E501
    elif device == "openvino":
        with patch("vllm.attention.selector.current_platform",
                   OpenVinoPlatform()):
            backend = which_attn_to_use(16, torch.float16, torch.float16, 16,
                                        False)
        assert backend == "vllm.attention.backends.openvino.OpenVINOAttentionBackend"  # noqa: E501
    else:
        with patch("vllm.attention.selector.current_platform", CudaPlatform()):
            backend = which_attn_to_use(16, torch.float16, torch.float16, 16,
                                        False)
        if name == "FLASHINFER":
            assert backend == "vllm.attention.backends.flashinfer.FlashInferBackend"  # noqa: E501
        if name == "XFORMERS":
            assert backend == "vllm.attention.backends.xformers.XFormersBackend"
        else:
            assert backend == "vllm.attention.backends.flash_attn.FlashAttentionBackend"  # noqa: E501


def test_flash_attn(monkeypatch):
    """Test FlashAttn validation."""
    # TODO: When testing for v1, pipe in `use_v1` as an argument to
    # which_attn_to_use

    override_backend_env_variable(monkeypatch, STR_FLASH_ATTN_VAL)

    # Unsupported CUDA arch
    with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
        backend = which_attn_to_use(16, torch.float16, None, 16, False)
        assert backend != "vllm.attention.backends.flash_attn.FlashAttentionBackend"  # noqa: E501

    # Unsupported data type
    backend = which_attn_to_use(16, torch.float8_e4m3fn, None, 16, False)
    assert backend != "vllm.attention.backends.flash_attn.FlashAttentionBackend"

    # Unsupported kv cache data type
    backend = which_attn_to_use(16, torch.float16, "fp8", 16, False)
    assert backend != "vllm.attention.backends.flash_attn.FlashAttentionBackend"

    # Unsupported block size
    backend = which_attn_to_use(16, torch.float16, None, 8, False)
    assert backend != "vllm.attention.backends.flash_attn.FlashAttentionBackend"

    # flash-attn is not installed
    with patch.dict('sys.modules', {'vllm_flash_attn': None}):
        backend = which_attn_to_use(16, torch.float16, None, 16, False)
        assert backend != "vllm.attention.backends.flash_attn.FlashAttentionBackend"  # noqa: E501

    # Unsupported head size
    backend = which_attn_to_use(17, torch.float16, None, 16, False)
    assert backend != "vllm.attention.backends.flash_attn.FlashAttentionBackend"

    # Attention-free models should bypass env and use PlaceholderAttention
    backend = which_attn_to_use(16, torch.float16, torch.float16, 16, True)
    assert backend != "vllm.attention.backends.flash_attn.FlashAttentionBackend"


def test_invalid_env(monkeypatch):
    """Throw an exception if the backend name is invalid."""
    override_backend_env_variable(monkeypatch, STR_INVALID_VAL)
    with pytest.raises(ValueError):
        which_attn_to_use(16, torch.float16, None, 16, False)
