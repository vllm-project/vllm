import enum
from functools import lru_cache
from typing import Optional, Type

import torch

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.utils import is_cpu, is_hip

logger = init_logger(__name__)

_ATTN_BACKEND: Optional[Type[AttentionBackend]] = None


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    XFORMERS = enum.auto()
    ROCM_FLASH = enum.auto()
    TORCH_SDPA = enum.auto()
    FLASHINFER = enum.auto()


def get_attn_backend(
    dtype: Optional[torch.dtype] = None,
    kv_cache_dtype: Optional[str] = None,
) -> Type[AttentionBackend]:
    """Returns the attention backend to use.

    For the first call, the backend is selected based on the dtype and
    kv_cache_dtype. The selected backend is cached for subsequent calls.
    """
    global _ATTN_BACKEND
    if dtype is None:
        assert kv_cache_dtype is None, "KV cache dtype should be None."
        assert _ATTN_BACKEND is not None, "Attention backend is not set."
        return _ATTN_BACKEND
    else:
        assert kv_cache_dtype is not None, "KV cache dtype is not set."
        _ATTN_BACKEND = select_attn_backend(dtype, kv_cache_dtype)
        return _ATTN_BACKEND


def select_attn_backend(
    dtype: torch.dtype,
    kv_cache_dtype: str,
) -> Type[AttentionBackend]:
    backend = _which_attn_to_use(dtype, kv_cache_dtype)
    if backend == _Backend.FLASH_ATTN:
        logger.info("Using FlashAttention-2 backend.")
        from vllm.attention.backends.flash_attn import (  # noqa: F401
            FlashAttentionBackend)
        return FlashAttentionBackend
    elif backend == _Backend.XFORMERS:
        logger.info("Using XFormers backend.")
        from vllm.attention.backends.xformers import (  # noqa: F401
            XFormersBackend)
        return XFormersBackend
    elif backend == _Backend.ROCM_FLASH:
        logger.info("Using ROCmFlashAttention backend.")
        from vllm.attention.backends.rocm_flash_attn import (  # noqa: F401
            ROCmFlashAttentionBackend)
        return ROCmFlashAttentionBackend
    elif backend == _Backend.TORCH_SDPA:
        logger.info("Using Torch SDPA backend.")
        from vllm.attention.backends.torch_sdpa import TorchSDPABackend
        return TorchSDPABackend
    elif backend == _Backend.FLASHINFER:
        logger.info("Using Flashinfer backend.")
        logger.warning("Eager mode is enforced for the Flashinfer backend.")
        from vllm.attention.backends.flashinfer import FlashInferBackend
        return FlashInferBackend
    else:
        raise ValueError("Invalid attention backend.")


@lru_cache(maxsize=None)
def _which_attn_to_use(
    dtype: torch.dtype,
    kv_cache_dtype: str,
) -> _Backend:
    """Returns which flash attention backend to use."""
    if is_cpu():
        return _Backend.TORCH_SDPA

    if is_hip():
        # AMD GPUs.
        if torch.cuda.get_device_capability()[0] != 9:
            # not Instinct series GPUs.
            logger.info("flash_atten is not supported on NAVI GPUs.")
        return _Backend.ROCM_FLASH

    # NVIDIA GPUs.
    if torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
        logger.info("Cannot use FlashAttention-2 backend for Volta and Turing "
                    "GPUs.")
        return _Backend.XFORMERS

    if dtype not in (torch.float16, torch.bfloat16):
        logger.info("Cannot use FlashAttention-2 backend for dtype other than "
                    "torch.float16 or torch.bfloat16.")
        return _Backend.XFORMERS

    if kv_cache_dtype.startswith("fp8"):
        logger.info("Cannot use FlashAttention-2 backend for FP8 KV cache.")
        return _Backend.XFORMERS

    try:
        import vllm_flash_attn  # noqa: F401
    except ImportError:
        logger.info(
            "Cannot use FlashAttention backend because the vllm_flash_attn "
            "package is not found. `pip install vllm-flash-attn` for better "
            "performance.")
        return _Backend.XFORMERS

    backend_by_env_var = envs.VLLM_ATTENTION_BACKEND
    if backend_by_env_var is not None:
        return _Backend[backend_by_env_var]

    # Default case.
    return _Backend.FLASH_ATTN
