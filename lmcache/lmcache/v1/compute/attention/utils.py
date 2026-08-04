# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.compute.attention.abstract import AttentionInterface

logger = init_logger(__name__)


def _is_rocm() -> bool:
    """Check if we're running on ROCm (AMD GPU)."""
    return torch.version.hip is not None


def _flashinfer_available() -> bool:
    """Check if flashinfer is importable."""
    try:
        # Third Party
        import flashinfer  # noqa: F401

        return True
    except ImportError:
        return False


def infer_attn_backend_from_vllm(
    vllm_attn: "torch.nn.Module",
    enable_sparse: bool = False,
) -> "AttentionInterface":
    attn_name = type(vllm_attn.impl).__name__

    if enable_sparse:
        # On ROCm or when flashinfer is unavailable, use Triton backend
        if _is_rocm() or not _flashinfer_available():
            # Local
            from .triton_sparse import LMCTritonSparseBackend

            logger.info(
                "Using LMCTritonSparseBackend for CacheBlend "
                f"(ROCm={_is_rocm()}, flashinfer={_flashinfer_available()})"
            )
            return LMCTritonSparseBackend(vllm_attn)

        # On CUDA with flashinfer available, use flashinfer backend
        # Local
        from .flash_infer_sparse import LMCFlashInferSparseBackend

        if attn_name == "FlashInferImpl":
            return LMCFlashInferSparseBackend(vllm_attn)
        else:
            # Fallback to Triton even on CUDA if flashinfer impl doesn't match
            # Local
            from .triton_sparse import LMCTritonSparseBackend

            logger.info(
                f"Attention impl {attn_name} is not FlashInferImpl; "
                "falling back to LMCTritonSparseBackend"
            )
            return LMCTritonSparseBackend(vllm_attn)

    elif attn_name == "FlashAttentionImpl" and not enable_sparse:
        # Local
        from .flash_attn import LMCFlashAttnBackend

        return LMCFlashAttnBackend(vllm_attn)
    else:
        raise ValueError(
            f"Attention backend {attn_name} is not supported in LMCache. "
            f"enable_sparse={enable_sparse}"
        )


def infer_attn_func_from_vllm(device_type):
    if device_type == "cuda":
        # Third Party
        from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
    elif device_type == "xpu":
        # CUDA flash-attention extensions are unavailable on Intel XPU; use the
        # vLLM-shipped XPU flash-attention ops instead.
        # Third Party
        from vllm._xpu_ops import xpu_ops

        flash_attn_varlen_func = xpu_ops.flash_attn_varlen_func
        get_scheduler_metadata = xpu_ops.get_scheduler_metadata
    else:
        raise AssertionError(
            f"Unsupported device type '{device_type}' for flash attention backends"
        )

    return flash_attn_varlen_func, get_scheduler_metadata
