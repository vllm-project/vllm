# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def is_sycl_compress_unsupported_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return (
        "xpu_compress_insert_fp8mix" in msg
        and ("unsupported" in msg or "not implemented" in msg)
    )


def compress_norm_rope_store_sycl_fp8mix(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    compress_ratio: int,
    overlap: bool,
    rope_head_dim: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    token_stride: int,
    scale_dim: int,
) -> None:
    """SYCL launcher for DeepSeekV4 fp8mix cache insert on XPU.

    This path targets the sparse-attn layout (head=512, nope=448 fp8 + rope bf16).
    It writes the same paged cache layout as the Triton sparse kernel.
    """
    # Import at runtime to avoid hard dependency when XPU kernels are absent.
    import vllm_xpu_kernels._xpu_C  # noqa: F401

    if kv_cache.ndim != 3:
        raise ValueError(
            "SYCL fp8mix path expects kv_cache layout [num_blocks, block_size, 584], "
            f"got ndim={kv_cache.ndim}."
        )

    # xpu_compress_insert_fp8mix currently expects fp32 RMSNorm weight.
    if rms_norm_weight.dtype != torch.float32:
        rms_norm_weight = rms_norm_weight.float()

    torch.ops._xpu_C.xpu_compress_insert_fp8mix(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],
        compress_ratio,
        overlap,
        rope_head_dim,
        token_stride,
        scale_dim,
        int(kv_cache.stride(0)),
    )


def _fused_kv_compress_norm_rope_insert_sparse_attn_xpu(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    compress_ratio: int,
    overlap: bool,
    rope_head_dim: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    token_stride: int,
    scale_dim: int,
    head_dim: int,
    use_fp4_cache: bool,
) -> bool:
    """Handle XPU sparse-attn dispatch for head_dim=512.

    Returns True when SYCL path completes and caller should return early.
    Returns False when caller should continue with Triton launch.
    """
    # Values: sycl | triton (default: sycl).
    xpu_impl = os.getenv("VLLM_XPU_COMPRESS_IMPL", "sycl").strip().lower()
    if xpu_impl == "sycl":
        logger.info_once(
            "XPU fused KV insert path: sparse_attn_sycl (head_dim=%d)",
            head_dim,
        )
        strict_sycl = os.getenv("VLLM_XPU_COMPRESS_IMPL_STRICT", "0") == "1"
        try:
            compress_norm_rope_store_sycl_fp8mix(
                state_cache=state_cache,
                token_to_req_indices=token_to_req_indices,
                positions=positions,
                slot_mapping=slot_mapping,
                block_table=block_table,
                cos_sin_cache=cos_sin_cache,
                kv_cache=kv_cache,
                k_cache_metadata=k_cache_metadata,
                compress_ratio=compress_ratio,
                overlap=overlap,
                rope_head_dim=rope_head_dim,
                rms_norm_weight=rms_norm_weight,
                rms_norm_eps=rms_norm_eps,
                token_stride=token_stride,
                scale_dim=scale_dim,
            )
            return True
        except RuntimeError as exc:
            if strict_sycl or not is_sycl_compress_unsupported_error(exc):
                raise
            logger.warning_once(
                "SYCL sparse_attn path failed or unsupported; falling back to Triton "
                "(set VLLM_XPU_COMPRESS_IMPL_STRICT=1 to disable fallback)."
            )

    logger.info_once(
        "XPU fused KV insert path: sparse_attn_triton (head_dim=%d, use_fp4_cache=%s)",
        head_dim,
        use_fp4_cache,
    )
    return False
