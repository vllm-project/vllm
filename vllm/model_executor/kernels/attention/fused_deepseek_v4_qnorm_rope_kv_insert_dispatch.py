# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform dispatch for the fused DeepseekV4 qnorm/RoPE KV-insert kernels.

CUDA runs the CuTe DSL port (``fused_deepseek_v4_qnorm_rope_kv_insert.py``,
jitted at first use; the extension neither compiles nor ships it). ROCm
keeps the HIP-compiled extension op from
``csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu``, which the build
compiles only for HIP. Both backends expose the same entry points with the
same positional schemas.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform

_CUTEDSL_ERROR = (
    "DeepseekV4 fused KV insert runs as a CuTe DSL kernel and "
    "needs CUDA with nvidia-cutlass-dsl installed"
)


def _backend(name: str):
    """Resolve the backend implementation of kernel ``name``."""
    if current_platform.is_rocm():
        return getattr(torch.ops._C, name)
    from vllm.utils.import_utils import has_cutedsl

    if not has_cutedsl():
        raise RuntimeError(_CUTEDSL_ERROR)
    from vllm.model_executor.kernels.attention import (
        fused_deepseek_v4_qnorm_rope_kv_insert as _cutedsl,
    )

    return getattr(_cutedsl, name)


def fused_deepseek_v4_kv_rope_insert(
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache_block_size: int,
    fp8_scale: torch.Tensor | None = None,
    kv_mxfp8: bool = False,
    kv_norm_weight: torch.Tensor | None = None,
    kv_norm_eps: float = 1e-6,
) -> None:
    """KV half alone: RoPE + quant + insert, no query tensors touched."""
    if current_platform.is_rocm():
        if kv_norm_weight is not None:
            raise NotImplementedError(
                "fused_deepseek_v4_kv_rope_insert: kv_norm needs the CuTe "
                "DSL kernel and is not available on ROCm"
            )
        torch.ops._C.fused_deepseek_v4_kv_rope_insert(
            kv,
            k_cache,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            cache_block_size,
            fp8_scale,
            kv_mxfp8,
        )
        return
    _backend("fused_deepseek_v4_kv_rope_insert")(
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        cache_block_size,
        fp8_scale,
        kv_mxfp8,
        kv_norm_weight,
        kv_norm_eps,
    )


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q_in: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_head_padded: int,
    eps: float,
    cache_block_size: int,
    apply_q_norm: bool = True,
    kv_mxfp8: bool = False,
    apply_q_rope: bool = True,
    is_q_interleaved: bool = False,
) -> torch.Tensor:
    """Q half (RMSNorm + RoPE) fused with the KV insert; returns the roped Q."""
    return _backend("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")(
        q_in,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        q_head_padded,
        eps,
        cache_block_size,
        apply_q_norm,
        kv_mxfp8,
        apply_q_rope,
        is_q_interleaved,
    )


def fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    cache_block_size: int,
    apply_q_norm: bool = True,
) -> None:
    """Rewrite q in place (RMSNorm + RoPE) and insert the bf16 KV row."""
    _backend("fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert")(
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        cache_block_size,
        apply_q_norm,
    )


def fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_fp8: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    fp8_scale: torch.Tensor,
    q_fp8_scale_inv: torch.Tensor,
    eps: float,
    cache_block_size: int,
    apply_q_norm: bool = True,
) -> None:
    """Quantize q and the KV row to per-tensor-scaled e4m3."""
    _backend("fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert")(
        q,
        kv,
        q_fp8,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        fp8_scale,
        q_fp8_scale_inv,
        eps,
        cache_block_size,
        apply_q_norm,
    )
