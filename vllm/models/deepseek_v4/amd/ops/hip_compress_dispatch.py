# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx942 DeepSeek-V4 HIP compressor dispatch."""

from typing import Any

import torch

from vllm import envs

SUPPORTED_SHAPES = frozenset({(512, 4), (512, 128)})


def hip_compressor_selected(head_dim: int, compress_ratio: int) -> bool:
    return (
        envs.VLLM_ROCM_DSV4_HIP_COMPRESSOR
        and (
            head_dim,
            compress_ratio,
        )
        in SUPPORTED_SHAPES
    )


def hip_compressor_runtime_available() -> bool:
    try:
        from vllm.platforms.rocm import on_gfx942

        return on_gfx942()
    except Exception:
        return False


def _aot_op(head_dim: int, compress_ratio: int):
    rocm_C = getattr(torch.ops, "_rocm_C", None)
    if rocm_C is None:
        return None
    if head_dim == 512 and compress_ratio == 4:
        return getattr(rocm_C, "dsv4_csa_compress", None)
    if head_dim == 512 and compress_ratio == 128:
        return getattr(rocm_C, "dsv4_hca_compress", None)
    return None


def hip_compressor_available(head_dim: int, compress_ratio: int) -> bool:
    return (
        hip_compressor_runtime_available()
        and _aot_op(head_dim, compress_ratio) is not None
    )


def hip_compressor_enabled(
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    cache_dtype: str,
) -> bool:
    return (
        hip_compressor_selected(head_dim, compress_ratio)
        and hip_compressor_available(head_dim, compress_ratio)
        and rope_head_dim == 64
        and cache_dtype == "fp8_ds_mla"
    )


def hip_compressor_supported(
    head_dim: int,
    compress_ratio: int,
    kv_cache: torch.Tensor,
    allowed_shapes: frozenset[tuple[int, int]] | None = None,
) -> bool:
    if allowed_shapes is None:
        allowed_shapes = SUPPORTED_SHAPES
    if (head_dim, compress_ratio) not in allowed_shapes:
        return False
    if kv_cache.dtype != torch.uint8:
        return False
    return hip_compressor_available(head_dim, compress_ratio)


def compress_norm_rope_store_hip(
    *,
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
    ape: torch.Tensor,
    use_bf16_state_cache: bool = True,
    hca_plan_scratch: torch.Tensor | None = None,
    hca_counter_scratch: torch.Tensor | None = None,
    **_ignored: Any,
) -> None:
    if num_actual == 0:
        return

    op = _aot_op(head_dim, compress_ratio)
    if op is None:
        raise RuntimeError(
            f"HIP compressor op unavailable for (head_dim={head_dim}, "
            f"compress_ratio={compress_ratio}); was _rocm_C built with "
            f"VLLM_ROCM_GFX942?"
        )

    args = [
        state_cache,
        num_actual,
        ape,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],
        scale_dim,
    ]
    if head_dim == 512 and compress_ratio == 128:
        if hca_plan_scratch is None or hca_counter_scratch is None:
            raise RuntimeError(
                "HCA HIP compressor requires reusable plan/counter scratch buffers."
            )
        args.extend([hca_plan_scratch, hca_counter_scratch])
    op(*args)
