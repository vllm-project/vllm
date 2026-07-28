# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime adapter for checked-in fused QK norm and RoPE kernels."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from vllm.kernels.helion_generated.dispatcher import (
    _load_launcher,
    _runtime_platform,
    _select_bucketed_module,
    _selected_cases,
    vllm_helion_generated_lib,
)
from vllm.utils.torch_utils import direct_register_custom_op

OP_NAME = "fused_qk_norm_rope"
NATIVE_OP_NAME = "fused_qk_norm_rope"


def _eligible_module(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int,
) -> str | None:
    num_tokens = qkv.shape[0] if qkv.ndim == 2 else 0
    rotary_dim = cos_sin_cache.shape[1] if cos_sin_cache.ndim == 2 else 0
    embed_dim = rotary_dim // 2
    if (
        qkv.ndim != 2
        or qkv.dtype != torch.bfloat16
        or not qkv.is_cuda
        or not qkv.is_contiguous()
        or num_heads_q < 1
        or num_heads_k < 1
        or num_heads_v != num_heads_k
        or head_dim < 1
        or head_dim & (head_dim - 1)
        or qkv.shape[1] != (num_heads_q + num_heads_k + num_heads_v) * head_dim
        or q_weight.shape != (head_dim,)
        or q_weight.dtype != qkv.dtype
        or q_weight.device != qkv.device
        or not q_weight.is_contiguous()
        or k_weight.shape != (head_dim,)
        or k_weight.dtype != qkv.dtype
        or k_weight.device != qkv.device
        or not k_weight.is_contiguous()
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[0] < 1
        or rotary_dim < 2
        or rotary_dim % 2 != 0
        or rotary_dim > head_dim
        or embed_dim & (embed_dim - 1)
        or cos_sin_cache.dtype != qkv.dtype
        or cos_sin_cache.device != qkv.device
        or not cos_sin_cache.is_contiguous()
        or position_ids.shape != (num_tokens,)
        or position_ids.dtype != torch.int64
        or position_ids.device != qkv.device
        or not position_ids.is_contiguous()
        or forced_token_heads_per_warp != -1
    ):
        return None
    return _select_bucketed_module(
        OP_NAME,
        _runtime_platform(),
        (num_heads_q, num_heads_k),
        num_tokens,
    )


def fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int = -1,
) -> None:
    module_path = _eligible_module(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids,
        forced_token_heads_per_warp,
    )
    if module_path is None:
        torch.ops._C.fused_qk_norm_rope(
            qkv,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            eps,
            q_weight,
            k_weight,
            cos_sin_cache,
            is_neox,
            position_ids,
            forced_token_heads_per_warp,
        )
        return
    _load_launcher(module_path)(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids,
        forced_token_heads_per_warp,
    )


def warmup(
    token_counts: Iterable[int],
    device: torch.device | str = "cuda",
) -> None:
    cos_sin_cache: torch.Tensor | None = None
    for q_heads, kv_heads, num_tokens in _selected_cases(OP_NAME, token_counts):
        head_dim = 128
        qkv = torch.empty(
            (num_tokens, (q_heads + 2 * kv_heads) * head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        q_weight = torch.empty(head_dim, device=device, dtype=qkv.dtype)
        k_weight = torch.empty_like(q_weight)
        if cos_sin_cache is None:
            cos_sin_cache = torch.empty(
                (40960, head_dim), device=device, dtype=qkv.dtype
            )
        position_ids = torch.arange(num_tokens, device=device, dtype=torch.int64)
        fused_qk_norm_rope(
            qkv,
            q_heads,
            kv_heads,
            kv_heads,
            head_dim,
            1e-6,
            q_weight,
            k_weight,
            cos_sin_cache,
            True,
            position_ids,
        )


direct_register_custom_op(
    op_name="fused_qk_norm_rope",
    op_func=fused_qk_norm_rope,
    mutates_args=["qkv"],
    fake_impl=lambda *args, **kwargs: None,
    target_lib=vllm_helion_generated_lib,
)
