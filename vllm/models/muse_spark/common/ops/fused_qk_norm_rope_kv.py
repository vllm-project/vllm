# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse-Spark QK-norm + RoPE + KV-cache insert as one launch.

Not a custom op since ``MuseSpark1xModel`` is ``@ignore_torch_compile``;
re-enabling compile requires hiding the forward-context lookup from dynamo.
"""

from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention.attention import get_attention_context

# nvfp4 pages a different layout and keeps the unfused path.
SUPPORTED_KV_CACHE_DTYPES = frozenset(
    {"auto", "bfloat16", "float16", "fp8", "fp8_e4m3", "fp8_e5m2"}
)


def supports_fused_kv_insert(attn_layer: torch.nn.Module) -> bool:
    """Whether this attention layer's backend and cache can take the insert.

    The backend must keep its KV write out of ``forward``. Off under DCP, whose
    FlashInfer prefill attends over the raw new-token k/v.
    """
    parallel_config = get_current_vllm_config().parallel_config
    if parallel_config.decode_context_parallel_size > 1:
        return False
    if attn_layer.attn_backend.forward_includes_kv_cache_update:
        return False
    if not attn_layer.attn_backend.supports_kv_cache_update_bypass:
        return False
    if attn_layer.kv_sharing_target_layer_name is not None:
        return False
    return attn_layer.kv_cache_dtype in SUPPORTED_KV_CACHE_DTYPES


def fused_qk_norm_rope_kv_insert(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    layer_name: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> None:
    """Normalize/rotate QK and insert KV into the attention cache."""
    _, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)

    # No cache bound (profiling) or no slots: skip the insert, still norm+rotate.
    insert = slot_mapping is not None and kv_cache.numel() > 0
    insert_kwargs: dict[str, Any] = {}
    if insert:
        kv_cache_dtype = attn_layer.kv_cache_dtype
        is_fp8 = kv_cache_dtype not in ("auto", "bfloat16", "float16")
        insert_kwargs = {
            "slot_mapping": slot_mapping,
            "kv_cache": kv_cache,
            "block_size": kv_cache.shape[2],
            "kv_cache_dtype": kv_cache_dtype,
            "k_scale": attn_layer._k_scale if is_fp8 else None,
            "v_scale": attn_layer._v_scale if is_fp8 else None,
        }

    ops.fused_muse_spark_qknorm_rope_kv_insert(
        qkv,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        eps,
        cos_sin_cache=cos_sin_cache,
        rotary_dim=rotary_dim,
        **insert_kwargs,
    )
