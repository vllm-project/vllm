# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)

logger = init_logger(__name__)

_COS_SIN_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def fused_qk_norm_mrope_and_unified_kv_cache_update_impl(
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    rms_norm_eps: float,
    is_neox: bool,
    is_interleaved: bool,
    mrope_section: list[int],
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)
    if slot_mapping is None or kv_cache is None or kv_cache.numel() == 0:
        q_out.zero_()
        return

    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(3, -1).contiguous()

    logger.info_once(
        "Qwen3-VL fused attention prologue is active: positions=%s, kv_cache=%s",
        tuple(positions.shape),
        tuple(kv_cache.shape),
    )

    impl = attn_layer.impl
    key_cache, value_cache = impl._split_kv_cache(kv_cache)
    cache_key = (cos_sin_cache.data_ptr(), qkv.device, qkv.dtype)
    converted_cos_sin = _COS_SIN_CACHE.get(cache_key)
    if converted_cos_sin is None:
        converted_cos_sin = cos_sin_cache.to(qkv.dtype).contiguous()
        _COS_SIN_CACHE[cache_key] = converted_cos_sin

    rocm_aiter_ops.do_qk_norm_mrope_kvcache_update(
        qkv=qkv.contiguous(),
        q_weight=q_weight,
        k_weight=k_weight,
        cos_sin_cache=converted_cos_sin,
        positions=positions,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_size,
        is_neox=is_neox,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        rms_norm_eps=rms_norm_eps,
        q_out=q_out,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        k_scale=attn_layer._k_scale,
        v_scale=attn_layer._v_scale,
        kv_cache_dtype=impl.kv_cache_dtype,
    )


def fused_qk_norm_mrope_and_unified_kv_cache_update_fake(
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    rms_norm_eps: float,
    is_neox: bool,
    is_interleaved: bool,
    mrope_section: list[int],
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="fused_qk_norm_mrope_and_unified_kv_cache_update",
    op_func=fused_qk_norm_mrope_and_unified_kv_cache_update_impl,
    mutates_args=["q_out"],
    fake_impl=fused_qk_norm_mrope_and_unified_kv_cache_update_fake,
)

fused_qk_norm_mrope_and_unified_kv_cache_update = (
    torch.ops.vllm.fused_qk_norm_mrope_and_unified_kv_cache_update.default
)
