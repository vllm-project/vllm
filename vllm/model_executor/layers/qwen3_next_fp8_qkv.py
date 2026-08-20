# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM integration wrapper for AITER Qwen3 Next FP8 QKV preparation."""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_qk_norm_rope import (
    fused_qk_rmsnorm_rope_gate,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

MAX_PREQUANTIZED_SEQUENCES = 256

Qwen3NextFp8PrepOutputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _allocate_prequantized_outputs(
    q_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    total_tokens = q_gate.shape[0]
    fp8_dtype = current_platform.fp8_dtype()
    query_fp8 = torch.empty(
        (total_tokens, num_query_heads, head_dim),
        dtype=fp8_dtype,
        device=q_gate.device,
    )
    key_fp8 = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=fp8_dtype,
        device=key.device,
    )
    value_fp8 = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=fp8_dtype,
        device=value.device,
    )
    query_descale = torch.empty(
        (MAX_PREQUANTIZED_SEQUENCES, num_kv_heads),
        dtype=torch.float32,
        device=q_gate.device,
    )
    key_descale = torch.empty_like(query_descale)
    value_descale = torch.empty_like(query_descale)
    return (
        query_fp8,
        key_fp8,
        value_fp8,
        query_descale,
        key_descale,
        value_descale,
    )


def _allocate_outputs(
    q_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Qwen3NextFp8PrepOutputs:
    total_tokens = q_gate.shape[0]
    query = torch.empty(
        (total_tokens, num_query_heads * head_dim),
        dtype=q_gate.dtype,
        device=q_gate.device,
    )
    output_key = torch.empty(
        (total_tokens, num_kv_heads * head_dim),
        dtype=key.dtype,
        device=key.device,
    )
    gate = torch.empty_like(query)
    return (
        query,
        output_key,
        gate,
        *_allocate_prequantized_outputs(
            q_gate,
            key,
            value,
            num_query_heads,
            num_kv_heads,
            head_dim,
        ),
    )


def _qwen3_next_fp8_qkv_prep_impl(
    q_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    layer_name: LayerNameType,
    eps: float,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    query_out: torch.Tensor,
    key_out: torch.Tensor,
    gate_out: torch.Tensor,
    query_fp8_out: torch.Tensor,
    key_fp8_out: torch.Tensor,
    value_fp8_out: torch.Tensor,
    query_descale_out: torch.Tensor,
    key_descale_out: torch.Tensor,
    value_descale_out: torch.Tensor,
) -> None:
    from vllm.model_executor.layers.attention.attention import (
        get_attention_context,
    )

    resolved_name = _resolve_layer_name(layer_name)
    attn_metadata, _, _, _ = get_attention_context(resolved_name)
    total_tokens = q_gate.shape[0]
    if attn_metadata is None:
        num_actual_tokens = total_tokens
        cu_seqlens = (
            torch.arange(
                2,
                dtype=torch.int32,
                device=q_gate.device,
            )
            * total_tokens
        )
        quant_token_start = 0
        quant_sequence_start = 0
    else:
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        cu_seqlens = attn_metadata.query_start_loc
        num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
        quant_token_start = int(getattr(attn_metadata, "num_decode_tokens", 0))
        quant_sequence_start = num_decodes

    if quant_token_start == num_actual_tokens:
        fused_qk_rmsnorm_rope_gate(
            q_gate,
            key,
            query_norm_weight.float() + 1.0,
            key_norm_weight.float() + 1.0,
            cos_sin_cache,
            positions,
            eps,
            num_query_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            query_out=query_out,
            key_out=key_out,
            gate_out=gate_out,
        )
        return

    rocm_aiter_ops.qwen3_next_fp8_qkv_prep(
        q_gate,
        key,
        value,
        query_norm_weight,
        key_norm_weight,
        cos_sin_cache,
        positions,
        cu_seqlens,
        num_actual_tokens=num_actual_tokens,
        quant_token_start=quant_token_start,
        quant_sequence_start=quant_sequence_start,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        query_out=query_out,
        key_out=key_out,
        gate_out=gate_out,
        query_fp8_out=query_fp8_out,
        key_fp8_out=key_fp8_out,
        value_fp8_out=value_fp8_out,
        query_descale_out=query_descale_out,
        key_descale_out=key_descale_out,
        value_descale_out=value_descale_out,
    )


def _qwen3_next_fp8_qkv_prep_fake(
    q_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    layer_name: LayerNameType,
    eps: float,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    query_out: torch.Tensor,
    key_out: torch.Tensor,
    gate_out: torch.Tensor,
    query_fp8_out: torch.Tensor,
    key_fp8_out: torch.Tensor,
    value_fp8_out: torch.Tensor,
    query_descale_out: torch.Tensor,
    key_descale_out: torch.Tensor,
    value_descale_out: torch.Tensor,
) -> None:
    return


# The implementation reads per-step attention metadata from forward context.
# Keep it outside piecewise CUDA graphs so capture-time decode/prefill offsets
# are not replayed against a different runtime batch.
direct_register_custom_op(
    op_name="qwen3_next_fp8_qkv_prep",
    op_func=_qwen3_next_fp8_qkv_prep_impl,
    fake_impl=_qwen3_next_fp8_qkv_prep_fake,
    mutates_args=[
        "query_out",
        "key_out",
        "gate_out",
        "query_fp8_out",
        "key_fp8_out",
        "value_fp8_out",
        "query_descale_out",
        "key_descale_out",
        "value_descale_out",
    ],
    tags=(torch.Tag.cudagraph_unsafe,),
)


def qwen3_next_fp8_qkv_prep(
    q_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    eps: float,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> Qwen3NextFp8PrepOutputs:
    outputs = _allocate_outputs(
        q_gate,
        key,
        value,
        num_query_heads,
        num_kv_heads,
        head_dim,
    )
    torch.ops.vllm.qwen3_next_fp8_qkv_prep(
        q_gate,
        key,
        value,
        query_norm_weight,
        key_norm_weight,
        cos_sin_cache,
        positions,
        _encode_layer_name(layer_name),
        eps,
        num_query_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        *outputs,
    )
    return outputs
