from typing import Optional, Tuple

import torch

try:
    from vllm._C import cache_ops as vllm_cache_ops
    from vllm._C import ops as vllm_ops
except ImportError:
    pass


# activation ops
def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.silu_and_mul(out, x)


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.gelu_and_mul(out, x)


def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.gelu_tanh_and_mul(out, x)


def gelu_fast(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.gelu_fast(out, x)


def gelu_new(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.gelu_new(out, x)


# page attention ops
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_ops.paged_attention_v1(out, query, key_cache, value_cache,
                                num_kv_heads, scale, block_tables, seq_lens,
                                block_size, max_seq_len, alibi_slopes,
                                kv_cache_dtype, kv_scale)


def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_ops.paged_attention_v2(out, exp_sum, max_logits, tmp_out, query,
                                key_cache, value_cache, num_kv_heads, scale,
                                block_tables, seq_lens, block_size,
                                max_seq_len, alibi_slopes, kv_cache_dtype,
                                kv_scale)


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    vllm_ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache,
                              is_neox)


def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                             key: torch.Tensor, head_size: int,
                             cos_sin_cache: torch.Tensor, is_neox: bool,
                             rot_dim: int,
                             cos_sin_cache_offsets: torch.Tensor) -> None:
    vllm_ops.batched_rotary_embedding(positions, query, key, head_size,
                                      cos_sin_cache, is_neox, rot_dim,
                                      cos_sin_cache_offsets)


# layer norm ops
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    vllm_ops.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float) -> None:
    vllm_ops.fused_add_rms_norm(input, residual, weight, epsilon)


# quantization ops
# awq
def awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                   zeros: torch.Tensor, split_k_iters: int, thx: int,
                   thy: int) -> torch.Tensor:
    return vllm_ops.awq_dequantize(qweight, scales, zeros, split_k_iters, thx,
                                   thy)


def awq_gemm(input: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
             scales: torch.Tensor, split_k_iters: int) -> torch.Tensor:
    return vllm_ops.awq_gemm(input, qweight, qzeros, scales, split_k_iters)


# gptq
def gptq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    return vllm_ops.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                              b_g_idx, use_exllama, bit)


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor,
                 bit: int) -> None:
    vllm_ops.gptq_shuffle(q_weight, q_perm, bit)


# squeezellm
def squeezellm_gemm(vec: torch.Tensor, mat: torch.Tensor, mul: torch.Tensor,
                    lookup_table: torch.Tensor) -> None:
    vllm_ops.squeezellm_gemm(vec, mat, mul, lookup_table)


# marlin
def marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                b_scales: torch.Tensor, workspace: torch.Tensor, size_m: int,
                size_n: int, size_k: int) -> torch.Tensor:
    return vllm_ops.marlin_gemm(a, b_q_weight, b_scales, workspace, size_m,
                                size_n, size_k)


# aqlm
def aqlm_gemm(input: torch.Tensor, codes: torch.Tensor,
              codebooks: torch.Tensor, scales: torch.Tensor,
              codebook_partition_sizes: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
    return vllm_ops.aqlm_gemm(input, codes, codebooks, scales,
                              codebook_partition_sizes, bias)


def aqlm_dequant(codes: torch.Tensor, codebooks: torch.Tensor,
                 codebook_partition_sizes: torch.Tensor) -> torch.Tensor:
    return vllm_ops.aqlm_dequant(codes, codebooks, codebook_partition_sizes)


# gptq_marlin
def gptq_marlin_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                       size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    return vllm_ops.gptq_marlin_repack(b_q_weight, perm, size_k, size_n,
                                       num_bits)


def gptq_marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                     b_scales: torch.Tensor, g_idx: torch.Tensor,
                     perm: torch.Tensor, workspace: torch.Tensor,
                     num_bits: int, size_m: int, size_n: int, size_k: int,
                     is_k_full: bool) -> torch.Tensor:
    return vllm_ops.gptq_marlin_gemm(a, b_q_weight, b_scales, g_idx, perm,
                                     workspace, num_bits, size_m, size_n,
                                     size_k, is_k_full)


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    batch_dim_padding: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensor for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        batch_dim_padding: If specified, pad the first dimension
            of the output to at least this value.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    if batch_dim_padding:
        shape = (max(batch_dim_padding, input.shape[0]), *input.shape[1:])
        output = torch.empty(shape,
                             device=input.device,
                             dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        vllm_ops.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        vllm_ops.static_scaled_fp8_quant(output, input, scale)
    return output, scale


# moe
def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                         block_size: int, sorted_token_ids: torch.Tensor,
                         experts_ids: torch.Tensor,
                         num_tokens_post_pad: torch.Tensor) -> None:
    vllm_ops.moe_align_block_size(topk_ids, num_experts, block_size,
                                  sorted_token_ids, experts_ids,
                                  num_tokens_post_pad)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_cache_ops.reshape_and_cache(key, value, key_cache, value_cache,
                                     slot_mapping, kv_cache_dtype, kv_scale)


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
) -> None:
    vllm_cache_ops.reshape_and_cache_flash(key, value, key_cache, value_cache,
                                           slot_mapping, kv_cache_dtype)


def copy_blocks(key_caches: torch.Tensor, value_caches: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    vllm_cache_ops.copy_blocks(key_caches, value_caches, block_mapping)


def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    vllm_cache_ops.swap_blocks(src, dst, block_mapping)


def convert_fp8(output: torch.Tensor,
                input: torch.Tensor,
                scale: float = 1.0,
                kv_dtype: str = "fp8") -> None:
    vllm_cache_ops.convert_fp8(output, input, scale, kv_dtype)


#TODO: cuda_utils, custom_ar
