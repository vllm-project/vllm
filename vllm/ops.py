from typing import Dict, Optional

import torch

try:
    from vllm._C import cache_ops as vllm_cache_ops
    from vllm._C import ops as vllm_ops
except ImportError:
    pass


class ops:

    # activation ops
    def silu_and_mul(out: torch.Tensor, x: torch.Tensor):
        vllm_ops.silu_and_mul(out, x)

    def gelu_and_mul(out: torch.Tensor, x: torch.Tensor):
        vllm_ops.gelu_and_mul(out, x)

    def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor):
        vllm_ops.gelu_tanh_and_mul(out, x)

    def gelu_fast(out: torch.Tensor, x: torch.Tensor):
        vllm_ops.gelu_fast(out, x)

    def gelu_new(out: torch.Tensor, x: torch.Tensor):
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
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: Optional[torch.Tensor],
        kv_cache_dtype: str,
        kv_scale: float,
    ):
        vllm_ops.paged_attention_v1(out, query, key_cache, value_cache,
                                    num_kv_heads, scale, block_tables,
                                    context_lens, block_size, max_context_len,
                                    alibi_slopes, kv_cache_dtype, kv_scale)

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
        context_lens: torch.Tensor,
        block_size: int,
        max_context_len: int,
        alibi_slopes: Optional[torch.Tensor],
        kv_cache_dtype: str,
        kv_scale: float,
    ):
        vllm_ops.paged_attention_v2(out, exp_sum, max_logits, tmp_out, query,
                                    key_cache, value_cache, num_kv_heads,
                                    scale, block_tables, context_lens,
                                    block_size, max_context_len, alibi_slopes,
                                    kv_cache_dtype, kv_scale)

    # pos encoding ops
    def rotary_embedding(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
    ):
        vllm_ops.rotary_embedding(positions, query, key, head_size,
                                  cos_sin_cache, is_neox)

    def batched_rotary_embedding(positions: torch.tensor, query: torch.tensor,
                                 key: torch.tensor, head_size: int,
                                 cos_sin_cache: torch.tensor, is_neox: bool,
                                 rot_dim: int,
                                 cos_sin_cache_offsets: torch.tensor):
        vllm_ops.batched_rotary_embedding(positions, query, key, head_size,
                                          cos_sin_cache, is_neox, rot_dim,
                                          cos_sin_cache_offsets)

    # layer norm ops
    def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                 epsilon: float):
        vllm_ops.rms_norm(out, input, weight, epsilon)

    def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                           weight: torch.Tensor, epsilon: float):
        vllm_ops.fused_add_rms_norm(input, residual, weight, epsilon)

    # quantization ops
    # awq
    def awq_dequantize(qweight: torch.tensor, scales: torch.tensor,
                       zeros: torch.tensor, split_k_iters, thx, thy):
        vllm_ops.awq_dequantize(qweight, scales, zeros, split_k_iters, thx,
                                thy)

    def awq_gemm(input: torch.tensor, qweight: torch.tensor,
                 qzeros: torch.tensor, scales: torch.tensor,
                 split_k_iters: int):
        vllm_ops.awq_gemm(input, qweight, qzeros, scales, split_k_iters)

    # gptq
    def gptq_gemm(a: torch.tensor, b_q_weight: torch.tensor,
                  b_gptq_qzeros: torch.tensor, b_gptq_scales: torch.tensor,
                  b_g_idx: torch.tensor, use_exllama: bool, bit: int):
        vllm_ops.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                           b_g_idx, use_exllama, bit)

    def gptq_shuffle(q_weight: torch.tensor, q_perm: torch.Tensor, bit: int):
        vllm_ops.gptq_shuffle(q_weight, q_perm, bit)

    # squeezellm
    def squeezellm_gemm(vec: torch.tensor, mat: torch.tensor,
                        mul: torch.tensor, lookup_table: torch.tensor):
        vllm_ops.squeezellm_gemm(vec, mat, mul, lookup_table)

    # marlin
    def marlin_gemm(a: torch.tensor, b_q_weight: torch.tensor,
                    b_scales: torch.tensor, workspace: torch.tensor,
                    size_m: int, size_n: int, size_k: int):
        return vllm_ops.marlin_gemm(a, b_q_weight, b_scales, workspace, size_m,
                                    size_n, size_k)

    # moe
    def moe_align_block_size(topk_ids: torch.tensor, num_experts: int,
                             block_size: int, sorted_token_ids: torch.tensor,
                             experts_ids: torch.tensor,
                             num_tokens_post_pad: torch.tensor):
        vllm_ops.moe_align_block_size(topk_ids, num_experts, block_size,
                                      sorted_token_ids, experts_ids,
                                      num_tokens_post_pad)


class cache_ops:

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
                                         slot_mapping, kv_cache_dtype,
                                         kv_scale)

    def copy_blocks(key_caches, value_caches, block_mapping):
        vllm_cache_ops.copy_blocks(key_caches, value_caches, block_mapping)

    def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                    block_mapping: Dict[int, int]):
        vllm_cache_ops.swap_blocks(src, dst, block_mapping)

    def convert_fp8(output: torch.tensor, input: torch.tensor):
        vllm_cache_ops.convert_fp8(output, input)


#TODO: cuda_utils, custom_ar
