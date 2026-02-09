# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from typing import TYPE_CHECKING, Optional, Union

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType

logger = init_logger(__name__)

torch_npu = None
with contextlib.suppress(ImportError):
    import torch_npu as _torch_npu
    torch_npu = _torch_npu

if not current_platform.is_tpu() and not current_platform.is_xpu():
    try:
        import vllm._C
    except ImportError as e:
        logger.warning("Failed to import from vllm._C with %r", e)

supports_moe_ops = False
with contextlib.suppress(ImportError):
    import vllm._moe_C  # noqa: F401
    supports_moe_ops = True

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


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
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    torch.ops._C.paged_attention_v1(
        out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
        seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype,
        k_scale, v_scale, tp_rank, blocksparse_local_blocks,
        blocksparse_vert_stride, blocksparse_block_size,
        blocksparse_head_sliding_step)


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
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    torch.ops._C.paged_attention_v2(
        out, exp_sum, max_logits, tmp_out, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len,
        alibi_slopes, kv_cache_dtype, k_scale, v_scale, tp_rank,
        blocksparse_local_blocks, blocksparse_vert_stride,
        blocksparse_block_size, blocksparse_head_sliding_step)


def paged_attention_rocm(
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
    query_start_loc: Optional[torch.Tensor],
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: Optional[torch.Tensor] = None,
    mfma_type: str = "fp8" if envs.VLLM_ROCM_FP8_MFMA_PAGE_ATTN else "f16",
) -> None:
    torch.ops._rocm_C.paged_attention(out, exp_sum, max_logits, tmp_out, query,
                                      key_cache, value_cache, num_kv_heads,
                                      scale, block_tables, seq_lens,
                                      query_start_loc, block_size, max_seq_len,
                                      alibi_slopes, kv_cache_dtype, k_scale,
                                      v_scale, fp8_out_scale, mfma_type)


def mla_decode_kvcache_cpu(
    out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    torch.ops._C_cpu.mla_decode_kvcache(out, query, kv_cache, scale,
                                        block_tables, seq_lens)


# merge attn states ops
def merge_attn_states(output: torch.Tensor,
                      prefix_output: torch.Tensor,
                      prefix_lse: torch.Tensor,
                      suffix_output: torch.Tensor,
                      suffix_lse: torch.Tensor,
                      output_lse: Optional[torch.Tensor] = None) -> None:
    torch.ops._C.merge_attn_states(output, output_lse, prefix_output,
                                   prefix_lse, suffix_output, suffix_lse)


def convert_vertical_slash_indexes(
    q_seqlens: torch.Tensor,  # [BATCH, ]
    kv_seqlens: torch.Tensor,  # [BATCH, ]
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    block_count = torch.zeros(batch_size,
                              num_heads,
                              num_rows,
                              dtype=q_seqlens.dtype,
                              device=q_seqlens.device)
    block_offset = torch.zeros(batch_size,
                               num_heads,
                               num_rows,
                               nnz_slash,
                               dtype=q_seqlens.dtype,
                               device=q_seqlens.device)
    column_count = torch.zeros(batch_size,
                               num_heads,
                               num_rows,
                               dtype=q_seqlens.dtype,
                               device=q_seqlens.device)
    column_index = torch.zeros(batch_size,
                               num_heads,
                               num_rows,
                               nnz_vertical,
                               dtype=q_seqlens.dtype,
                               device=q_seqlens.device)

    torch.ops._C.convert_vertical_slash_indexes(
        block_count, block_offset, column_count, column_index, q_seqlens,
        kv_seqlens, vertical_indexes, slash_indexes, context_size,
        block_size_M, block_size_N, causal)
    return block_count, block_offset, column_count, column_index


def convert_vertical_slash_indexes_mergehead(
    q_seqlens: torch.Tensor,  # [BATCH, ]
    kv_seqlens: torch.Tensor,  # [BATCH, ]
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    # [N_HEADS] : different head use different number of indices
    vertical_indices_count: torch.Tensor,
    slash_indices_count: torch.Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    block_count = torch.empty(batch_size,
                              num_heads,
                              num_rows,
                              dtype=q_seqlens.dtype,
                              device=q_seqlens.device)
    block_offset = torch.empty(batch_size,
                               num_heads,
                               num_rows,
                               nnz_slash,
                               dtype=q_seqlens.dtype,
                               device=q_seqlens.device)
    column_count = torch.empty(batch_size,
                               num_heads,
                               num_rows,
                               dtype=q_seqlens.dtype,
                               device=q_seqlens.device)
    column_index = torch.empty(batch_size,
                               num_heads,
                               num_rows,
                               nnz_vertical,
                               dtype=q_seqlens.dtype,
                               device=q_seqlens.device)

    torch.ops._C.convert_vertical_slash_indexes_mergehead(
        block_count, block_offset, column_count, column_index, q_seqlens,
        kv_seqlens, vertical_indexes, slash_indexes, vertical_indices_count,
        slash_indices_count, context_size, block_size_M, block_size_N, causal)
    return block_count, block_offset, column_count, column_index


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    if (torch_npu is not None and key is not None
            and query.device.type == "npu" and key.device.type == "npu"):
        if not query.is_contiguous():
            query = query.contiguous()
        if not key.is_contiguous():
            key = key.contiguous()
        torch_npu._npu_rotary_embedding(positions, query, key, head_size,
                                        cos_sin_cache, is_neox)
        return
    torch.ops._C.rotary_embedding(positions, query, key, head_size,
                                  cos_sin_cache, is_neox)


# layer norm ops
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    # TODO: Remove this contiguous call when the kernel is updated to support non-contiguous input
    input_contiguous = input.contiguous()
    if torch_npu is not None and input_contiguous.device.type == "npu":
        npu_out, _ = torch_npu.npu_rms_norm(input_contiguous, weight, epsilon)
        out.copy_(npu_out)
        return
    torch.ops._C.rms_norm(out, input_contiguous, weight, epsilon)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float) -> None:
    if torch_npu is not None and input.device.type == "npu":
        torch_npu.npu_add_rms_norm(input, residual, weight, epsilon)
        return
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)


def poly_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
              bias: torch.Tensor, epsilon: float) -> None:
    # TODO: Remove this contiguous call when the kernel is updated to support non-contiguous input
    input_contiguous = input.contiguous()
    torch.ops._C.poly_norm(out, input_contiguous, weight, bias, epsilon)


def apply_repetition_penalties_torch(
        logits: torch.Tensor, prompt_mask: torch.Tensor,
        output_mask: torch.Tensor, repetition_penalties: torch.Tensor) -> None:
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, logits.size(1))
    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
    penalties = torch.where(prompt_mask | output_mask, repetition_penalties,
                            1.0)
    # If logits are positive, divide by penalty, otherwise multiply by penalty.
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits *= scaling


def apply_repetition_penalties_cuda(
        logits: torch.Tensor, prompt_mask: torch.Tensor,
        output_mask: torch.Tensor, repetition_penalties: torch.Tensor) -> None:
    torch.ops._C.apply_repetition_penalties_(logits, prompt_mask, output_mask,
                                             repetition_penalties)


def apply_repetition_penalties(logits: torch.Tensor, prompt_mask: torch.Tensor,
                               output_mask: torch.Tensor,
                               repetition_penalties: torch.Tensor) -> None:
    """Apply repetition penalties to logits in-place.

    Args:
        logits: The logits tensor of shape [num_seqs, vocab_size].
        prompt_mask: A boolean tensor indicating which tokens appear in the prompt.
        output_mask: A boolean tensor indicating which tokens appear in the output.
        repetition_penalties: The repetition penalties of shape (num_seqs, ).
    """
    if logits.device.type == "cuda" and logits.is_contiguous():
        apply_repetition_penalties_cuda(logits, prompt_mask, output_mask,
                                        repetition_penalties)
    else:
        apply_repetition_penalties_torch(logits, prompt_mask, output_mask,
                                         repetition_penalties)


# fused quant layer norm ops
def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    scale_ub: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, dtype=quant_dtype)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                         device=input.device,
                         dtype=torch.float32)

    torch.ops._C.rms_norm_dynamic_per_token_quant(output, input, weight,
                                                  scales, epsilon, scale_ub,
                                                  residual)
    return output, scales


# quantization ops
# awq
def awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                   zeros: torch.Tensor, split_k_iters: int, thx: int,
                   thy: int) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import (
            awq_dequantize_triton)
        return awq_dequantize_triton(qweight, scales, zeros)
    return torch.ops._C.awq_dequantize(qweight, scales, zeros, split_k_iters,
                                       thx, thy)


def awq_gemm(input: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
             scales: torch.Tensor, split_k_iters: int) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import (
            awq_gemm_triton)
        return awq_gemm_triton(input, qweight, qzeros, scales, split_k_iters)
    return torch.ops._C.awq_gemm(input, qweight, qzeros, scales, split_k_iters)


# gptq
def gptq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    return torch.ops._C.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                                  b_g_idx, use_exllama, bit)


if hasattr(torch.ops._C, "gptq_gemm"):

    @register_fake("_C::gptq_gemm")
    def _gptq_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                        b_gptq_qzeros: torch.Tensor,
                        b_gptq_scales: torch.Tensor, b_g_idx: torch.Tensor,
                        use_exllama: bool, bit: int) -> torch.Tensor:
        return torch.empty((a.size(0), b_q_weight.size(1)),
                           dtype=a.dtype,
                           device=a.device)


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor,
                 bit: int) -> None:
    torch.ops._C.gptq_shuffle(q_weight, q_perm, bit)


# marlin_24
def gptq_marlin_24_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                        b_meta: torch.Tensor, b_scales: torch.Tensor,
                        workspace: torch.Tensor, b_q_type: ScalarType,
                        size_m: int, size_n: int, size_k: int) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_24_gemm(a, b_q_weight, b_meta, b_scales,
                                            workspace, b_q_type.id, size_m,
                                            size_n, size_k)


if hasattr(torch.ops._C, "gptq_marlin_24_gemm"):

    @register_fake("_C::gptq_marlin_24_gemm")
    def _gptq_marlin_24_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                                  b_meta: torch.Tensor, b_scales: torch.Tensor,
                                  workspace: torch.Tensor,
                                  b_q_type: ScalarType, size_m: torch.SymInt,
                                  size_n: torch.SymInt,
                                  size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @register_fake("_C::gptq_marlin_gemm")
    def _gptq_marlin_gemm_fake(a: torch.Tensor,
                               c: Optional[torch.Tensor],
                               b_q_weight: torch.Tensor,
                               b_bias: Optional[torch.Tensor],
                               b_scales: torch.Tensor,
                               global_scale: Optional[torch.Tensor],
                               b_zeros: Optional[torch.Tensor],
                               g_idx: Optional[torch.Tensor],
                               perm: Optional[torch.Tensor],
                               workspace: torch.Tensor,
                               b_q_type_id: int,
                               size_m: torch.SymInt,
                               size_n: torch.SymInt,
                               size_k: torch.SymInt,
                               is_k_full: bool = True,
                               use_atomic_add: bool = False,
                               use_fp32_reduce: bool = False,
                               is_zp_float: bool = False) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @register_fake("_C::awq_dequantize")
    def _awq_dequantize_fake(qweight: torch.Tensor, scales: torch.Tensor,
                             zeros: torch.Tensor, split_k_iters: torch.SymInt,
                             thx: int, thy: int) -> torch.Tensor:
        in_c = qweight.size(0)
        qout_c = qweight.size(1)
        out_c = qout_c * 8
        return torch.empty((in_c, out_c),
                           dtype=scales.dtype,
                           device=scales.device)

    @register_fake("_C::awq_gemm")
    def _awq_gemm_fake(input: torch.Tensor, qweight: torch.Tensor,
                       qzeros: torch.Tensor, scales: torch.Tensor,
                       split_k_iters: torch.SymInt) -> torch.Tensor:
        num_in_feats = input.size(0)
        return torch.empty((split_k_iters, num_in_feats, qweight.size(1) * 8),
                           dtype=input.dtype,
                           device=input.device).sum(0)

    @register_fake("_C::machete_mm")
    def machete_mm_fake(
        a: torch.Tensor,
        # b_q Should be the tensor returned by machete_prepack_B
        b_q: torch.Tensor,
        b_type: ScalarType,
        out_type: Optional[torch.dtype] = None,
        b_group_scales: Optional[torch.Tensor] = None,
        b_group_zeros: Optional[torch.Tensor] = None,
        b_group_size: Optional[int] = None,
        b_channel_scales: Optional[torch.Tensor] = None,
        a_token_scales: Optional[torch.Tensor] = None,
        schedule: Optional[str] = None,
    ) -> torch.Tensor:
        m = a.size(0)
        n = b_q.size(1)
        return torch.empty((m, n), device=a.device, dtype=a.dtype)

    @register_fake("_C::machete_prepack_B")
    def machete_prepack_B_fake(
            b_q_weight: torch.Tensor, a_type: torch.dtype, b_type: ScalarType,
            group_scales_type: Optional[torch.dtype]) -> torch.Tensor:
        return torch.empty_like(b_q_weight,
                                memory_format=torch.contiguous_format)

    @register_fake("_C::cutlass_w4a8_mm")
    def cutlass_w4a8_mm_fake(
            a: torch.Tensor,
            # b_q Should be the tensor returned by cutlass_encode_and_reorder_int4b
            b_q: torch.Tensor,
            b_group_scales: torch.Tensor,
            b_group_size: int,
            b_channel_scales: torch.Tensor,
            a_token_scales: torch.Tensor,
            out_type: Optional[torch.dtype] = None,
            maybe_schedule: Optional[str] = None) -> torch.Tensor:
        m = a.size(0)
        n = b_q.size(1)
        out_dtype = out_type if out_type is not None else torch.bfloat16
        return torch.empty((m, n), device=a.device, dtype=out_dtype)

    @register_fake("_C::cutlass_pack_scale_fp8")
    def cutlass_pack_scale_fp8_fake(scales: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(scales, memory_format=torch.contiguous_format)

    @register_fake("_C::cutlass_encode_and_reorder_int4b")
    def cutlass_encode_and_reorder_int4b_fake(b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(b, memory_format=torch.contiguous_format)


if hasattr(torch.ops._C, "allspark_w8a16_gemm"):

    @register_fake("_C::allspark_w8a16_gemm")
    def _allspark_w8a16_gemm_fake(a: torch.Tensor, b_qweight: torch.Tensor,
                                  b_scales: torch.Tensor,
                                  b_qzeros: Optional[torch.Tensor],
                                  n: torch.SymInt, group_size: torch.SymInt,
                                  sm_count: torch.SymInt,
                                  sm_version: torch.SymInt,
                                  CUBLAS_M_THRESHOLD: torch.SymInt,
                                  has_zp: bool,
                                  n32k16_reorder: bool) -> torch.Tensor:
        m = a.size(0)
        return torch.empty((m, n), device=a.device, dtype=a.dtype)


if hasattr(torch.ops._C, "ggml_dequantize"):

    @register_fake("_C::ggml_dequantize")
    def _ggml_dequantize_fake(
            W: torch.Tensor,
            quant_type: int,
            m: torch.SymInt,
            n: torch.SymInt,
            dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return torch.empty((m, n), dtype=torch.float16, device=W.device)

    @register_fake("_C::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.shape[0], row), dtype=X.dtype, device=W.device)

    @register_fake("_C::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        batch = X.size(0)
        return torch.empty((batch, row), dtype=X.dtype, device=W.device)

    @register_fake("_C::ggml_moe_a8")
    def _ggml_moe_a8_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
        top_k: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        tokens = X.size(0)
        return torch.empty((tokens * top_k, row),
                           dtype=torch.float16,
                           device=W.device)


if hasattr(torch.ops._C, "ggml_moe_a8_vec"):

    @register_fake("_C::ggml_moe_a8_vec")
    def _ggml_moe_a8_vec_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        quant_type: int,
        row: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        tokens = X.size(0)
        return torch.empty((tokens * top_k, row),
                           dtype=X.dtype,
                           device=W.device)


# cutlass
def cutlass_scaled_mm_supports_fp4(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_fp4(cuda_device_capability)


def cutlass_blockwise_scaled_grouped_mm(
    output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
):
    torch.ops._C.cutlass_blockwise_scaled_grouped_mm(output, a, b, scales_a,
                                                     scales_b, problem_sizes,
                                                     expert_offsets)


def cutlass_scaled_fp4_mm(a: torch.Tensor, b: torch.Tensor,
                          block_scale_a: torch.Tensor,
                          block_scale_b: torch.Tensor, alpha: torch.Tensor,
                          out_dtype: torch.dtype) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    torch.ops._C.cutlass_scaled_fp4_mm(out, a, b, block_scale_a, block_scale_b,
                                       alpha)
    return out


def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_fp8(cuda_device_capability)


def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_scaled_mm_supports_block_fp8(
        cuda_device_capability)


def cutlass_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.numel(
    ) == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])

    cutlass_compatible_b = (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    if current_platform.is_rocm() or not cutlass_compatible_b:
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa
            triton_scaled_mm)
        out = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    else:
        out = torch.empty((a.shape[0], b.shape[1]),
                          dtype=out_dtype,
                          device=a.device)
        torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    return out.view(*target_shape)


def cutlass_scaled_mm_azp(a: torch.Tensor,
                          b: torch.Tensor,
                          scale_a: torch.Tensor,
                          scale_b: torch.Tensor,
                          out_dtype: torch.dtype,
                          azp_adj: torch.Tensor,
                          azp: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    :param azp_adj: In the per-tensor case, this should include the azp.
    Always per-channel.
    :param azp: Only set in the per-token case. Per-token if set.
    """
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.numel(
    ) == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])
    assert azp is None or azp.numel() == a.shape[0]

    out = torch.empty((a.shape[0], b.shape[1]),
                      dtype=out_dtype,
                      device=a.device)
    torch.ops._C.cutlass_scaled_mm_azp(out, a, b, scale_a, scale_b, azp_adj,
                                       azp, bias)
    return out.view(*target_shape)


def cutlass_sparse_scaled_mm_supported(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_sparse_scaled_mm_supported(
        cuda_device_capability)


def cutlass_group_gemm_supported(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_group_gemm_supported(cuda_device_capability)


def cutlass_sparse_compress(a: torch.Tensor) \
    -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compresses a sparse matrix for use with Cutlass sparse operations.

    This function takes a dense tensor and compresses it into two components:
    non-zero elements and metadata. The compressed representation is compatible
    with Cutlass sparse kernels.

    Args:
        a (torch.Tensor):
            The input tensor to be compressed. Must have one of the following data types:
            - `torch.int8`
            - `torch.float8_e4m3fn`
            - `torch.bfloat16`
            - `torch.float16`

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            A tuple containing:
            - `a_nzs` (torch.Tensor): A tensor containing non-zero elements of `a`.
            - `a_meta` (torch.Tensor): A tensor containing metadata for the sparse representation.

    Raises:
        ValueError: If the compression operation fails.

    Notes:
        - The `a_meta` tensor has a data type of `torch.uint8`.
        - Each metadata element encodes the sparsity of 4 non-zero elements (i.e., `elemsPerMetaElem = 4`).
        - The shape of `a_nzs` is `(m, k // 2)`, where `m` and `k` are the dimensions of the input tensor.
        - The shape of `a_meta` is `(m, k // 2 // elemsPerMetaElem)`.
    """
    assert (a.dtype in [
        torch.int8, torch.float8_e4m3fn, torch.bfloat16, torch.float16
    ])
    assert (a.is_contiguous())

    # a_meta.dtype: torch.uint8 so elemsPerMetaElem = 8b / 2b_per_nz = 4
    elemsPerMetaElem = 4
    assert (a.shape[1] % (2 * elemsPerMetaElem) == 0)

    return torch.ops._C.cutlass_sparse_compress(a)


def cutlass_scaled_sparse_mm(
        a: torch.Tensor,
        bt_nzs: torch.Tensor,
        bt_meta: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Performs a scaled sparse matrix multiplication using Cutlass.

    Steps:
    1. Create a dense matrix `a` of shape (m, k) on the CUDA device:
    `a = torch.randn((m, k), device='cuda')`.

    2. Create a dense matrix `b` of shape (k, n) on the CUDA device:
    `b = torch.randn((k, n), device='cuda')`.

    3. Prune matrix `b` to 2:4 sparsity along the specified dimension:
    `b = prune_to_2_4(b, dim=0)`.

    4. Compress the transposed sparse matrix `b.t()`:
    `bt_nzs, bt_meta = cutlass_sparse_compress(b.t())`.

    5. Perform sparse matrix multiplication using the compressed matrix,
    applying scaling factors for `a` and `b`, and the output data type:
    `out = cutlass_scaled_sparse_mm(a, bt_nzs, bt_meta, scale_a, scale_b, out_dtype)`.

    Returns:
    - The result of the scaled sparse matrix multiplication.
    """
    assert (bt_nzs.shape[0] % 16 == 0 and bt_nzs.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.shape[0] == bt_nzs.shape[0] \
        and bias.dtype == out_dtype

    m = a.shape[0]
    n = bt_nzs.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_sparse_mm(out, a, bt_nzs, bt_meta, scale_a,
                                          scale_b, bias)

    return out


def get_cutlass_moe_mm_data(topk_ids: torch.Tensor,
                            expert_offsets: torch.Tensor,
                            problem_sizes1: torch.Tensor,
                            problem_sizes2: torch.Tensor,
                            input_permutation: torch.Tensor,
                            output_permutation: torch.Tensor,
                            num_experts: int,
                            n: int,
                            k: int,
                            blockscale_offsets: Optional[torch.Tensor] = None):
    """
    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in topk_ids (token-expert mapping) and uses it to
    compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation after the input is sorted with
                      input_permutation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    - input_permutation: Permutation that must be used to shuffle the input
                         before executing the MMs.
    - output_permutation: Permutation that must be used to shuffle the output
                          after executing the MMs.
    - blockscale_offsets: Optional argument passed for fp4 moe. Indices that
                          mark at which block scale index each expert begins
                          its computation. The number of block scale rows
                          computed with expert E is blockscale_offsets[E + 1] -
                          blockscale_offsets[E]
    """
    return torch.ops._C.get_cutlass_moe_mm_data(topk_ids, expert_offsets,
                                                problem_sizes1, problem_sizes2,
                                                input_permutation,
                                                output_permutation,
                                                num_experts, n, k,
                                                blockscale_offsets)


def get_cutlass_moe_mm_problem_sizes(
        topk_ids: torch.Tensor,
        problem_sizes1: torch.Tensor,
        problem_sizes2: torch.Tensor,
        num_experts: int,
        n: int,
        k: int,
        blockscale_offsets: Optional[torch.Tensor] = None):
    """
    Compute only the per-expert problem sizes needed by the two grouped matrix
    multiplications used in CUTLASS-based fused MoE.

    The function takes in topk_ids (token→expert mapping) and computes:
    - problem_sizes1, problem_sizes2: M×N×K sizes of each expert's
                                    multiplication for the two grouped MMs
                                    used in the fused MoE operation.
    """
    return torch.ops._C.get_cutlass_moe_mm_problem_sizes(
        topk_ids, problem_sizes1, problem_sizes2, num_experts, n, k,
        blockscale_offsets)


def shuffle_rows(input_tensor: torch.Tensor, dst2src_map: torch.Tensor):
    """
    Shuffle and expand the input tensor according to the dst2src_map and store the result in output_tensor.
    This is used in MoE to permute the input tensor before performing grouped matrix multiplications.
    """
    num_tokens_permuted = dst2src_map.shape[0]
    output_tensor = torch.empty((num_tokens_permuted, input_tensor.shape[1]),
                                device=input_tensor.device,
                                dtype=input_tensor.dtype)
    torch.ops._moe_C.shuffle_rows(input_tensor, dst2src_map, output_tensor)
    return output_tensor


def get_cutlass_pplx_moe_mm_data(expert_offsets: torch.Tensor,
                                 problem_sizes1: torch.Tensor,
                                 problem_sizes2: torch.Tensor,
                                 expert_num_tokens: torch.Tensor,
                                 num_local_experts: int, padded_m: int, n: int,
                                 k: int):
    """
    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in expert_num_tokens (token count per expert) and
    non_zero_expert_idxs (consecutive indices of experts with non-zero token 
    counts) and uses them to compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation.
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    """
    return torch.ops._C.get_cutlass_pplx_moe_mm_data(
        expert_offsets, problem_sizes1, problem_sizes2, expert_num_tokens,
        num_local_experts, padded_m, n, k)


def cutlass_moe_mm(out_tensors: torch.Tensor, a_tensors: torch.Tensor,
                   b_tensors: torch.Tensor, a_scales: torch.Tensor,
                   b_scales: torch.Tensor, expert_offsets: torch.Tensor,
                   problem_sizes: torch.Tensor, a_strides: torch.Tensor,
                   b_strides: torch.Tensor, c_strides: torch.Tensor,
                   per_act_token: bool, per_out_ch: bool):
    """
    A single grouped matrix multiplication used in CUTLASS-based fused MoE.
    The function executes fp8-quantized OUT = AB matrix multiplication.

    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    - a/b/c_strides: The data strides passed to grouped matrix multiplication.
    """
    return torch.ops._C.cutlass_moe_mm(out_tensors, a_tensors, b_tensors,
                                       a_scales, b_scales, expert_offsets,
                                       problem_sizes, a_strides, b_strides,
                                       c_strides, per_act_token, per_out_ch)


def cutlass_fp4_moe_mm(out_tensors: torch.Tensor, a_tensors: torch.Tensor,
                       b_tensors: torch.Tensor, a_scales: torch.Tensor,
                       b_scales: torch.Tensor, alphas: torch.Tensor,
                       problem_sizes: torch.Tensor,
                       expert_offsets: torch.Tensor, sf_offsets: torch.Tensor):
    """
    An FP4 Blockscaled Group Gemm that takes in  a_tensors, b_tensors and runs
    the gemms for each combination based on the specified problem sizes.

    This is used as the MoE gemm during NVFP4 Quantized FusedMoE forward.
    - a/b_tensors: the NVFP4 a_ptrs and b_ptrs tensors which are quantized
                     input and expert weights.
    - a_/b_scales: The blockscales in FP8-E4M3 precision
    - expert_offsets/sf_offsets: Indices that mark at which token index
                    each expert begins its computation. The number of tokens
                    computed with expert E is expert_offsets[E + 1] -
                    expert_offsets[E] And the sf_size per expert is
                    sf_offset[E+1] - sf_offset[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    """
    return torch.ops._C.cutlass_fp4_group_mm(out_tensors, a_tensors, b_tensors,
                                             a_scales, b_scales, alphas,
                                             problem_sizes, expert_offsets,
                                             sf_offsets)


# gptq_marlin
def gptq_marlin_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                       size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_repack(b_q_weight, perm, size_k, size_n,
                                           num_bits)


# gptq_marlin
def awq_marlin_repack(b_q_weight: torch.Tensor, size_k: int, size_n: int,
                      num_bits: int) -> torch.Tensor:
    return torch.ops._C.awq_marlin_repack(b_q_weight, size_k, size_n, num_bits)


def gptq_marlin_moe_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                           size_k: int, size_n: int,
                           num_bits: int) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty((num_experts, size_k // 16, size_n * (num_bits // 2)),
                         device=b_q_weight.device,
                         dtype=b_q_weight.dtype)
    for e in range(num_experts):
        output[e] = torch.ops._C.gptq_marlin_repack(b_q_weight[e], perm[e],
                                                    size_k, size_n, num_bits)
    return output


def awq_marlin_moe_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                          size_k: int, size_n: int,
                          num_bits: int) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty((num_experts, size_k // 16, size_n * (num_bits // 2)),
                         device=b_q_weight.device,
                         dtype=b_q_weight.dtype)
    for e in range(num_experts):
        output[e] = torch.ops._C.awq_marlin_repack(b_q_weight[e], size_k,
                                                   size_n, num_bits)
    return output


def gptq_marlin_gemm(a: torch.Tensor,
                     c: Optional[torch.Tensor],
                     b_q_weight: torch.Tensor,
                     b_bias: Optional[torch.Tensor],
                     b_scales: torch.Tensor,
                     global_scale: Optional[torch.Tensor],
                     b_zeros: Optional[torch.Tensor],
                     g_idx: Optional[torch.Tensor],
                     perm: Optional[torch.Tensor],
                     workspace: torch.Tensor,
                     b_q_type: ScalarType,
                     size_m: int,
                     size_n: int,
                     size_k: int,
                     is_k_full: bool = True,
                     use_atomic_add: bool = False,
                     use_fp32_reduce: bool = False,
                     is_zp_float: bool = False) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_gemm(a, c, b_q_weight, b_bias, b_scales,
                                         global_scale, b_zeros, g_idx, perm,
                                         workspace, b_q_type.id, size_m,
                                         size_n, size_k, is_k_full,
                                         use_atomic_add, use_fp32_reduce,
                                         is_zp_float)


# machete
def machete_supported_schedules(
        a_type: torch.dtype,
        b_type: ScalarType,
        group_scales_type: Optional[torch.dtype],
        group_zeros_type: Optional[torch.dtype] = None,
        channel_scales_type: Optional[torch.dtype] = None,
        token_scales_type: Optional[torch.dtype] = None,
        out_type: Optional[torch.dtype] = None) -> list[str]:
    return torch.ops._C.machete_supported_schedules(
        a_type, b_type.id, group_scales_type, group_zeros_type,
        channel_scales_type, token_scales_type, out_type)


def machete_mm(
        a: torch.Tensor,
        # b_q Should be the tensor returned by machete_prepack_B
        b_q: torch.Tensor,
        b_type: ScalarType,
        out_type: Optional[torch.dtype] = None,
        b_group_scales: Optional[torch.Tensor] = None,
        b_group_zeros: Optional[torch.Tensor] = None,
        b_group_size: Optional[int] = None,
        b_channel_scales: Optional[torch.Tensor] = None,
        a_token_scales: Optional[torch.Tensor] = None,
        schedule: Optional[str] = None) -> torch.Tensor:
    return torch.ops._C.machete_mm(a, b_q, b_type.id, out_type, b_group_scales,
                                   b_group_zeros, b_group_size,
                                   b_channel_scales, a_token_scales, schedule)


def machete_prepack_B(
        b_q_weight: torch.Tensor, a_type: torch.dtype, b_type: ScalarType,
        group_scales_type: Optional[torch.dtype]) -> torch.Tensor:
    return torch.ops._C.machete_prepack_B(b_q_weight, a_type, b_type.id,
                                          group_scales_type)


# CUTLASS W4A8
def cutlass_w4a8_mm(
        a: torch.Tensor,
        # b_q Should be the tensor returned by cutlass_encode_and_reorder_int4b
        b_q: torch.Tensor,
        b_group_scales: torch.Tensor,
        b_group_size: int,
        b_channel_scales: torch.Tensor,
        a_token_scales: torch.Tensor,
        out_type: Optional[torch.dtype] = None,
        maybe_schedule: Optional[str] = None) -> torch.Tensor:
    return torch.ops._C.cutlass_w4a8_mm(a, b_q, b_group_scales, b_group_size,
                                        b_channel_scales, a_token_scales,
                                        out_type, maybe_schedule)


def cutlass_pack_scale_fp8(scales: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.cutlass_pack_scale_fp8(scales)


def cutlass_encode_and_reorder_int4b(b: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.cutlass_encode_and_reorder_int4b(b)


if hasattr(torch.ops._C, "permute_cols"):

    @register_fake("_C::permute_cols")
    def _permute_cols_fake(a: torch.Tensor,
                           perm: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)


def permute_cols(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.permute_cols(a, perm)


# fp4
def scaled_fp4_quant(
        input: torch.Tensor,
        input_global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in the sizzled layout.
    """
    assert not current_platform.is_rocm()
    assert input.ndim >= 1, (
        f'input.ndim needs to be >= 1, but got {input.ndim}.')
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, (
        f'last dim has to be multiple of 16, but got {n}.')
    assert input.dtype in (torch.float16, torch.bfloat16), (
        f'input.dtype needs to be fp16 or bf16 but got {input.dtype}.')

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Due to the
    # requirement of the Tensor Core, the minimum tile is 128x4 for the scales.
    # So, we first pad the scales to multiples of 128 and 4. Then, the scales
    # (in float8_e4m3fn) are packed into an int32 for every 4 values. More:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x
    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)
    scale_n = n // block_size
    rounded_n = round_up(scale_n, 4)
    output_scale = torch.empty((rounded_m, rounded_n // 4),
                               device=device,
                               dtype=torch.int32)

    torch.ops._C.scaled_fp4_quant(output, input, output_scale,
                                  input_global_scale)
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


def scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale, for
    packed MoE Inputs.
    Args:
        input_tensor: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.
        expert_offsets: The expert offsets tensor
        blockscale_offsets: The blockscale offsets tensor
    Outputs:
        output: The quantized tensor in FP4
        output_scales: The blockscale tensor in FP8-E4M3
    """
    assert not current_platform.is_rocm()
    assert input_tensor.ndim == 2, (
        f'input.ndim needs to be == 2, but got {input_tensor.ndim}.')

    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE Expert Quantization. This is used to prevent the kernel
    # from running out of memory. This value can also be increased to support
    # larger models.
    MAX_TOKENS_PER_EXPERT = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE
    m_numtopk, k = input_tensor.shape

    assert (m_numtopk <= MAX_TOKENS_PER_EXPERT * topk), (
        f"m_numtopk must be less than MAX_TOKENS_PER_EXPERT("
        f"{MAX_TOKENS_PER_EXPERT})"
        f" for cutlass_moe_fp4, observed m_numtopk = {m_numtopk}. Use"
        f" VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE to set this value.")
    scales_k = k // 16
    padded_k = (scales_k + (4 - 1)) // 4

    # output is uint8 and packed fp4 values
    output = torch.empty(m_numtopk,
                         k // 2,
                         device=input_tensor.device,
                         dtype=torch.uint8)
    output_scales = torch.empty(MAX_TOKENS_PER_EXPERT * topk,
                                padded_k,
                                dtype=torch.int32,
                                device=input_tensor.device)
    torch.ops._C.scaled_fp4_experts_quant(output, output_scales, input_tensor,
                                          input_global_scale, expert_offsets,
                                          blockscale_offsets)
    output_scales = output_scales.view(torch.float8_e4m3fn)
    return output, output_scales


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
    output: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert (input.ndim == 2)
    shape: Union[tuple[int, int], torch.Size] = input.shape
    # For ROCm on MI300, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    if output is None:
        output = torch.empty(shape, device=input.device, dtype=out_dtype)
    else:
        assert num_token_padding is None, \
            "padding not supported if output passed in"
        assert output.dtype == out_dtype

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1),
                                device=input.device,
                                dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub)
        else:
            scale = torch.empty(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        assert scale.numel() == 1, f"{scale.shape}"
        torch.ops._C.static_scaled_fp8_quant(output, input, scale)

    return output, scale


# gptq allspark
def allspark_repack_weight(
        qweight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
        has_zp: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rearrange qweight, scale, and zero_point(if asymmetric) to n32k16 format
    for Ampere W8A16 Fused Gemm kernel

    Args:
        qweight: uint8 weight tensor, original k x n format.
        scale: fp16/bf16 weight scale tensor, 1 x n format.
        zero_point: fp16/bf16 weight zero_point tensor, 1 x n format.
            Must be provided for asymmetric quantization.
        has_zp: if use symmetric quantization, has_zp = False.
            if use asymmetric quantization, has_zp = True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] :
            rearranged weight, scale, and optionally zero_point.
    """
    K = qweight.shape[0]
    N = qweight.shape[1]
    N_32align = (N + 32 - 1) // 32 * 32

    qweight_reorder = torch.empty((N_32align, K),
                                  device=qweight.device,
                                  dtype=qweight.dtype)
    scale_reorder = torch.empty((1, N_32align),
                                device=scale.device,
                                dtype=scale.dtype)
    zero_point_reorder = None
    if has_zp:
        assert zero_point is not None, (
            "zero_point must be provided for asymmetric quantization.")
        zero_point_reorder = torch.empty((1, N_32align),
                                         device=zero_point.device,
                                         dtype=zero_point.dtype)

    torch.ops._C.rearrange_kn_weight_as_n32k16_order(
        qweight, scale, zero_point, has_zp, qweight_reorder, scale_reorder,
        zero_point_reorder, K, N, N_32align)

    return qweight_reorder, scale_reorder, zero_point_reorder


def allspark_w8a16_gemm(a: torch.Tensor, b_qweight: torch.Tensor,
                        b_scales: torch.Tensor,
                        b_qzeros: Optional[torch.Tensor], n: int,
                        group_size: int, sm_count: int, sm_version: int,
                        CUBLAS_M_THRESHOLD: int, has_zp: bool,
                        n32k16_reorder: bool) -> torch.Tensor:

    return torch.ops._C.allspark_w8a16_gemm(a, b_qweight, b_scales, b_qzeros,
                                            n, group_size, sm_count,
                                            sm_version, CUBLAS_M_THRESHOLD,
                                            has_zp, n32k16_reorder)


# int8
def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp
            is None), "azp must only be provided for asymmetric quantization."
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    torch.ops._C.dynamic_scaled_int8_quant(output, input.contiguous(),
                                           input_scales, input_azp)
    return output, input_scales, input_azp


# gguf
def ggml_dequantize(W: torch.Tensor, quant_type: int, m: int, n: int,
                    dtype: Optional[torch.dtype]) -> torch.Tensor:
    return torch.ops._C.ggml_dequantize(W, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_vec_a8(W, X, quant_type, row)


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_a8(W, X, quant_type, row)


def ggml_moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_moe_a8(X, W, sorted_token_ids, expert_ids,
                                    num_tokens_post_padded, quant_type, row,
                                    top_k, tokens)


def ggml_moe_a8_vec(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: torch.SymInt,
    tokens: torch.SymInt,
) -> torch.Tensor:
    return torch.ops._C.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row,
                                        tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return torch.ops._C.ggml_moe_get_block_size(quant_type)


# mamba
def selective_scan_fwd(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                       B: torch.Tensor, C: torch.Tensor,
                       D_: Optional[torch.Tensor], z_: Optional[torch.Tensor],
                       delta_bias_: Optional[torch.Tensor],
                       delta_softplus: bool,
                       query_start_loc: Optional[torch.Tensor],
                       cache_indices: Optional[torch.Tensor],
                       has_initial_state: Optional[torch.Tensor],
                       ssm_states: torch.Tensor, pad_slot_id: int):
    torch.ops._C.selective_scan_fwd(u, delta, A, B, C, D_, z_, delta_bias_,
                                    delta_softplus, query_start_loc,
                                    cache_indices, has_initial_state,
                                    ssm_states, pad_slot_id)


# ROCm skinny gemms
def LLMM1(a: torch.Tensor, b: torch.Tensor,
          rows_per_block: int) -> torch.Tensor:
    return torch.ops._rocm_C.LLMM1(a, b, rows_per_block)


def wvSplitK(a: torch.Tensor,
             b: torch.Tensor,
             cu_count: int,
             bias: torch.Tensor = None) -> torch.Tensor:
    return torch.ops._rocm_C.wvSplitK(a, b, bias, cu_count)


def wvSplitKQ(a: torch.Tensor,
              b: torch.Tensor,
              out_dtype: torch.dtype,
              scale_a: torch.Tensor,
              scale_b: torch.Tensor,
              cu_count: int,
              bias: torch.Tensor = None) -> torch.Tensor:
    out = torch.empty((b.shape[0], a.shape[0]),
                      dtype=out_dtype,
                      device=b.device)
    torch.ops._rocm_C.wvSplitKQ(a, b, bias, out, scale_a, scale_b, cu_count)
    return out


# moe
def moe_sum(input: torch.Tensor, output: torch.Tensor):
    torch.ops._moe_C.moe_sum(input, output)


def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                         block_size: int, sorted_token_ids: torch.Tensor,
                         experts_ids: torch.Tensor,
                         num_tokens_post_pad: torch.Tensor) -> None:
    torch.ops._moe_C.moe_align_block_size(topk_ids, num_experts, block_size,
                                          sorted_token_ids, experts_ids,
                                          num_tokens_post_pad)


def moe_wna16_gemm(input: torch.Tensor, output: torch.Tensor,
                   b_qweight: torch.Tensor, b_scales: torch.Tensor,
                   b_qzeros: Optional[torch.Tensor],
                   topk_weights: Optional[torch.Tensor],
                   sorted_token_ids: torch.Tensor, experts_ids: torch.Tensor,
                   num_tokens_post_pad: torch.Tensor, top_k: int,
                   BLOCK_SIZE_M: int, BLOCK_SIZE_N: int, BLOCK_SIZE_K: int,
                   bit: int) -> torch.Tensor:
    if not current_platform.is_cuda():
        raise NotImplementedError(
            "The optimized moe_wna16_gemm kernel is only "
            "available on CUDA platforms")
    torch.ops._moe_C.moe_wna16_gemm(input, output, b_qweight, b_scales,
                                    b_qzeros, topk_weights, sorted_token_ids,
                                    experts_ids, num_tokens_post_pad, top_k,
                                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                    bit)


def topk_softmax(topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 token_expert_indices: torch.Tensor,
                 gating_output: torch.Tensor) -> None:
    torch.ops._moe_C.topk_softmax(topk_weights, topk_ids, token_expert_indices,
                                  gating_output)


def grouped_topk(scores: torch.Tensor, scores_with_bias: torch.Tensor,
                 num_expert_group: int, topk_group: int, topk: int,
                 renormalize: bool, routed_scaling_factor: float):
    if not current_platform.is_cuda():
        raise NotImplementedError("The fused grouped_topk kernel is only "
                                  "available on CUDA platforms")
    return torch.ops._moe_C.grouped_topk(scores, scores_with_bias,
                                         num_expert_group, topk_group, topk,
                                         renormalize, routed_scaling_factor)


def moe_wna16_marlin_gemm(input: torch.Tensor, output: Optional[torch.Tensor],
                          b_qweight: torch.Tensor,
                          b_bias: Optional[torch.Tensor],
                          b_scales: torch.Tensor,
                          global_scale: Optional[torch.Tensor],
                          b_qzeros: Optional[torch.Tensor],
                          g_idx: Optional[torch.Tensor],
                          perm: Optional[torch.Tensor],
                          workspace: torch.Tensor,
                          sorted_token_ids: torch.Tensor,
                          expert_ids: torch.Tensor,
                          num_tokens_past_padded: torch.Tensor,
                          topk_weights: torch.Tensor, moe_block_size: int,
                          top_k: int, mul_topk_weights: bool, is_ep: bool,
                          b_q_type: ScalarType, size_m: int, size_n: int,
                          size_k: int, is_k_full: bool, use_atomic_add: bool,
                          use_fp32_reduce: bool,
                          is_zp_float: bool) -> torch.Tensor:
    return torch.ops._moe_C.moe_wna16_marlin_gemm(
        input, output, b_qweight, b_bias, b_scales, global_scale, b_qzeros,
        g_idx, perm, workspace, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, is_ep, b_q_type.id, size_m, size_n, size_k,
        is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float)


if supports_moe_ops and hasattr(torch.ops._moe_C, "marlin_gemm_moe"):

    @register_fake("_moe_C::marlin_gemm_moe")
    def marlin_gemm_moe_fake(a: torch.Tensor, b_q_weights: torch.Tensor,
                             sorted_ids: torch.Tensor,
                             topk_weights: torch.Tensor,
                             topk_ids: torch.Tensor, b_scales: torch.Tensor,
                             b_zero_points: torch.Tensor, g_idx: torch.Tensor,
                             perm: torch.Tensor, workspace: torch.Tensor,
                             b_q_type: ScalarType, size_m: torch.SymInt,
                             size_n: torch.SymInt, size_k: torch.SymInt,
                             is_k_full: bool, num_experts: int, topk: int,
                             moe_block_size: int, replicate_input: bool,
                             apply_weights: bool) -> torch.Tensor:
        return torch.empty((size_m, topk, size_n),
                           dtype=a.dtype,
                           device=a.device)

    @register_fake("_moe_C::moe_wna16_marlin_gemm")
    def moe_wna16_marlin_gemm_fake(input: torch.Tensor,
                                   output: Optional[torch.Tensor],
                                   b_qweight: torch.Tensor,
                                   b_scales: torch.Tensor,
                                   b_qzeros: Optional[torch.Tensor],
                                   g_idx: Optional[torch.Tensor],
                                   perm: Optional[torch.Tensor],
                                   workspace: torch.Tensor,
                                   sorted_token_ids: torch.Tensor,
                                   expert_ids: torch.Tensor,
                                   num_tokens_past_padded: torch.Tensor,
                                   topk_weights: torch.Tensor,
                                   moe_block_size: int, top_k: int,
                                   mul_topk_weights: bool, is_ep: bool,
                                   b_q_type: ScalarType, size_m: int,
                                   size_n: int, size_k: int, is_k_full: bool,
                                   use_atomic_add: bool, use_fp32_reduce: bool,
                                   is_zp_float: bool) -> torch.Tensor:
        return torch.empty((size_m * top_k, size_n),
                           dtype=input.dtype,
                           device=input.device)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache(key, value, key_cache,
                                             value_cache, slot_mapping,
                                             kv_cache_dtype, k_scale, v_scale)


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.reshape_and_cache_flash(key, value, key_cache,
                                                   value_cache, slot_mapping,
                                                   kv_cache_dtype, k_scale,
                                                   v_scale)


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.concat_and_cache_mla(kv_c, k_pe, kv_cache,
                                                slot_mapping, kv_cache_dtype,
                                                scale)


def copy_blocks(key_caches: list[torch.Tensor],
                value_caches: list[torch.Tensor],
                block_mapping: torch.Tensor) -> None:
    torch.ops._C_cache_ops.copy_blocks(key_caches, value_caches, block_mapping)


def copy_blocks_mla(kv_caches: list[torch.Tensor],
                    block_mapping: torch.Tensor) -> None:
    torch.ops._C_cache_ops.copy_blocks_mla(kv_caches, block_mapping)


def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    torch.ops._C_cache_ops.swap_blocks(src, dst, block_mapping)


def convert_fp8(output: torch.Tensor,
                input: torch.Tensor,
                scale: float = 1.0,
                kv_dtype: str = "fp8") -> None:
    torch.ops._C_cache_ops.convert_fp8(output, input, scale, kv_dtype)


def gather_and_maybe_dequant_cache(
        src_cache: torch.Tensor,
        dst: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        batch_size: int,
        kv_cache_dtype: str,
        scale: torch.Tensor,
        seq_starts: Optional[torch.Tensor] = None) -> None:
    torch.ops._C_cache_ops.gather_and_maybe_dequant_cache(
        src_cache, dst, block_table, cu_seq_lens, batch_size, kv_cache_dtype,
        scale, seq_starts)


def cp_gather_cache(src_cache: torch.Tensor,
                    dst: torch.Tensor,
                    block_table: torch.Tensor,
                    cu_seq_lens: torch.Tensor,
                    batch_size: int,
                    seq_starts: Optional[torch.Tensor] = None) -> None:
    torch.ops._C_cache_ops.cp_gather_cache(src_cache, dst, block_table,
                                           cu_seq_lens, batch_size, seq_starts)


def indexer_k_quant_and_cache(k: torch.Tensor, kv_cache: torch.Tensor,
                              slot_mapping: torch.Tensor,
                              quant_block_size: int,
                              kv_cache_dtype: str) -> None:
    torch.ops._C_cache_ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping,
                                                     quant_block_size,
                                                     kv_cache_dtype)


def get_device_attribute(attribute: int, device: int) -> int:
    return torch.ops._C_cuda_utils.get_device_attribute(attribute, device)


def get_max_shared_memory_per_block_device_attribute(device: int) -> int:
    # ruff: noqa: E501
    return torch.ops._C_cuda_utils.get_max_shared_memory_per_block_device_attribute(
        device)


# custom ar
def init_custom_ar(ipc_tensors: list[torch.Tensor], rank_data: torch.Tensor,
                   rank: int, fully_connected: bool) -> int:
    return torch.ops._C_custom_ar.init_custom_ar(ipc_tensors, rank_data, rank,
                                                 fully_connected)


def all_reduce(fa: int, inp: torch.Tensor, out: torch.Tensor, reg_buffer: int,
               reg_buffer_sz_bytes: int) -> None:
    torch.ops._C_custom_ar.all_reduce(fa, inp, out, reg_buffer,
                                      reg_buffer_sz_bytes)


def dispose(fa: int) -> None:
    torch.ops._C_custom_ar.dispose(fa)


def meta_size() -> int:
    return torch.ops._C_custom_ar.meta_size()


def register_buffer(fa: int, ipc_tensors: list[int]) -> None:
    return torch.ops._C_custom_ar.register_buffer(fa, ipc_tensors)


def get_graph_buffer_ipc_meta(fa: int) -> tuple[list[int], list[int]]:
    return torch.ops._C_custom_ar.get_graph_buffer_ipc_meta(fa)


def register_graph_buffers(fa: int, handles: list[list[int]],
                           offsets: list[list[int]]) -> None:
    torch.ops._C_custom_ar.register_graph_buffers(fa, handles, offsets)


def allocate_shared_buffer_and_handle(size: int) -> tuple[int, torch.Tensor]:
    return torch.ops._C_custom_ar.allocate_shared_buffer_and_handle(size)


def open_mem_handle(mem_handle: torch.Tensor):
    return torch.ops._C_custom_ar.open_mem_handle(mem_handle)


def free_shared_buffer(ptr: int) -> None:
    torch.ops._C_custom_ar.free_shared_buffer(ptr)


# quick all reduce
def init_custom_qr(rank: int,
                   world_size: int,
                   qr_max_size: Optional[int] = None) -> int:
    return torch.ops._C_custom_ar.init_custom_qr(rank, world_size, qr_max_size)


def qr_destroy(fa: int) -> None:
    torch.ops._C_custom_ar.qr_destroy(fa)


def qr_all_reduce(fa: int,
                  inp: torch.Tensor,
                  out: torch.Tensor,
                  quant_level: int,
                  cast_bf2half: bool = False) -> None:
    torch.ops._C_custom_ar.qr_all_reduce(fa, inp, out, quant_level,
                                         cast_bf2half)


def qr_get_handle(fa: int) -> torch.Tensor:
    return torch.ops._C_custom_ar.qr_get_handle(fa)


def qr_open_handles(fa: int, handles: list[torch.Tensor]) -> None:
    return torch.ops._C_custom_ar.qr_open_handles(fa, handles)


def qr_max_size() -> int:
    return torch.ops._C_custom_ar.qr_max_size()


def get_flash_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Return:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    return torch.ops._C.get_flash_mla_metadata(cache_seqlens,
                                               num_heads_per_head_k,
                                               num_heads_k)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, return by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    out, softmax_lse = torch.ops._C.flash_mla_fwd_kvcache(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse


def sm100_cutlass_mla_decode(out: torch.Tensor, lse: torch.Tensor,
                             q_nope: torch.Tensor, q_pe: torch.Tensor,
                             kv_c_and_k_pe_cache: torch.Tensor,
                             seq_lens: torch.Tensor, page_table: torch.Tensor,
                             workspace: torch.Tensor, scale: float,
                             num_kv_splits: int) -> torch.Tensor:
    torch.ops._C.sm100_cutlass_mla_decode(out, lse, q_nope, q_pe,
                                          kv_c_and_k_pe_cache, seq_lens,
                                          page_table, workspace, scale,
                                          num_kv_splits)
    return out


def sm100_cutlass_mla_get_workspace_size(max_seq_len: int, num_batches: int,
                                         sm_count: int,
                                         num_kv_splits: int) -> int:
    return torch.ops._C.sm100_cutlass_mla_get_workspace_size(
        max_seq_len, num_batches, sm_count, num_kv_splits)


if hasattr(torch.ops._C, "weight_packed_linear"):

    @register_fake("_C::weight_packed_linear")
    def weight_packed_linear_fake(mat1: torch.Tensor, mat2: torch.Tensor,
                                  bias: Optional[torch.Tensor],
                                  is_vnni: bool) -> torch.Tensor:
        return torch.empty((mat1.size(0), mat2.size(0)),
                           dtype=mat1.dtype,
                           device=mat2.device)


if hasattr(torch.ops._C, "fused_experts_cpu"):

    @register_fake("_C::fused_experts_cpu")
    def fused_experts_cpu_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool,
        use_int8_w8a8: bool,
        use_fp8_w8a16: bool,
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        block_size: Optional[list[int]],
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        is_vnni: bool,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)


if hasattr(torch.ops._C, "int8_scaled_mm_with_quant"):

    @register_fake("_C::int8_scaled_mm_with_quant")
    def int8_scaled_mm_with_quant_fake(
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        scales2: torch.Tensor,
        bias: Optional[torch.Tensor],
        out_dtype: torch.dtype,
        is_vnni: bool,
    ) -> torch.Tensor:
        M = mat1.size(0)
        N = mat2.size(0)
        return torch.empty((M, N), dtype=out_dtype)


class CPUDNNLGEMMHandler:

    def __init__(self) -> None:
        self.handler: Optional[int] = None
        self.n = -1
        self.k = -1

    def __del__(self):
        if self.handler is not None:
            torch.ops._C.release_dnnl_matmul_handler(self.handler)


if hasattr(torch.ops._C, "create_onednn_mm_handler"):
    _supports_onednn = True
else:
    _supports_onednn = False


def create_onednn_mm(
    weight: torch.Tensor,  # [K, N]
    primitive_cache_size: int = 128,
) -> CPUDNNLGEMMHandler:
    handler = CPUDNNLGEMMHandler()
    handler.k, handler.n = weight.size()
    handler.handler = torch.ops._C.create_onednn_mm_handler(
        weight, primitive_cache_size)
    return handler


def onednn_mm(
    dnnl_handler: CPUDNNLGEMMHandler,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    output = torch.empty((*x.shape[0:-1], dnnl_handler.n), dtype=x.dtype)
    torch.ops._C.onednn_mm(output, x.reshape(-1, dnnl_handler.k), bias,
                           dnnl_handler.handler)

    return output


def create_onednn_scaled_mm(
    weight: torch.Tensor,  # [K, N]
    weight_scales: torch.Tensor,
    output_type: torch.dtype,
    dynamic_quant: bool,
    use_azp: bool,
    primitive_cache_size: int = 128,
) -> CPUDNNLGEMMHandler:
    handler = CPUDNNLGEMMHandler()
    handler.k, handler.n = weight.size()
    handler.handler = torch.ops._C.create_onednn_scaled_mm_handler(
        weight, weight_scales, output_type, dynamic_quant, use_azp,
        primitive_cache_size)
    return handler


def onednn_scaled_int8_quant(input: torch.Tensor,
                             scale: Optional[torch.Tensor] = None,
                             azp: Optional[torch.Tensor] = None,
                             symmetric: bool = True):
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    token_num = input.numel() // input.shape[-1]
    input = input.view((token_num, input.shape[-1]))
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp
            is None), "azp must only be provided for asymmetric quantization."
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty((token_num, 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    torch.ops._C.dynamic_scaled_int8_quant(output, input, input_scales,
                                           input_azp)
    return output, input_scales, input_azp


def onednn_scaled_mm(
    dnnl_handler: CPUDNNLGEMMHandler,
    x: torch.Tensor,
    output: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    input_zp: Optional[torch.Tensor],
    input_zp_adj: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    torch.ops._C.onednn_scaled_mm(output, x, input_scale, input_zp,
                                  input_zp_adj, bias, dnnl_handler.handler)

    return output


def hadacore_transform(x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """
    Perform Hadamard transforms using [Hadacore](https://arxiv.org/abs/2412.08832)
    kernels. Note that these kernels exploit the recursive properties of
    Sylvester Hadamards, and therefore do not require transform weight data

    Note that sylvester hadamard transforms are also symmetric, which means that
    this function is also applies the (transpose <=> inverse) transform.
    
    :param x: value to be transformed inplace
    :param inplace: modify value in place
    :return: value after transformation
    """
    return torch.ops._C.hadacore_transform(x, inplace)


if hasattr(torch.ops._C, "hadacore_transform"):

    @register_fake("_C::hadacore_transform")
    def _hadacore_transform_fake(x: torch.Tensor,
                                 inplace: bool) -> torch.Tensor:
        return torch.empty_like(x) if not inplace else x
