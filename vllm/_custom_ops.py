# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.library

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType

logger = init_logger(__name__)

if not current_platform.is_tpu() and not current_platform.is_hpu():
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
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    torch.ops._rocm_C.paged_attention(out, exp_sum, max_logits, tmp_out, query,
                                      key_cache, value_cache, num_kv_heads,
                                      scale, block_tables, seq_lens,
                                      block_size, max_seq_len, alibi_slopes,
                                      kv_cache_dtype, k_scale, v_scale)


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    torch.ops._C.rotary_embedding(positions, query, key, head_size,
                                  cos_sin_cache, is_neox)


def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                             key: torch.Tensor, head_size: int,
                             cos_sin_cache: torch.Tensor, is_neox: bool,
                             rot_dim: int,
                             cos_sin_cache_offsets: torch.Tensor) -> None:
    torch.ops._C.batched_rotary_embedding(positions, query, key, head_size,
                                          cos_sin_cache, is_neox, rot_dim,
                                          cos_sin_cache_offsets)


# layer norm ops
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    torch.ops._C.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float) -> None:
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)


def advance_step_flashattn(num_seqs: int, num_queries: int, block_size: int,
                           input_tokens: torch.Tensor,
                           sampled_token_ids: torch.Tensor,
                           input_positions: torch.Tensor,
                           seq_lens: torch.Tensor, slot_mapping: torch.Tensor,
                           block_tables: torch.Tensor) -> None:
    """Advance a step on GPU for existing inputs for a multi-step runner"""
    return torch.ops._C.advance_step_flashattn(num_seqs, num_queries,
                                               block_size, input_tokens,
                                               sampled_token_ids,
                                               input_positions, seq_lens,
                                               slot_mapping, block_tables)


def advance_step_flashinfer(num_seqs: int, num_queries: int, block_size: int,
                            input_tokens: torch.Tensor,
                            sampled_token_ids: torch.Tensor,
                            input_positions: torch.Tensor,
                            seq_lens: torch.Tensor, slot_mapping: torch.Tensor,
                            block_tables: torch.Tensor,
                            paged_kv_indices: torch.Tensor,
                            paged_kv_indptr: torch.Tensor,
                            paged_kv_last_page_len: torch.Tensor,
                            block_table_bound: torch.Tensor) -> None:

    return torch.ops._C.advance_step_flashinfer(
        num_seqs, num_queries, block_size, input_tokens, sampled_token_ids,
        input_positions, seq_lens, slot_mapping, block_tables,
        paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len,
        block_table_bound)


# fused quant layer norm ops
def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    scale_ub: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
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


# marlin
def marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                b_scales: torch.Tensor, workspace: torch.Tensor, size_m: int,
                size_n: int, size_k: int) -> torch.Tensor:
    return torch.ops._C.marlin_gemm(a, b_q_weight, b_scales, workspace, size_m,
                                    size_n, size_k)


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
                               b_q_weight: torch.Tensor,
                               b_scales: torch.Tensor,
                               b_zeros: torch.Tensor,
                               g_idx: torch.Tensor,
                               perm: torch.Tensor,
                               workspace: torch.Tensor,
                               b_q_type: ScalarType,
                               size_m: torch.SymInt,
                               size_n: torch.SymInt,
                               size_k: torch.SymInt,
                               is_k_full: bool,
                               has_zp: bool = False,
                               use_fp32_reduce: bool = False,
                               is_zp_float: bool = False) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @register_fake("_C::marlin_qqq_gemm")
    def _marlin_qqq_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                              s_tok: torch.Tensor, s_ch: torch.Tensor,
                              s_group: torch.Tensor, workspace: torch.Tensor,
                              size_m: torch.SymInt, size_n: torch.SymInt,
                              size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n),
                           dtype=torch.float16,
                           device=a.device)

    @register_fake("_C::marlin_gemm")
    def _marlin_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                          b_scales: torch.Tensor, workspace: torch.Tensor,
                          size_m: torch.SymInt, size_n: torch.SymInt,
                          size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n),
                           dtype=torch.float16,
                           device=a.device)

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

    @register_fake("_C::aqlm_gemm")
    def _aqlm_gemm_fake(input: torch.Tensor, codes: torch.Tensor,
                        codebooks: torch.Tensor, scales: torch.Tensor,
                        codebook_partition_sizes: List[int],
                        bias: Optional[torch.Tensor]) -> torch.Tensor:
        out_features = codes.size(0) * codebooks.size(2)
        flat_input = input.reshape((-1, input.size(-1)))
        flat_output = torch.empty((flat_input.size(0), out_features),
                                  dtype=input.dtype,
                                  device=input.device)

        output_sizes = list(input.shape)
        output_sizes.pop()
        output_sizes.append(-1)
        return flat_output.reshape(tuple(output_sizes))

    @register_fake("_C::aqlm_dequant")
    def _aqlm_dequant_fake(
            codes: torch.Tensor, codebooks: torch.Tensor,
            codebook_partition_sizes: List[int]) -> torch.Tensor:
        in_features = codes.size(1) * 8
        out_features = codes.size(0)
        return torch.empty((out_features, in_features),
                           dtype=codebooks.dtype,
                           device=codebooks.device)

    @register_fake("_C::fp8_marlin_gemm")
    def _fp8_marlin_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                              b_scales: torch.Tensor, workspace: torch.Tensor,
                              num_bits: int, size_m: torch.SymInt,
                              size_n: torch.SymInt,
                              size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n), dtype=a.dtype, device=a.device)

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


if hasattr(torch.ops._C, "ggml_dequantize"):

    @register_fake("_C::ggml_dequantize")
    def _ggml_dequantize_fake(W: torch.Tensor, quant_type: int,
                              m: torch.SymInt,
                              n: torch.SymInt) -> torch.Tensor:
        return torch.empty((m, n), dtype=torch.float16, device=W.device)

    @register_fake("_C::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((1, row), dtype=torch.float16, device=W.device)

    @register_fake("_C::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        batch = X.size(0)
        return torch.empty((batch, row), dtype=torch.float16, device=W.device)


# cutlass
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
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.shape[0] == b.shape[
        1] and bias.dtype == out_dtype

    m = a.shape[0]
    n = b.shape[1]

    if current_platform.is_rocm():
        triton_scaled_mm_module = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "triton_scaled_mm")
        triton_scaled_mm = triton_scaled_mm_module.triton_scaled_mm
        return triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    return out


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
    assert azp is None or azp.numel() == a.shape[0]

    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_mm_azp(out, a, b, scale_a, scale_b, azp_adj,
                                       azp, bias)
    return out


def cutlass_sparse_scaled_mm_supported(cuda_device_capability: int) -> bool:
    return torch.ops._C.cutlass_sparse_scaled_mm_supported(
        cuda_device_capability)


def cutlass_sparse_compress(a: torch.Tensor) \
    -> Tuple[torch.Tensor, torch.Tensor]:
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
        Tuple[torch.Tensor, torch.Tensor]: 
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


# aqlm
def aqlm_gemm(input: torch.Tensor, codes: torch.Tensor,
              codebooks: torch.Tensor, scales: torch.Tensor,
              codebook_partition_sizes: List[int],
              bias: Optional[torch.Tensor]) -> torch.Tensor:
    return torch.ops._C.aqlm_gemm(input, codes, codebooks, scales,
                                  codebook_partition_sizes, bias)


def aqlm_dequant(codes: torch.Tensor, codebooks: torch.Tensor,
                 codebook_partition_sizes: List[int]) -> torch.Tensor:
    return torch.ops._C.aqlm_dequant(codes, codebooks,
                                     codebook_partition_sizes)


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
                     b_q_weight: torch.Tensor,
                     b_scales: torch.Tensor,
                     b_zeros: torch.Tensor,
                     g_idx: torch.Tensor,
                     perm: torch.Tensor,
                     workspace: torch.Tensor,
                     b_q_type: ScalarType,
                     size_m: int,
                     size_n: int,
                     size_k: int,
                     is_k_full: bool,
                     has_zp: bool = False,
                     use_fp32_reduce: bool = False,
                     is_zp_float: bool = False) -> torch.Tensor:
    return torch.ops._C.gptq_marlin_gemm(a, b_q_weight, b_scales, b_zeros,
                                         g_idx, perm, workspace, b_q_type.id,
                                         size_m, size_n, size_k, is_k_full,
                                         has_zp, use_fp32_reduce, is_zp_float)


# fp8 marlin
def fp8_marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                    b_scales: torch.Tensor, workspace: torch.Tensor,
                    num_bits: int, size_m: int, size_n: int,
                    size_k: int) -> torch.Tensor:
    return torch.ops._C.fp8_marlin_gemm(a, b_q_weight, b_scales, workspace,
                                        num_bits, size_m, size_n, size_k)


# machete
def machete_supported_schedules(
        a_type: torch.dtype,
        b_type: ScalarType,
        group_scales_type: Optional[torch.dtype],
        group_zeros_type: Optional[torch.dtype] = None,
        channel_scales_type: Optional[torch.dtype] = None,
        token_scales_type: Optional[torch.dtype] = None,
        out_type: Optional[torch.dtype] = None) -> List[str]:
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
        input_global_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
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


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert (input.ndim == 2)
    shape: Union[Tuple[int, int], torch.Size] = input.shape
    # For rocm, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype: torch.dtype = torch.float8_e4m3fnuz \
            if current_platform.is_rocm() else torch.float8_e4m3fn
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    output = torch.empty(shape, device=input.device, dtype=out_dtype)

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1),
                                device=input.device,
                                dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub)
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        # num_token_padding not implemented for this case
        assert (scale.numel() == 1 or num_token_padding is None)
        torch.ops._C.static_scaled_fp8_quant(output, input, scale)

    return output, scale


# int8
def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
      Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
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
    torch.ops._C.dynamic_scaled_int8_quant(output, input, input_scales,
                                           input_azp)
    return output, input_scales, input_azp


# qqq ops
def marlin_qqq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                    s_tok: torch.Tensor, s_ch: torch.Tensor,
                    s_group: torch.Tensor, workspace: torch.Tensor,
                    size_m: int, size_n: int, size_k: int) -> torch.Tensor:
    return torch.ops._C.marlin_qqq_gemm(a, b_q_weight, s_tok, s_ch, s_group,
                                        workspace, size_m, size_n, size_k)


# gguf
def ggml_dequantize(W: torch.Tensor, quant_type: int, m: int,
                    n: int) -> torch.Tensor:
    return torch.ops._C.ggml_dequantize(W, quant_type, m, n)


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


# mamba
def causal_conv1d_fwd(x: torch.Tensor, weight: torch.Tensor,
                      bias_: Optional[torch.Tensor],
                      conv_states: Optional[torch.Tensor],
                      query_start_loc: Optional[torch.Tensor],
                      cache_indices: Optional[torch.Tensor],
                      has_initial_state: Optional[torch.Tensor],
                      silu_activation: bool, pad_slot_id: int):
    torch.ops._C.causal_conv1d_fwd(x, weight, bias_, conv_states,
                                   query_start_loc, cache_indices,
                                   has_initial_state, silu_activation,
                                   pad_slot_id)


def causal_conv1d_update(x: torch.Tensor, conv_state: torch.Tensor,
                         weight: torch.Tensor, bias_: Optional[torch.Tensor],
                         silu_activation: bool,
                         cache_seqlens: Optional[torch.Tensor],
                         conv_state_indices: Optional[torch.Tensor],
                         pad_slot_id: int):
    torch.ops._C.causal_conv1d_update(x, conv_state, weight, bias_,
                                      silu_activation, cache_seqlens,
                                      conv_state_indices, pad_slot_id)


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


def sgl_moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                             block_size: int, sorted_token_ids: torch.Tensor,
                             experts_ids: torch.Tensor,
                             num_tokens_post_pad: torch.Tensor) -> None:
    torch.ops._moe_C.sgl_moe_align_block_size(topk_ids, num_experts,
                                              block_size, sorted_token_ids,
                                              experts_ids, num_tokens_post_pad)


def topk_softmax(topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 token_expert_indicies: torch.Tensor,
                 gating_output: float) -> None:
    torch.ops._moe_C.topk_softmax(topk_weights, topk_ids,
                                  token_expert_indicies, gating_output)


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


def copy_blocks(key_caches: List[torch.Tensor],
                value_caches: List[torch.Tensor],
                block_mapping: torch.Tensor) -> None:
    torch.ops._C_cache_ops.copy_blocks(key_caches, value_caches, block_mapping)


def copy_blocks_mla(kv_caches: List[torch.Tensor],
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


def gather_cache(src_cache: torch.Tensor,
                 dst: torch.Tensor,
                 block_table: torch.Tensor,
                 cu_seq_lens: torch.Tensor,
                 batch_size: int,
                 seq_starts: Optional[torch.Tensor] = None) -> None:
    torch.ops._C_cache_ops.gather_cache(src_cache, dst, block_table,
                                        cu_seq_lens, batch_size, seq_starts)


def get_device_attribute(attribute: int, device: int) -> int:
    return torch.ops._C_cuda_utils.get_device_attribute(attribute, device)


def get_max_shared_memory_per_block_device_attribute(device: int) -> int:
    # ruff: noqa: E501
    return torch.ops._C_cuda_utils.get_max_shared_memory_per_block_device_attribute(
        device)


# custom ar
def init_custom_ar(ipc_tensors: List[torch.Tensor], rank_data: torch.Tensor,
                   rank: int, full_nvlink: bool) -> int:
    return torch.ops._C_custom_ar.init_custom_ar(ipc_tensors, rank_data, rank,
                                                 full_nvlink)


def all_reduce(fa: int, inp: torch.Tensor, out: torch.Tensor, reg_buffer: int,
               reg_buffer_sz_bytes: int) -> None:
    torch.ops._C_custom_ar.all_reduce(fa, inp, out, reg_buffer,
                                      reg_buffer_sz_bytes)


def dispose(fa: int) -> None:
    torch.ops._C_custom_ar.dispose(fa)


def meta_size() -> int:
    return torch.ops._C_custom_ar.meta_size()


def register_buffer(fa: int, ipc_tensors: List[int]) -> None:
    return torch.ops._C_custom_ar.register_buffer(fa, ipc_tensors)


def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
    return torch.ops._C_custom_ar.get_graph_buffer_ipc_meta(fa)


def register_graph_buffers(fa: int, handles: List[List[int]],
                           offsets: List[List[int]]) -> None:
    torch.ops._C_custom_ar.register_graph_buffers(fa, handles, offsets)
