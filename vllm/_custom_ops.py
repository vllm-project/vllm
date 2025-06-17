# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import importlib
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.library
import vllm_kernels.custom_ops as custom_ops

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

# These are actually defined in vllm_kernels/custom_ops.py
# but are pasesed through here for backwards compat purposes
paged_attention_v1 = custom_ops.paged_attention_v1
paged_attention_v2 = custom_ops.paged_attention_v2
paged_attention_rocm = custom_ops.paged_attention_rocm
mla_decode_kvcache_cpu = custom_ops.mla_decode_kvcache_cpu
merge_attn_states = custom_ops.merge_attn_states
convert_vertical_slash_indexes = custom_ops.convert_vertical_slash_indexes
convert_vertical_slash_indexes_mergehead = custom_ops.convert_vertical_slash_indexes_mergehead
rotary_embedding = custom_ops.rotary_embedding
rms_norm = custom_ops.rms_norm
fused_add_rms_norm = custom_ops.fused_add_rms_norm
apply_repetition_penalties_cuda = custom_ops.apply_repetition_penalties_cuda
advance_step_flashattn = custom_ops.advance_step_flashattn
advance_step_flashinfer = custom_ops.advance_step_flashinfer
rms_norm_dynamic_per_token_quant = custom_ops.rms_norm_dynamic_per_token_quant
gptq_gemm = custom_ops.gptq_gemm
gptq_shuffle = custom_ops.gptq_shuffle
marlin_gemm = custom_ops.marlin_gemm
gptq_marlin_24_gemm = custom_ops.gptq_marlin_24_gemm
aqlm_gemm = custom_ops.aqlm_gemm
aqlm_dequant = custom_ops.aqlm_dequant
permute_cols = custom_ops.permute_cols
marlin_qqq_gemm = custom_ops.marlin_qqq_gemm
ggml_dequantize = custom_ops.ggml_dequantize
ggml_mul_mat_vec_a8 = custom_ops.ggml_mul_mat_vec_a8
ggml_mul_mat_a8 = custom_ops.ggml_mul_mat_a8
ggml_moe_a8 = custom_ops.ggml_moe_a8
ggml_moe_a8_vec = custom_ops.ggml_moe_a8_vec
ggml_moe_get_block_size = custom_ops.ggml_moe_get_block_size
causal_conv1d_fwd = custom_ops.causal_conv1d_fwd
causal_conv1d_update = custom_ops.causal_conv1d_update
selective_scan_fwd = custom_ops.selective_scan_fwd
LLMM1 = custom_ops.LLMM1
wvSplitK = custom_ops.wvSplitK
moe_sum = custom_ops.moe_sum
moe_align_block_size = custom_ops.moe_align_block_size
sgl_moe_align_block_size = custom_ops.sgl_moe_align_block_size
topk_softmax = custom_ops.topk_softmax
reshape_and_cache = custom_ops.reshape_and_cache
reshape_and_cache_flash = custom_ops.reshape_and_cache_flash
concat_and_cache_mla = custom_ops.concat_and_cache_mla
copy_blocks = custom_ops.copy_blocks
copy_blocks_mla = custom_ops.copy_blocks_mla
swap_blocks = custom_ops.swap_blocks
convert_fp8 = custom_ops.convert_fp8
gather_cache = custom_ops.gather_cache
get_device_attribute = custom_ops.get_device_attribute
get_max_shared_memory_per_block_device_attribute = custom_ops.get_max_shared_memory_per_block_device_attribute
init_custom_ar = custom_ops.init_custom_ar
all_reduce = custom_ops.all_reduce
dispose = custom_ops.dispose
meta_size = custom_ops.meta_size
register_buffer = custom_ops.register_buffer
get_graph_buffer_ipc_meta = custom_ops.get_graph_buffer_ipc_meta
register_graph_buffers = custom_ops.register_graph_buffers
allocate_shared_buffer_and_handle = custom_ops.allocate_shared_buffer_and_handle
open_mem_handle = custom_ops.open_mem_handle
free_shared_buffer = custom_ops.free_shared_buffer
get_flash_mla_metadata = custom_ops.get_flash_mla_metadata
flash_mla_with_kvcache = custom_ops.flash_mla_with_kvcache
cutlass_mla_decode = custom_ops.cutlass_mla_decode
cutlass_scaled_mm_supports_fp4 = custom_ops.cutlass_scaled_mm_supports_fp4
cutlass_scaled_fp4_mm = custom_ops.cutlass_scaled_fp4_mm
cutlass_scaled_mm_supports_fp8 = custom_ops.cutlass_scaled_mm_supports_fp8
cutlass_scaled_mm_supports_block_fp8 = custom_ops.cutlass_scaled_mm_supports_block_fp8


# Rule 3: Does not use torch.ops._C.*
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


# Rule 2: Uses current_platform and custom_ops (vllm imports)
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
    if current_platform.is_cuda() and logits.is_contiguous():
        custom_ops.apply_repetition_penalties_cuda(logits, prompt_mask,
                                                   output_mask,
                                                   repetition_penalties)
    else:
        apply_repetition_penalties_torch(logits, prompt_mask, output_mask,
                                         repetition_penalties)


# Rule 2: Uses envs.VLLM_USE_TRITON_AWQ and custom_ops (vllm imports)
def awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                   zeros: torch.Tensor, split_k_iters: int, thx: int,
                   thy: int) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import (
            awq_dequantize_triton)
        return awq_dequantize_triton(qweight, scales, zeros)
    return custom_ops.awq_dequantize(qweight, scales, zeros, split_k_iters,
                                     thx, thy)


# Rule 2: Uses envs.VLLM_USE_TRITON_AWQ and custom_ops (vllm imports)
def awq_gemm(input: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
             scales: torch.Tensor, split_k_iters: int) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import (
            awq_gemm_triton)
        return awq_gemm_triton(input, qweight, qzeros, scales, split_k_iters)
    return custom_ops.awq_gemm(input, qweight, qzeros, scales, split_k_iters)


# quantization ops
# awq


# TODO: Migrate the rest of the functions in here to utilize things from `vllm_kernels.custom_ops`
#       Rules for the migration:
#           1. If a function is simple (i.e. only utilizes torch.ops._C.*) then prefer to structure it like so:
#               paged_attention_v1 = custom_ops.paged_attention_v1
#           2. If a function utilizes anything from the main vllm (like current_platform or ScalarType) prefer
#               to write wrapper functions, similar to what we're doing with awq_gemm
#               a. Make changes in `vllm-kernels/vllm_kernels/custom_ops.py` if you need to to make it easier on yourself
#           3. If a function does not use torch.ops._C.* then leave it alone
#           4. Don't assume you have to do everything at once, write down the list of functions you need to
#               migrate and do them (at most) 3 at a time and try to validate you made the write moves
#           5. Prefer to make the most minimal of changes possible

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
                        codebook_partition_sizes: list[int],
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
            codebook_partition_sizes: list[int]) -> torch.Tensor:
        in_features = codes.size(1) * 8
        out_features = codes.size(0)
        return torch.empty((out_features, in_features),
                           dtype=codebooks.dtype,
                           device=codebooks.device)

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
        return torch.empty((1, row), dtype=X.dtype, device=W.device)

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
# cutlass support functions (migrated to custom_ops)


# Rule 2: Uses current_platform and importlib (vllm imports)
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
    assert bias is None or bias.shape[0] == b.shape[
        1] and bias.dtype == out_dtype

    m = a.shape[0]
    n = b.shape[1]

    cutlass_compatible_b = (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    if current_platform.is_rocm() or not cutlass_compatible_b:
        triton_scaled_mm_module = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "triton_scaled_mm")
        triton_scaled_mm = triton_scaled_mm_module.triton_scaled_mm
        return triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    return out


# Rule 1: Only uses torch.ops._C.*
cutlass_scaled_mm_azp = custom_ops.cutlass_scaled_mm_azp


# Rule 1: Only uses torch.ops._C.*
cutlass_sparse_scaled_mm_supported = custom_ops.cutlass_sparse_scaled_mm_supported


# Rule 1: Only uses torch.ops._C.*
cutlass_group_gemm_supported = custom_ops.cutlass_group_gemm_supported

# Rule 1: Only uses torch.ops._C.*
cutlass_sparse_compress = custom_ops.cutlass_sparse_compress


# Rule 1: Only uses torch.ops._C.*
cutlass_scaled_sparse_mm = custom_ops.cutlass_scaled_sparse_mm


# Rule 1: Only uses torch.ops._C.*
get_cutlass_moe_mm_data = custom_ops.get_cutlass_moe_mm_data


# Rule 1: Only uses torch.ops._moe_C.*
shuffle_rows = custom_ops.shuffle_rows


# Rule 1: Only uses torch.ops._C.*
get_cutlass_pplx_moe_mm_data = custom_ops.get_cutlass_pplx_moe_mm_data


# Rule 1: Only uses torch.ops._C.*
cutlass_moe_mm = custom_ops.cutlass_moe_mm


# Rule 1: Only uses torch.ops._C.*
cutlass_fp4_moe_mm = custom_ops.cutlass_fp4_moe_mm


# aqlm (migrated to custom_ops)


# gptq_marlin
# Rule 1: Only uses torch.ops._C.*
gptq_marlin_repack = custom_ops.gptq_marlin_repack


# gptq_marlin
# Rule 1: Only uses torch.ops._C.*
awq_marlin_repack = custom_ops.awq_marlin_repack


# Rule 1: Only uses torch.ops._C.*
gptq_marlin_moe_repack = custom_ops.gptq_marlin_moe_repack


# Rule 1: Only uses torch.ops._C.*
awq_marlin_moe_repack = custom_ops.awq_marlin_moe_repack


# Rule 2: Uses ScalarType (vllm import)
def gptq_marlin_gemm(a: torch.Tensor,
                     c: Optional[torch.Tensor],
                     b_q_weight: torch.Tensor,
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
    return torch.ops._C.gptq_marlin_gemm(a, c, b_q_weight, b_scales,
                                         global_scale, b_zeros, g_idx, perm,
                                         workspace, b_q_type.id, size_m,
                                         size_n, size_k, is_k_full,
                                         use_atomic_add, use_fp32_reduce,
                                         is_zp_float)


# machete
# Rule 2: Uses ScalarType (vllm import)
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


# Rule 2: Uses ScalarType (vllm import)
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


# Rule 2: Uses ScalarType (vllm import)
def machete_prepack_B(
        b_q_weight: torch.Tensor, a_type: torch.dtype, b_type: ScalarType,
        group_scales_type: Optional[torch.dtype]) -> torch.Tensor:
    return torch.ops._C.machete_prepack_B(b_q_weight, a_type, b_type.id,
                                          group_scales_type)


# permute_cols (migrated to custom_ops)


# fp4
# Rule 2: Uses current_platform (vllm import)
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


# Rule 2: Uses current_platform and envs (vllm imports)
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
# Rule 2: Uses current_platform (vllm import)
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
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


# gptq allspark
# Rule 1: Only uses torch.ops._C.*
allspark_repack_weight = custom_ops.allspark_repack_weight


# Rule 1: Only uses torch.ops._C.*
allspark_w8a16_gemm = custom_ops.allspark_w8a16_gemm


# int8
# Rule 1: Only uses torch.ops._C.*
scaled_int8_quant = custom_ops.scaled_int8_quant


# qqq ops (migrated to custom_ops)

# gguf (migrated to custom_ops)


# ggml_mul_mat_a8 (migrated to custom_ops)


# ggml_moe functions (migrated to custom_ops)


# mamba (migrated to custom_ops)


# ROCm skinny gemms (migrated to custom_ops)


# Rule 1: Only uses torch.ops._rocm_C.*
wvSplitKQ = custom_ops.wvSplitKQ


# moe (simple functions migrated to custom_ops)


# Rule 2: Uses current_platform (vllm import)
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


# topk_softmax (migrated to custom_ops)


# Rule 2: Uses ScalarType (vllm import)
def moe_wna16_marlin_gemm(input: torch.Tensor, output: Optional[torch.Tensor],
                          b_qweight: torch.Tensor, b_scales: torch.Tensor,
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
        input, output, b_qweight, b_scales, global_scale, b_qzeros, g_idx,
        perm, workspace, sorted_token_ids, expert_ids, num_tokens_past_padded,
        topk_weights, moe_block_size, top_k, mul_topk_weights, is_ep,
        b_q_type.id, size_m, size_n, size_k, is_k_full, use_atomic_add,
        use_fp32_reduce, is_zp_float)


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


# cache operations (migrated to custom_ops)


# device utility functions (migrated to custom_ops)


# custom_ar (migrated to custom_ops)


# get_flash_mla_metadata (migrated to custom_ops)


# flash_mla_with_kvcache and cutlass_mla_decode (migrated to custom_ops)
