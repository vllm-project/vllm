# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import importlib
from typing import Optional, Union

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


# These are actually defined in vllm_kernels/custom_ops.py
# but are pasesed through here for backwards compat purposes
paged_attention_v1 = custom_ops.paged_attention_v1
paged_attention_v2 = custom_ops.paged_attention_v2
paged_attention_rocm = custom_ops.paged_attention_rocm
mla_decode_kvcache_cpu = custom_ops.mla_decode_kvcache_cpu
merge_attn_states = custom_ops.merge_attn_states
convert_vertical_slash_indexes = custom_ops.convert_vertical_slash_indexes
convert_vertical_slash_indexes_mergehead = (
    custom_ops.convert_vertical_slash_indexes_mergehead
)
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
get_max_shared_memory_per_block_device_attribute = (
    custom_ops.get_max_shared_memory_per_block_device_attribute
)
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
cutlass_scaled_mm_supports_block_fp8 = (
    custom_ops.cutlass_scaled_mm_supports_block_fp8
)
cutlass_scaled_mm_azp = custom_ops.cutlass_scaled_mm_azp
cutlass_sparse_scaled_mm_supported = (
    custom_ops.cutlass_sparse_scaled_mm_supported
)
cutlass_group_gemm_supported = custom_ops.cutlass_group_gemm_supported
cutlass_sparse_compress = custom_ops.cutlass_sparse_compress
cutlass_scaled_sparse_mm = custom_ops.cutlass_scaled_sparse_mm
get_cutlass_moe_mm_data = custom_ops.get_cutlass_moe_mm_data
shuffle_rows = custom_ops.shuffle_rows
get_cutlass_pplx_moe_mm_data = custom_ops.get_cutlass_pplx_moe_mm_data
cutlass_moe_mm = custom_ops.cutlass_moe_mm
cutlass_fp4_moe_mm = custom_ops.cutlass_fp4_moe_mm
gptq_marlin_repack = custom_ops.gptq_marlin_repack
awq_marlin_repack = custom_ops.awq_marlin_repack
gptq_marlin_moe_repack = custom_ops.gptq_marlin_moe_repack
awq_marlin_moe_repack = custom_ops.awq_marlin_moe_repack
allspark_repack_weight = custom_ops.allspark_repack_weight
allspark_w8a16_gemm = custom_ops.allspark_w8a16_gemm
scaled_int8_quant = custom_ops.scaled_int8_quant
wvSplitKQ = custom_ops.wvSplitKQ


def apply_repetition_penalties_torch(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, logits.size(1)
    )
    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
    penalties = torch.where(
        prompt_mask | output_mask, repetition_penalties, 1.0
    )
    # If logits are positive, divide by penalty, otherwise multiply by penalty.
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits *= scaling


def apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """Apply repetition penalties to logits in-place.

    Args:
        logits: The logits tensor of shape [num_seqs, vocab_size].
        prompt_mask: A boolean tensor indicating which tokens appear in the prompt.
        output_mask: A boolean tensor indicating which tokens appear in the output.
        repetition_penalties: The repetition penalties of shape (num_seqs, ).
    """
    if current_platform.is_cuda() and logits.is_contiguous():
        custom_ops.apply_repetition_penalties_cuda(
            logits, prompt_mask, output_mask, repetition_penalties
        )
    else:
        apply_repetition_penalties_torch(
            logits, prompt_mask, output_mask, repetition_penalties
        )


def awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    split_k_iters: int,
    thx: int,
    thy: int,
) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import (
            awq_dequantize_triton,
        )

        return awq_dequantize_triton(qweight, scales, zeros)
    return custom_ops.awq_dequantize(
        qweight, scales, zeros, split_k_iters, thx, thy
    )


def awq_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    split_k_iters: int,
) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from vllm.model_executor.layers.quantization.awq_triton import (
            awq_gemm_triton,
        )

        return awq_gemm_triton(input, qweight, qzeros, scales, split_k_iters)
    return custom_ops.awq_gemm(input, qweight, qzeros, scales, split_k_iters)


def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert (
        bias is None or bias.shape[0] == b.shape[1] and bias.dtype == out_dtype
    )

    m = a.shape[0]
    n = b.shape[1]

    cutlass_compatible_b = b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    if current_platform.is_rocm() or not cutlass_compatible_b:
        triton_scaled_mm_module = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "triton_scaled_mm"
        )
        triton_scaled_mm = triton_scaled_mm_module.triton_scaled_mm
        return triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    custom_ops.cutlass_scaled_mm_op(out, a, b, scale_a, scale_b, bias)

    return out


gptq_marlin_gemm = custom_ops.gptq_marlin_gemm
machete_supported_schedules = custom_ops.machete_supported_schedules
machete_mm = custom_ops.machete_mm
machete_prepack_B = custom_ops.machete_prepack_B


def scaled_fp4_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return custom_ops.scaled_fp4_quant(input, input_global_scale)


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
    return custom_ops.scaled_fp4_experts_quant(
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
        topk,
    )


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
    # For ROCm on MI300, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    return custom_ops.scaled_fp8_quant(
        input,
        out_dtype,
        scale,
        num_token_padding,
        scale_ub,
        use_per_token_if_dynamic,
    )


def moe_wna16_gemm(
    input: torch.Tensor,
    output: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scales: torch.Tensor,
    b_qzeros: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    top_k: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    bit: int,
) -> torch.Tensor:
    if not current_platform.is_cuda():
        raise NotImplementedError(
            "The optimized moe_wna16_gemm kernel is only "
            "available on CUDA platforms"
        )
    return custom_ops.moe_wna16_gemm(
        input,
        output,
        b_qweight,
        b_scales,
        b_qzeros,
        topk_weights,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        top_k,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )


moe_wna16_marlin_gemm = custom_ops.moe_wna16_marlin_gemm
