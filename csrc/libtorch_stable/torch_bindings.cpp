#include "ops.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

// Register ops with STABLE_TORCH_LIBRARY for libtorch stable ABI compatibility.
// Note: We register under namespace "_C" so ops are accessible as
// torch.ops._C.<op_name> for compatibility with existing code.
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
#ifndef USE_ROCM
  ops.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
#endif

#ifndef USE_ROCM
  // Compute per-token-group FP8 quantized tensor and scaling factor.
  // The dummy arguments are here so we can correctly fuse with RMSNorm.
  ops.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, "
      "int group_size, float eps, float fp8_min, float fp8_max, bool "
      "scale_ue8m0, bool dummy_is_scale_transposed, bool dummy_is_tma_aligned "
      ") -> ()");
  // Compute per-token-group 8-bit quantized tensor and UE8M0-packed,
  // TMA-aligned scales for DeepGEMM.
  ops.def(
      "per_token_group_fp8_quant_packed(Tensor input, Tensor! output_q, "
      "Tensor! output_s_packed, int group_size, float eps, float fp8_min, "
      "float fp8_max) -> ()");
  // Compute per-token-group INT8 quantized tensor and scaling factor.
  ops.def(
      "per_token_group_quant_int8(Tensor input, Tensor! output_q, Tensor! "
      "output_s, int group_size, float eps, float int8_min, float int8_max) -> "
      "()");

  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column
  // quantization, as well as bias
  ops.def(
      "cutlass_scaled_mm(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor? bias) -> ()");

  // CUTLASS w8a8 GEMM, supporting asymmetric per-tensor or per-row/column
  // quantization.
  ops.def(
      "cutlass_scaled_mm_azp(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor azp_adj,"
      "                  Tensor? azp, Tensor? bias) -> ()");

  // Check if cutlass scaled_mm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool");

  // Check if cutlass grouped gemm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_group_gemm_supported(int cuda_device_capability) -> bool");

  // CUTLASS w8a8 grouped GEMM
  ops.def(
      "cutlass_moe_mm(Tensor! out_tensors, Tensor a_tensors, Tensor b_tensors, "
      "               Tensor a_scales, Tensor b_scales, Tensor expert_offsets, "
      "               Tensor problem_sizes, Tensor a_strides, "
      "               Tensor b_strides, Tensor c_strides, bool per_act_token, "
      "               bool per_out_ch) -> ()");

  // A function that computes data required to run fused MoE with w8a8 grouped
  // GEMM. It takes topk_ids as an input, and computes expert_offsets
  // (token start indices of each expert). In addition to this, it computes
  // problem sizes for each expert's multiplication used by the two mms called
  // from fused MoE operation, and arrays with permutations required to shuffle
  // and de-shuffle the input/output of the fused operation.
  ops.def(
      "get_cutlass_moe_mm_data(Tensor topk_ids, Tensor! expert_offsets, "
      "                        Tensor! problem_sizes1, Tensor! problem_sizes2, "
      "                        Tensor! input_permutation, "
      "                        Tensor! output_permutation, int num_experts, "
      "                        int n, int k, Tensor? blockscale_offsets, "
      "                        bool is_gated) -> ()");

  // compute per-expert problem sizes from expert_first_token_offset
  // produced by vLLM's moe_permute kernel
  ops.def(
      "get_cutlass_moe_mm_problem_sizes_from_expert_offsets("
      "    Tensor expert_first_token_offset, "
      "    Tensor! problem_sizes1, "
      "    Tensor! problem_sizes2, "
      "    int n, int k, bool swap_ab) -> ()");

  // A function that computes data required to run fused MoE with w8a8 grouped
  // GEMM in batched expert format. It takes expert_num_tokens
  // as an input, and computes expert_offsets (token start indices of each
  // expert). In addition to this, it computes problem sizes for each expert's
  // multiplication used by the two mms called from fused MoE operation.
  ops.def(
      "get_cutlass_batched_moe_mm_data(Tensor! expert_offsets, "
      "                             Tensor! problem_sizes1, "
      "                             Tensor! problem_sizes2, "
      "                             Tensor expert_num_tokens, "
      "                             int num_local_experts, int padded_m, "
      "                             int n, int k) -> ()");

  // Check if cutlass scaled_mm supports block quantization (used by DeepSeekV3)
  ops.def(
      "cutlass_scaled_mm_supports_block_fp8(int cuda_device_capability) -> "
      "bool");

  // CUTLASS nvfp4 block scaled GEMM
  ops.def(
      "cutlass_scaled_fp4_mm(Tensor! out, Tensor a, Tensor b,"
      "                      Tensor block_scale_a, Tensor block_scale_b,"
      "                      Tensor alpha) -> ()");

  // cutlass nvfp4 block scaled group GEMM
  ops.def(
      "cutlass_fp4_group_mm(Tensor! out, Tensor a, Tensor b,"
      " Tensor a_blockscale, Tensor b_blockscales, Tensor alphas,"
      " Tensor problem_sizes, Tensor expert_offsets, Tensor sf_offsets) -> ()");

  // cutlass mxfp4 block scaled group GEMM (MXFP4 x MXFP4 MoE)
  ops.def(
      "cutlass_mxfp4_group_mm(Tensor! out, Tensor a, Tensor b,"
      " Tensor a_blockscale, Tensor b_blockscales,"
      " Tensor problem_sizes, Tensor expert_offsets, Tensor sf_offsets) -> ()");

  // Compute NVFP4 block quantized tensor.
  ops.def(
      "scaled_fp4_quant(Tensor input,"
      "                 Tensor input_scale, bool "
      "is_sf_swizzled_layout) -> (Tensor, Tensor)");

  // Out variant
  // TODO: Add out_variant tag once PyTorch supports it (added in 2.11)
  // This registration is now migrated to stable ABI
  // at::Tag::out_variant is not available in the stable ABI (enum_tag.h is not
  // yet in torch/headeronly), the tag should be applied from Python
  // via torch.library.Library.define(..., tags=(torch.Tag.out_variant,))
  // with the .impl remaining in C++.
  // See pytorch/pytorch#176117.
  ops.def(
      "scaled_fp4_quant.out(Tensor input,"
      "                     Tensor input_scale, bool "
      "is_sf_swizzled_layout, *, Tensor(a!) output, Tensor(b!) output_scale) "
      "-> ()");

  // Compute NVFP4 experts quantization.
  ops.def(
      "scaled_fp4_experts_quant(Tensor! output, Tensor! output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts) -> ()");

  // Fused SiLU+Mul+NVFP4 experts quantization.
  ops.def(
      "silu_and_mul_scaled_fp4_experts_quant(Tensor! output, Tensor! "
      "output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts) -> ()");

  // Compute MXFP4 experts quantization (32-element blocks, E8M0 SFs).
  ops.def(
      "mxfp4_experts_quant(Tensor! output, Tensor! output_scale,"
      "Tensor input, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts, int n_experts) -> ()");

  // Fused SiLU+Mul+MXFP4 experts quantization.
  ops.def(
      "silu_and_mul_mxfp4_experts_quant(Tensor! output, Tensor! "
      "output_scale,"
      "Tensor input, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts, int n_experts) -> ()");

  // Fused SiLU+Mul+NVFP4 quantization.
  ops.def(
      "silu_and_mul_nvfp4_quant(Tensor! result, Tensor! result_block_scale, "
      "Tensor input, Tensor input_global_scale) -> ()");

  // Check if cutlass_scaled_mm_fp4 is supported for CUDA devices
  // of the given capability
  ops.def("cutlass_scaled_mm_supports_fp4(int cuda_device_capability) -> bool");

  // CUTLASS w4a8 GEMM
  ops.def(
      "cutlass_w4a8_mm("
      "   Tensor A,"
      "   Tensor B,"
      "   Tensor group_scales,"
      "   int    group_size,"
      "   Tensor channel_scales,"
      "   Tensor token_scales,"
      "   ScalarType? out_type,"
      "   str?   maybe_schedule"
      ") -> Tensor");

  // pack scales
  ops.def("cutlass_pack_scale_fp8(Tensor scales) -> Tensor");

  // encode and reorder weight matrix
  ops.def("cutlass_encode_and_reorder_int4b(Tensor B) -> Tensor");

  // CUTLASS w4a8 grouped GEMM
  ops.def(
      "cutlass_w4a8_moe_mm("
      "   Tensor! out_tensors,"
      "   Tensor a_tensors,"
      "   Tensor b_tensors,"
      "   Tensor a_scales,"
      "   Tensor b_scales,"
      "   Tensor b_group_scales,"
      "   int b_group_size,"
      "   Tensor expert_offsets,"
      "   Tensor problem_sizes,"
      "   Tensor a_strides,"
      "   Tensor b_strides,"
      "   Tensor c_strides,"
      "   Tensor group_scale_strides,"
      "   str? maybe_schedule"
      ") -> ()");

  ops.def(
      "cutlass_encode_and_reorder_int4b_grouped(Tensor b_tensors) -> (Tensor, "
      "Tensor)");
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
#ifndef USE_ROCM
  ops.impl("permute_cols", TORCH_BOX(&permute_cols));
#endif

#ifndef USE_ROCM
  // Per-token group quantization
  ops.impl("per_token_group_fp8_quant", TORCH_BOX(&per_token_group_quant_fp8));
  ops.impl("per_token_group_fp8_quant_packed",
           TORCH_BOX(&per_token_group_quant_8bit_packed));
  ops.impl("per_token_group_quant_int8",
           TORCH_BOX(&per_token_group_quant_int8));

  // CUTLASS scaled_mm ops
  ops.impl("cutlass_scaled_mm", TORCH_BOX(&cutlass_scaled_mm));
  ops.impl("cutlass_scaled_mm_azp", TORCH_BOX(&cutlass_scaled_mm_azp));
  ops.impl("cutlass_moe_mm", TORCH_BOX(&cutlass_moe_mm));
  ops.impl("get_cutlass_moe_mm_data", TORCH_BOX(&get_cutlass_moe_mm_data));
  ops.impl("get_cutlass_moe_mm_problem_sizes_from_expert_offsets",
           TORCH_BOX(&get_cutlass_moe_mm_problem_sizes_from_expert_offsets));
  ops.impl("get_cutlass_batched_moe_mm_data",
           TORCH_BOX(&get_cutlass_batched_moe_mm_data));

  // FP4/NVFP4 ops
  ops.impl("cutlass_scaled_fp4_mm", TORCH_BOX(&cutlass_scaled_fp4_mm));
  ops.impl("scaled_fp4_quant", TORCH_BOX(&scaled_fp4_quant_func));
  ops.impl("scaled_fp4_quant.out", TORCH_BOX(&scaled_fp4_quant_out));
  ops.impl("scaled_fp4_experts_quant", TORCH_BOX(&scaled_fp4_experts_quant));
  ops.impl("silu_and_mul_scaled_fp4_experts_quant",
           TORCH_BOX(&silu_and_mul_scaled_fp4_experts_quant));
  ops.impl("silu_and_mul_nvfp4_quant", TORCH_BOX(&silu_and_mul_nvfp4_quant));
  // mxfp4_experts_quant: registered in mxfp4_experts_quant.cu (SM100 only).
  // W4A8 ops: registered in w4a8_mm_entry.cu / w4a8_grouped_mm_entry.cu.
#endif
}

// These capability-check functions take only primitive args (no tensors), so
// there is no device to dispatch on. CompositeExplicitAutograd makes them
// available for all backends. This is the stable ABI equivalent of calling
// ops.impl("op_name", &func) without a dispatch key in the non-stable API.
STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, ops) {
#ifndef USE_ROCM
  ops.impl("cutlass_scaled_mm_supports_fp8",
           TORCH_BOX(&cutlass_scaled_mm_supports_fp8));
  ops.impl("cutlass_group_gemm_supported",
           TORCH_BOX(&cutlass_group_gemm_supported));
  ops.impl("cutlass_scaled_mm_supports_block_fp8",
           TORCH_BOX(&cutlass_scaled_mm_supports_block_fp8));
  ops.impl("cutlass_scaled_mm_supports_fp4",
           TORCH_BOX(&cutlass_scaled_mm_supports_fp4));
#endif
}

REGISTER_EXTENSION(_C_stable_libtorch)
