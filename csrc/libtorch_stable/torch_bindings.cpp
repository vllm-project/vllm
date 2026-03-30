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
#endif
}

REGISTER_EXTENSION(_C_stable_libtorch)
