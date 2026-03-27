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
#endif
}

REGISTER_EXTENSION(_C_stable_libtorch)
