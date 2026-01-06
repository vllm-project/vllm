#include "ops.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

// Register ops using STABLE_TORCH_LIBRARY for stable ABI compatibility.
// Note: We register under namespace "_C" so ops are accessible as
// torch.ops._C.<op_name> for compatibility with existing code.
STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  // Activation ops - gated activations (input: [..., 2*d] -> output: [..., d])
  m.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  m.def("mul_and_silu(Tensor! out, Tensor input) -> ()");
  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  m.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");
  m.def(
      "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float "
      "limit=7.0) -> ()");

  // Activation ops - element-wise (input: [..., d] -> output: [..., d])
  m.def("gelu_new(Tensor! out, Tensor input) -> ()");
  m.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  m.def("gelu_quick(Tensor! out, Tensor input) -> ()");

  // Utility ops
  m.def("get_cuda_view_from_cpu_tensor(Tensor! cpu_tensor) -> Tensor");
#ifndef USE_ROCM
  m.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
#endif

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  m.def(
      "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> "
      "()");

  // In-place fused Add and RMS Normalization.
  m.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");

  // Layernorm-quant
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  m.def(
      "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");

  // In-place fused Add and RMS Normalization.
  m.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, Tensor scale, float epsilon) -> ()");

  // Fused Layernorm + Quant kernels
  m.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");

  // Fused Layernorm + Block quant kernels
  m.def(
      "rms_norm_per_block_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual, int group_size, "
      "bool is_scale_transposed) -> ()");

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  m.def(
      "rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, "
      "int head_size, Tensor cos_sin_cache, bool is_neox) -> ()");

  // Function for fused QK Norm and RoPE
  m.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, int num_heads_k, "
      "int num_heads_v, int head_dim, float eps, Tensor q_weight, "
      "Tensor k_weight, Tensor cos_sin_cache, bool is_neox, "
      "Tensor position_ids) -> ()");

  // Apply repetition penalties to logits in-place
  m.def(
      "apply_repetition_penalties_(Tensor! logits, Tensor prompt_mask, "
      "Tensor output_mask, Tensor repetition_penalties) -> ()");

  // Optimized top-k per row operation
  m.def(
      "top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, "
      "Tensor! indices, int numRows, int stride0, int stride1, int topK) -> "
      "()");
  m.def(
      "top_k_per_row_decode(Tensor logits, int next_n, Tensor seqLens, "
      "Tensor! indices, int numRows, int stride0, int stride1, int topK) -> "
      "()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  // Gated activations
  m.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
  m.impl("mul_and_silu", TORCH_BOX(&mul_and_silu));
  m.impl("gelu_and_mul", TORCH_BOX(&gelu_and_mul));
  m.impl("gelu_tanh_and_mul", TORCH_BOX(&gelu_tanh_and_mul));
  m.impl("fatrelu_and_mul", TORCH_BOX(&fatrelu_and_mul));
  m.impl("swigluoai_and_mul", TORCH_BOX(&swigluoai_and_mul));

  // Element-wise activations
  m.impl("gelu_new", TORCH_BOX(&gelu_new));
  m.impl("gelu_fast", TORCH_BOX(&gelu_fast));
  m.impl("gelu_quick", TORCH_BOX(&gelu_quick));

  // Layernorm ops
  m.impl("rms_norm", TORCH_BOX(&rms_norm));
  m.impl("fused_add_rms_norm", TORCH_BOX(&fused_add_rms_norm));

  // Layernorm + quantization ops
  m.impl("rms_norm_static_fp8_quant", TORCH_BOX(&rms_norm_static_fp8_quant));
  m.impl("fused_add_rms_norm_static_fp8_quant",
         TORCH_BOX(&fused_add_rms_norm_static_fp8_quant));

  // Fused layernorm + dynamic quantization ops
  m.impl("rms_norm_dynamic_per_token_quant",
         TORCH_BOX(&rms_norm_dynamic_per_token_quant));
  m.impl("rms_norm_per_block_quant", TORCH_BOX(&rms_norm_per_block_quant));

  // Positional encoding
  m.impl("rotary_embedding", TORCH_BOX(&rotary_embedding));
  m.impl("fused_qk_norm_rope", TORCH_BOX(&fused_qk_norm_rope));

  // Sampler
  m.impl("apply_repetition_penalties_",
         TORCH_BOX(&apply_repetition_penalties_));
  m.impl("top_k_per_row_prefill", TORCH_BOX(&top_k_per_row_prefill));
  m.impl("top_k_per_row_decode", TORCH_BOX(&top_k_per_row_decode));

#ifndef USE_ROCM
  // Utility ops
  m.impl("permute_cols", TORCH_BOX(&permute_cols));
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_C, CPU, m) {
  m.impl("get_cuda_view_from_cpu_tensor",
         TORCH_BOX(&get_cuda_view_from_cpu_tensor));
}

REGISTER_EXTENSION(_C_stable_libtorch)
