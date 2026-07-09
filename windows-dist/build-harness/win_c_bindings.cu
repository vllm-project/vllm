// torch.ops._C registration for the first batch of vLLM fused ops, built 1:1 from vLLM's own
// csrc kernels on native Windows. Schemas copied verbatim from csrc/torch_bindings.cpp.
//
// Compiled as .cu (hipcc/clang) so the inline c10::ivalue::Future helper that references
// c10::ValueError is emitted linkonce_odr and dead-stripped (it is never called on a
// single-GPU path), avoiding the LNK2019 seen when MSVC compiles this TU.
#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

TORCH_LIBRARY(_C, m) {
  m.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  m.def("rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()");
  m.impl("rms_norm", torch::kCUDA, &rms_norm);

  m.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) "
      "-> ()");
  m.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);

  m.def(
      "rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, int head_size, "
      "Tensor cos_sin_cache, bool is_neox) -> ()");
  m.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  // Native W4A16 GEMM (exllama). vLLM's ROCm mixed-precision kernel priority is
  // [Conch, Exllama]; building these lets the Exllama path run instead of the pure-Triton
  // conch fallback (which runs a fixed prefill-shaped M=128 tile, ~21x off bandwidth at M=1).
  m.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, Tensor b_gptq_scales, "
      "Tensor b_g_idx, bool use_exllama, bool use_v2_format, int bit) -> Tensor");
  m.impl("gptq_gemm", torch::kCUDA, &gptq_gemm);

  m.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");
  m.impl("gptq_shuffle", torch::kCUDA, &gptq_shuffle);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
