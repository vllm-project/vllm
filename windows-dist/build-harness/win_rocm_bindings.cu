// torch.ops._rocm_C registration for vLLM's rocm/ skinny GEMMs, built 1:1 from
// csrc/rocm/skinny_gemms.cu on native Windows ROCm (gfx1100). Only the ops that have a working
// RDNA3 (__HIP__GFX1X__) path are bound: LLMM1 (mat-vec) and wvSplitK (skinny mat-mat) -- exactly
// what vLLM's rocm_unquantized_gemm wants for the M=1 dense path (VLLM_ROCM_USE_SKINNY_GEMM=1).
// wvSplitKrc (gfx950) and wvSplitKQ (fp8/gfx12) are intentionally NOT bound here.
// Schemas copied verbatim from csrc/rocm/torch_bindings.cpp.
#include <torch/extension.h>
#include <torch/library.h>

#include "rocm/ops.h"

TORCH_LIBRARY(_rocm_C, m) {
  m.def("LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> Tensor");
  m.impl("LLMM1", torch::kCUDA, &LLMM1);

  m.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  m.impl("wvSplitK", torch::kCUDA, &wvSplitK);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
