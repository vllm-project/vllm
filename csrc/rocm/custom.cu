#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// declare templates for front (cpp) and back (cuda) sides of function:
// template <typename T>

void LLGemm_Silu(void* in_a, void* in_b, void* out_c, const int M, const int K,
                 cudaStream_t stream, const int rows_per_block);
void LLMM_Silu(at::Tensor& in_a, at::Tensor& in_b, at::Tensor& out_c,
               const int64_t rows_per_block) {
  auto M = in_a.size(0);
  auto K = in_a.size(1);
  LLGemm_Silu(in_a.data_ptr(), in_b.data_ptr(), out_c.data_ptr(), M, K,
              at::cuda::getCurrentCUDAStream(), rows_per_block);
}
