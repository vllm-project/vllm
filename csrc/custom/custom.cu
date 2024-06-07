#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

namespace py = pybind11;

// declare templates for front (cpp) and back (cuda) sides of function:
// template <typename T>

void LLGemm_Silu(void* in_a, void* in_b, void* out_c, const int M, const int K,
                 cudaStream_t stream, const int rows_per_block);
void LLMM_Silu(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c,
               const int rows_per_block) {
  int M = in_a.size(0);
  int K = in_a.size(1);
  LLGemm_Silu(in_a.data_ptr(), in_b.data_ptr(), out_c.data_ptr(), M, K,
              at::cuda::getCurrentCUDAStream(), rows_per_block);
}

void LLGemm1(void* in_a, void* in_b, void* out_c, const int M, const int K,
             cudaStream_t stream, const int rows_per_block);

// template <typename T>
void LLMM1(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c,
           const int rows_per_block = 4) {
  int M = in_a.size(0);
  int K = in_a.size(1);
  // if (N != in_b.numel())
  //         throw std::invalid_argument("Size mismatch A.numel(): " +
  //         std::to_string(in_a.numel())
  //                           + ", B.numel(): " +
  //                           std::to_string(in_b.numel()));

  // out_c.resize_({N});

  // call the kernel function...
  LLGemm1(in_a.data_ptr(), in_b.data_ptr(), out_c.data_ptr(), M, K,
          at::cuda::getCurrentCUDAStream(), rows_per_block);
}

void LLGemmZZ(void* in_a, void* in_b, void* out_c, const int M, const int K,
              cudaStream_t stream, const int solidx);

void LLZZ(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c,
          const int solidx = 0) {
  int M = in_a.size(0);
  int K = in_a.size(1);

  LLGemmZZ(in_a.data_ptr(), in_b.data_ptr(), out_c.data_ptr(), M, K,
           at::cuda::getCurrentCUDAStream(), solidx);
}
// instantiate the CPP template for T=float:
// template void AddGPU<float>(at::Tensor in_a, at::Tensor in_b, at::Tensor
// out_c);

void MMGPUKernel(float* in_a, float* in_b, float* out_c, int numARows,
                 int numAColumns, int numBRows, int numBColumns, int numCRows,
                 int numCColumns, cudaStream_t stream);

void MMCustomGPU(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c) {
  auto matA_sizes{in_a.sizes()};
  auto matB_sizes{in_b.sizes()};
  auto matO_sizes{out_c.sizes()};
  MMGPUKernel(in_a.data_ptr<float>(), in_b.data_ptr<float>(),
              out_c.data_ptr<float>(), matA_sizes[0], matA_sizes[1],
              matB_sizes[0], matB_sizes[1], matO_sizes[0], matO_sizes[1],
              at::cuda::getCurrentCUDAStream());
}

void paged_attention_custom(torch::Tensor& out, torch::Tensor& exp_sums,
                            torch::Tensor& max_logits, torch::Tensor& tmp_out,
                            torch::Tensor& query, torch::Tensor& key_cache,
                            torch::Tensor& value_cache, int num_kv_heads,
                            float scale, torch::Tensor& block_tables,
                            torch::Tensor& context_lens, int block_size,
                            int max_context_len,
#if 0
  torch::Tensor& qk_out,
  torch::Tensor& softmax_out,
#endif
                            const c10::optional<torch::Tensor>& alibi_slopes,
                            const std::string& kv_cache_dtype);

// declare the extension module with the AddGPU function:
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "pybind11 example plugin";
  m.def("LLMM1", &LLMM1);
  m.def("LLMM_Silu", &LLMM_Silu);
  m.def("LLZZ", &LLZZ);
  m.def("paged_attention_custom", &paged_attention_custom,
        "PagedAttention LL4Mi Custom.");
  // m.def("MMCustomGPU", &MMCustomGPU);
}
