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

void LLGemm1(void* in_a, void* in_b, void* out_c, const int M, const int K,
             cudaStream_t stream, const int rows_per_block);

// template <typename T>
void LLMM1(at::Tensor& in_a, at::Tensor& in_b, at::Tensor& out_c,
           const int64_t rows_per_block) {
  auto M = in_a.size(0);
  auto K = in_a.size(1);
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

void wvSpltK_(void* in_a, void* in_b, void* out_c, const int M, const int K,
              const int N, cudaStream_t stream, const int CuCount);

void wvSpltK(at::Tensor& in_a, at::Tensor& in_b, at::Tensor& out_c,
             const int64_t N_in, const int64_t CuCount) {
  auto M = in_a.size(0);
  auto K = in_a.size(1);
  int N = N_in;
  wvSpltK_(in_a.data_ptr(), in_b.data_ptr(), out_c.data_ptr(), M, K, N,
           at::cuda::getCurrentCUDAStream(), CuCount);
}

void wvSpltKQ_(void* in_a, void* in_b, void* out_c, void* scale_a,
               void* scale_b, const int M, const int K, const int Kp,
               const int N, const int Otp_in, cudaStream_t stream,
               const int CuCount);

void wvSpltKQ(at::Tensor& in_a, at::Tensor& in_b, at::Tensor& out_c,
              at::Tensor& scale_a, at::Tensor& scale_b, const int64_t N_in,
              const int64_t Otp_in, const int64_t CuCount) {
  auto M = in_a.size(0);
  auto K = in_a.size(1);
  auto Kp = in_a.stride(0);
  int N = N_in;
  int Otp = Otp_in;
  wvSpltKQ_(in_a.data_ptr(), in_b.data_ptr(), out_c.data_ptr(),
            scale_a.data_ptr(), scale_b.data_ptr(), M, K, Kp, N, Otp,
            at::cuda::getCurrentCUDAStream(), CuCount);
}

void LLGemmZZ(void* in_a, void* in_b, void* out_c, const int M, const int K,
              cudaStream_t stream, const int solidx);

void LLZZ(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c,
          const int64_t solidx = 0) {
  auto M = in_a.size(0);
  auto K = in_a.size(1);

  LLGemmZZ(in_a.data_ptr(), in_b.data_ptr(), out_c.data_ptr(), M, K,
           at::cuda::getCurrentCUDAStream(), solidx);
}
// instantiate the CPP template for T=float:
// template void AddGPU<float>(at::Tensor in_a, at::Tensor in_b, at::Tensor
// out_c);

void MMGPUKernel(float* in_a, float* in_b, float* out_c, int numARows,
                 int numAColumns, int numBRows, int numBColumns, int numCRows,
                 int numCColumns, cudaStream_t stream);

void MMCustomGPU(at::Tensor& in_a, at::Tensor& in_b, at::Tensor& out_c) {
  auto matA_sizes{in_a.sizes()};
  auto matB_sizes{in_b.sizes()};
  auto matO_sizes{out_c.sizes()};
  MMGPUKernel(in_a.data_ptr<float>(), in_b.data_ptr<float>(),
              out_c.data_ptr<float>(), matA_sizes[0], matA_sizes[1],
              matB_sizes[0], matB_sizes[1], matO_sizes[0], matO_sizes[1],
              at::cuda::getCurrentCUDAStream());
}
