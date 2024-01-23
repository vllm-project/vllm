#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "dequantize.cuh"
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>


namespace vllm {
namespace awq {

__global__ void __launch_bounds__(64) dequantize_weights(
    int* __restrict__ B, // 4096x64    4096 rows    64 cols
    half* __restrict__ scaling_factors,  // 32x512   32 rows    512 cols
    int* __restrict__ zeros,  // 32x64    32 rows     64 cols
    half* __restrict__ C, // 4096x512    4096 rows    512 cols
    int G
)
{
  int j_factors1 = 4;
  int row_stride2 = 4;
  int split_k_iters = 1;
  static constexpr uint32_t ZERO = 0x0;
  half B_shared[32 * (128 + 8)];

  half* B_shared_ptr2 = B_shared;

  half B_shared_warp[32];
  int OC = 512;

  int N = blockDim.x * gridDim.x;  // 2
  int col = (blockIdx.x * blockDim.x + threadIdx.x);
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int index1 = 8 * col + 8 * row * N;  // + i (<8)
  half* C_ptr2 = C + index1;

  int index2 = col + row * N;
  int* B_ptr2 = B + index2;

  int index3 = col + (int)(row / G) * N;
  int* zeros_ptr2 = zeros + index3;
  int index4 = 8 * col + (int)(row / G) * N * 8;  // + i (<8)
  half* scaling_factors_ptr2 = scaling_factors + index4;


    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr2);
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr2);
int j=0;

      uint32_t B_loaded = *(uint32_t*)(B_ptr2 + j);
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));

      *(uint4*)(B_shared_ptr2 + j) = B_loaded_fp16;

  for (int i=0; i<8; ++i) {
    *(C_ptr2 + i) = B_shared[i];
  }
}

} // namespace awq
} // namespace vllm

// Dequantization to fp16
torch::Tensor awq_dequantize(
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters,
    int thx,
    int thy)
{
    int in_c = _kernel.size(0);
    int qout_c = _kernel.size(1);
    int out_c = qout_c * 8;
    int G = in_c / _scaling_factors.size(0);

    int x_thread = thx;
    int y_thread = thy;

    int x_blocks = 1;
    int y_blocks = 1;
    if (thx==0) {
      x_thread = qout_c;
    }
    if (thy==0) {
      y_thread = in_c;
    }
    if (thx==0 && thy==0) {
      x_thread = 8;
      y_thread = 8;
      x_blocks = (int)(qout_c / 8);
      y_blocks = (int)(in_c / 8);
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(_scaling_factors));

    auto options = torch::TensorOptions().dtype(_scaling_factors.dtype()).device(_scaling_factors.device());
    at::Tensor _de_kernel = torch::empty({in_c, out_c}, options);  // row, col 4096x512

    auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto de_kernel = reinterpret_cast<half*>(_de_kernel.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());

    dim3 num_blocks(x_blocks, y_blocks);
    dim3 threads_per_block(x_thread, y_thread);  //  col, row 64x4096

    dequantize_weights<<<num_blocks, threads_per_block>>>(kernel, scaling_factors, zeros, de_kernel, G);

    return _de_kernel;
}
