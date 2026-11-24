// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-side launcher of the LLMM1 skinny GEMM; kernels in skinny_gemms.cuh.
#include "skinny_gemms.cuh"

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block) {
  auto M = in_a.size(0);
  auto K = in_a.size(1);
  auto N = in_b.size(0);

  TORCH_CHECK(N == 1, "Row number of activation tensor must be 1.");
  TORCH_CHECK(in_a.dtype() == in_b.dtype());
  TORCH_CHECK(in_b.dtype() == torch::kFloat16 ||
              in_b.dtype() == torch::kBFloat16);

  auto out_c = torch::empty(
      {N, M}, torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  // NUM_TREADS need to be a multiple of WARP_SIZE, as we are using warp shuffle
  // operations.
  const int NUM_THREADS =
      max(rows_per_block * 16,
          K * 2 / 16 % WARP_SIZE == 0
              ? K * 2 / 16
              : K * 2 / 16 + (WARP_SIZE - K * 2 / 16 % WARP_SIZE));

  int NUM_BLOCKS = M / rows_per_block;

  // Both launch dimensions are derived from runtime shapes, so both can be
  // invalid for inputs this function otherwise accepts: NUM_THREADS grows as
  // K/8 and passes the 1024-thread block limit once K > 8192, and NUM_BLOCKS
  // is zero when M < rows_per_block. Check them here so the caller gets a
  // clear error instead of the launch failing below.
  TORCH_CHECK(NUM_THREADS <= 1024, "LLMM1: K=", K, " needs ", NUM_THREADS,
              " threads per block, above the 1024 limit.");
  TORCH_CHECK(NUM_BLOCKS > 0, "LLMM1: M=", M,
              " is smaller than rows_per_block=", rows_per_block,
              ", giving an empty grid.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_b));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // call the kernel function...
  AT_DISPATCH_REDUCED_FLOATING_TYPES(in_b.scalar_type(), "LLGemm1", [&] {
    auto a_ptr = in_a.data_ptr<scalar_t>();
    auto b_ptr = in_b.data_ptr<scalar_t>();
    auto c_ptr = out_c.data_ptr<scalar_t>();
    if (rows_per_block == 2) {
      LLGemm1_kernel<scalar_t, 2>
          <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    } else if (rows_per_block == 4) {
      LLGemm1_kernel<scalar_t, 4>
          <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    } else if (rows_per_block == 8) {
      LLGemm1_kernel<scalar_t, 8>
          <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    } else if (rows_per_block == 16) {
      LLGemm1_kernel<scalar_t, 16>
          <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    } else {
      NUM_BLOCKS = M / 4;
      LLGemm1_kernel<scalar_t, 4>
          <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
    }
    // out_c comes from torch::empty(), so a launch that silently fails would
    // return uninitialised memory to the caller rather than raising.
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return out_c;
}
