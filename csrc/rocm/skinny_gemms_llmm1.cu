// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-side launcher of the LLMM1 skinny GEMM; kernels in skinny_gemms.cuh.
#include "skinny_gemms.cuh"

torch::stable::Tensor LLMM1(torch::stable::Tensor& in_a,
                            torch::stable::Tensor& in_b,
                            const int64_t rows_per_block) {
  auto M = in_a.size(0);
  auto K = in_a.size(1);
  auto N = in_b.size(0);

  STD_TORCH_CHECK(N == 1, "Row number of activation tensor must be 1.");
  STD_TORCH_CHECK(in_a.scalar_type() == in_b.scalar_type(),
                  "in_a and in_b must have the same dtype, got ",
                  in_a.scalar_type(), " and ", in_b.scalar_type());
  STD_TORCH_CHECK(
      in_b.scalar_type() == torch::headeronly::ScalarType::Half ||
          in_b.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "only float16 and bfloat16 are supported, got ", in_b.scalar_type());

  auto out_c = torch::stable::empty({N, M}, in_b.scalar_type(), std::nullopt,
                                    in_b.device());

  // NUM_TREADS need to be a multiple of WARP_SIZE, as we are using warp shuffle
  // operations.
  const int NUM_THREADS =
      max(rows_per_block * 16,
          K * 2 / 16 % WARP_SIZE == 0
              ? K * 2 / 16
              : K * 2 / 16 + (WARP_SIZE - K * 2 / 16 % WARP_SIZE));

  int NUM_BLOCKS = M / rows_per_block;

  const torch::stable::accelerator::DeviceGuard device_guard(
      in_b.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(in_b.get_device_index());

  // call the kernel function...
  VLLM_STABLE_DISPATCH_HALF_TYPES(in_b.scalar_type(), "LLGemm1", [&] {
    auto a_ptr = in_a.const_data_ptr<scalar_t>();
    auto b_ptr = in_b.const_data_ptr<scalar_t>();
    auto c_ptr = out_c.mutable_data_ptr<scalar_t>();
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
  });

  return out_c;
}
