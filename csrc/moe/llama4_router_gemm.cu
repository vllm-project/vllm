/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc7/cpp/tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Bf16Bf16Gemm.cu
 * Copyright (c) 2026, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

template <int VEC_SIZE>
struct __align__(8) aligned_bf16x4 {
  __align__(8) __nv_bfloat16 data[VEC_SIZE];
};

__device__ __forceinline__ float2 ffma2(float2 x, float2 y, float2 acc) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000))
  return __ffma2_rn(x, y, acc);
#else
  return make_float2(x.x * y.x + acc.x, x.y * y.y + acc.y);
#endif
}

template <int VEC_SIZE, int BLOCK_SIZE, int GEMM_K, int NUM_EXPERTS>
__global__ void llama4_router_gemm_kernel(
    int num_tokens,
    __nv_bfloat16 const* __restrict__ A,  // Input vector
                                          // [num_tokens][hidden_size]
    __nv_bfloat16 const* __restrict__ B,  // Input matrix
                                          // [num_experts][hidden_size]
    __nv_bfloat16* __restrict__ C  // Output vector [num_tokens][num_experts]
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
  // Shared memory for block reduction
  __shared__ float reduce_buffer[BLOCK_SIZE];

  // Each thread accumulates its partial sum
  float2 thread_sum;
  thread_sum.x = 0.0f;
  thread_sum.y = 0.0f;

  // Each thread processes 4 elements at a time, 5 times
  int const token_idx = blockIdx.x / NUM_EXPERTS;
  int const row =
      blockIdx.x % NUM_EXPERTS;  // Matrix row / Output element index
  int const tid = threadIdx.x;   // Thread ID within the block

  // PDL prefetch all B data
  aligned_bf16x4<VEC_SIZE> b_vec[GEMM_K / BLOCK_SIZE / VEC_SIZE];
  #pragma unroll
  for (int chunk = 0; chunk < GEMM_K / BLOCK_SIZE / VEC_SIZE; chunk++) {
    // Base index for this chunk
    int base_idx = chunk * BLOCK_SIZE + tid;

    // Load 4 elements at once
    b_vec[chunk] = reinterpret_cast<aligned_bf16x4<VEC_SIZE> const*>(
        B)[row * GEMM_K / VEC_SIZE + base_idx];
  }

  cudaGridDependencySynchronize();

  // Process 5 chunks of 4 elements each
  #pragma unroll
  for (int chunk = 0; chunk < GEMM_K / BLOCK_SIZE / VEC_SIZE; chunk++) {
    // Base index for this chunk
    int base_idx = chunk * BLOCK_SIZE + tid;

    // Load 4 elements at once
    aligned_bf16x4<VEC_SIZE> a_vec =
        reinterpret_cast<aligned_bf16x4<VEC_SIZE> const*>(
            A)[token_idx * GEMM_K / VEC_SIZE + base_idx];
  #pragma unroll
    for (int i = 0; i < VEC_SIZE; i += 2) {
      float2 a_val = make_float2(a_vec.data[i], a_vec.data[i + 1]);
      float2 b_val =
          make_float2(b_vec[chunk].data[i], b_vec[chunk].data[i + 1]);

      thread_sum = ffma2(a_val, b_val, thread_sum);
    }
  }

  // Warp-level reduction
  float warp_sum = thread_sum.x + thread_sum.y;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
  }

  // First thread in each warp writes to shared memory
  if (tid % warpSize == 0) {
    reduce_buffer[tid / warpSize] = warp_sum;
  }
  __syncthreads();

  // Final thread reduces across warps and writes the result
  if (tid == 0) {
    float block_sum = 0.0f;
    for (int i = 0; i < BLOCK_SIZE / warpSize; i++) {
      block_sum += reduce_buffer[i];
    }
    C[token_idx * NUM_EXPERTS + row] = __float2bfloat16(block_sum);
  }
#endif
}

void llama4_router_gemm_launcher(int num_tokens, __nv_bfloat16 const* A,
                                 __nv_bfloat16 const* B, __nv_bfloat16* C,
                                 cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  constexpr int NUM_EXPERTS = 128;
  constexpr int GEMM_K = 5120;
  constexpr int VEC_SIZE = 4;

  int const grid_size = NUM_EXPERTS * num_tokens;

  cudaLaunchConfig_t config;
  config.gridDim = dim3(grid_size);
  config.blockDim = dim3(BLOCK_SIZE);
  config.dynamicSmemBytes = 0;
  config.stream = stream;

  cudaLaunchAttribute attrs[1];
  config.attrs = attrs;
  config.numAttrs = 0;
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchKernelEx(
      &config,
      &llama4_router_gemm_kernel<VEC_SIZE, BLOCK_SIZE, GEMM_K, NUM_EXPERTS>,
      num_tokens, A, B, C);
}

void llama4_router_gemm_op(int num_tokens, void const* A, void const* B,
                           void* C, cudaStream_t stream) {
  __nv_bfloat16 const* A_bf16 = static_cast<__nv_bfloat16 const*>(A);
  __nv_bfloat16 const* B_bf16 = static_cast<__nv_bfloat16 const*>(B);
  __nv_bfloat16* C_bf16 = static_cast<__nv_bfloat16*>(C);

  llama4_router_gemm_launcher<16>(num_tokens, A_bf16, B_bf16, C_bf16, stream);
}

void llama4_router_gemm(torch::Tensor& output, torch::Tensor const& inputA,
                        torch::Tensor const& inputB) {
  TORCH_CHECK(inputA.scalar_type() == at::ScalarType::BFloat16,
              "inputA tensor must be bfloat16");
  TORCH_CHECK(inputB.scalar_type() == at::ScalarType::BFloat16,
              "inputB tensor must be bfloat16");

  TORCH_CHECK(inputA.dim() == 2, "inputA must be 2D.");
  TORCH_CHECK(inputB.dim() == 2, "inputB must be 2D.");
  TORCH_CHECK(inputA.sizes()[1] == 5120, "inputA.size(1) must be 5120");
  TORCH_CHECK(inputB.sizes()[0] == 128, "inputB.size(0) must be 128");
  TORCH_CHECK(inputB.sizes()[1] == 5120, "inputB.size(1) must be 5120");

  auto const num_tokens = inputA.sizes()[0];
  auto const num_experts = inputB.sizes()[0];

  auto stream = at::cuda::getCurrentCUDAStream(inputA.get_device());

  llama4_router_gemm_op(num_tokens, num_experts, inputA.data_ptr(),
                        inputB.data_ptr(), output.mutable_data_ptr(), stream);
}
