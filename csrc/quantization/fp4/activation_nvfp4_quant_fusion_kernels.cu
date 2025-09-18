/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp8.h>
#include "dispatch_utils.h"

#include "cuda_utils.h"
#include "nvfp4_utils.cuh"

namespace vllm {

// silu in float32
__device__ __forceinline__ float silu(float x) {
  return __fdividef(x, (1.f + __expf(-x)));
}

__device__ __forceinline__ float2 silu2(float2 x) {
  return make_float2(silu(x.x), silu(x.y));
}

template <class Type>
__inline__ __device__ PackedVec<Type> compute_silu_mul(PackedVec<Type>& vec,
                                                       PackedVec<Type>& vec2) {
  PackedVec<Type> result;
  using packed_type = typename TypeConverter<Type>::Type;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    // silu_mul in float32
    if constexpr (std::is_same_v<Type, half>) {
      float2 silu_vec = silu2(__half22float2(vec.elts[i]));
      result.elts[i] =
          __float22half2_rn(__fmul2_rn(silu_vec, __half22float2(vec2.elts[i])));
    } else {
      float2 silu_vec = silu2(__bfloat1622float2(vec.elts[i]));
      result.elts[i] = __float22bfloat162_rn(
          __fmul2_rn(silu_vec, __bfloat1622float2(vec2.elts[i])));
    }
  }
  return result;
}

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(1024, 4)
    silu_mul_cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                             float const* SFScale, uint32_t* out,
                             uint32_t* SFout) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  // Input tensor row/col loops.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD;
         colIdx += blockDim.x) {
      int64_t inOffset =
          rowIdx * (numCols * 2 / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      int64_t inOffset2 = rowIdx * (numCols * 2 / CVT_FP4_ELTS_PER_THREAD) +
                          numCols / CVT_FP4_ELTS_PER_THREAD + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      PackedVec in_vec2 = reinterpret_cast<PackedVec const*>(in)[inOffset2];

      // Get the output tensor offset.
      // Same as inOffset because 8 elements are packed into one uint32_t.
      int64_t outOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      auto& out_pos = out[outOffset];

      // Compute silu and mul
      PackedVec out_silu_mul = compute_silu_mul(in_vec, in_vec2);

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numCols, SFout);

      out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(out_silu_mul, SFScaleVal,
                                                     sf_out);
    }
  }
}

}  // namespace vllm

void silu_and_mul_nvfp4_quant_sm1xxa(torch::Tensor& output,  // [..., d]
                                     torch::Tensor& output_sf,
                                     torch::Tensor& input,  // [..., 2 * d]
                                     torch::Tensor& input_sf) {
  int32_t m = input.size(0);
  int32_t n = input.size(1) / 2;

  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half ||
                  input.scalar_type() == at::ScalarType::BFloat16,
              "Unsupported input data type for quantize_to_fp4.");

  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 1024));
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  VLLM_DISPATCH_HALF_TYPES(
      input.scalar_type(), "silu_and_mul_nvfp4_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
        vllm::silu_mul_cvt_fp16_to_fp4<cuda_type><<<grid, block, 0, stream>>>(
            m, n, input_ptr, input_sf_ptr,
            reinterpret_cast<uint32_t*>(output_ptr),
            reinterpret_cast<uint32_t*>(sf_out));
      });
}
