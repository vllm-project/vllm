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
#include "launch_bounds_utils.h"
#include "nvfp4_utils.cuh"

namespace vllm {

template <typename Int>
__host__ __device__ inline Int round_up(Int x, Int y) {
  static_assert(std::is_integral_v<Int>,
                "round_up argument must be integral type");
  return (x + y - 1) / y * y;
}


// // Use UE4M3 by default.
// template <class Type, bool UE8M0_SF = false>
// __global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
//     cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
//                     float const* SFScale, uint32_t* out, uint32_t* SFout) {
//   using PackedVec = PackedVec<Type>;
//   static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
//       (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
//   static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
//                 "Vec size is not matched.");

//   int sf_m = round_up<int>(numRows, 128);
//   int sf_n_unpadded = numCols / CVT_FP4_SF_VEC_SIZE;
//   int sf_n_int = round_up<int>(sf_n_unpadded, 4) / 4;
//   for (int row = numRows + blockIdx.x; row < sf_m; row += gridDim.x) {
//     // Each thread writes 4 uint32_t elements.
//     for (int col = sf_n_unpadded + threadIdx.x * 4; col < sf_n_int;
//          col += blockDim.x * 4) {
//       SFout[row * sf_n_int + col] = 0x00;
//     }
//   }

//   // Get the global scaling factor, which will be applied to the SF.
//   // Note SFScale is the same as next GEMM's alpha, which is
//   // (448.f / (Alpha_A / 6.f)).
//   float const global_scale = SFScale == nullptr ? 1.0f : SFScale[0];

//   // Input tensor row/col loops.
//   for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
//     for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD;
//          colIdx += blockDim.x) {
//       int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
//       PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
//       // Get the output tensor offset.
//       // Same as inOffset because 8 elements are packed into one uint32_t.
//       int64_t outOffset = inOffset;
//       auto& out_pos = out[outOffset];

//       auto sf_out =
//           cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
//                                              CVT_FP4_NUM_THREADS_PER_SF>(
//               rowIdx, colIdx, numCols, SFout);

//       out_pos =
//           cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, global_scale, sf_out);
//     }
//   }
// }





// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                    float const* SFScale, uint32_t* out, uint32_t* SFout) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  int sf_m = round_up<int>(numRows, 128);
  int sf_n_unpadded = numCols / CVT_FP4_SF_VEC_SIZE;
  int sf_n_int = round_up<int>(sf_n_unpadded, 4) / 4;

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const global_scale = SFScale == nullptr ? 1.0f : SFScale[0];

  // Calculate padded columns to ensure we cover the full tail block of SFs.
  // Each SF covers 16 elements. A block covers 4 SFs = 64 elements.
  int num_padded_cols = sf_n_int * 4 * CVT_FP4_SF_VEC_SIZE; 
  int numColThreads = numCols / CVT_FP4_ELTS_PER_THREAD;
  int numPaddedColThreads = num_padded_cols / CVT_FP4_ELTS_PER_THREAD;

  // Main loop: Process ALL rows and columns (including padding) like TensorRT-LLM
  // This ensures we zero scale factors at their correct swizzled addresses
  for (int rowIdx = blockIdx.x; rowIdx < sf_m; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numPaddedColThreads; colIdx += blockDim.x) {
      // Get the swizzled SF pointer - works for BOTH valid and padding regions
      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, num_padded_cols, SFout);

      // Check if we're in padding region (either row or column)
      bool is_padding_region = (rowIdx >= numRows) || (colIdx >= numColThreads);
      
      // DEBUG LOG
      if (blockIdx.x == 0 && threadIdx.x == 0) {
          // Log only the first few iterations or boundary conditions to avoid spam
          // Log boundary of valid/padding
          if (colIdx == numColThreads || colIdx == numColThreads - 1 || 
              (rowIdx == numRows || rowIdx == numRows - 1)) {
              printf("[FP4_DEBUG] row=%d, col=%d, is_padding=%d, sf_out_ptr=%p\n", 
                     rowIdx, colIdx, is_padding_region, sf_out);
          }
      }

      if (is_padding_region) {
          // Zero the scale factor for padding (like TRT-LLM lines 830-836)
          // This is CRITICAL - must use swizzled address from sf_out, not linear indexing!
          if (sf_out != nullptr) {
              *sf_out = 0x00;
              // DEBUG LOG
              if (blockIdx.x == 0 && threadIdx.x == 0 && rowIdx < numRows + 2) {
                  printf("[FP4_DEBUG] ZEROING Padding: row=%d, col=%d at %p\n", rowIdx, colIdx, sf_out);
              }
          }
          // Note: We do NOT write to the FP4 output buffer for padding regions
          // because the output buffer is allocated with original dimensions (m, n//2)
      } else {
          // Valid region - load input and quantize
          int64_t inOffset = rowIdx * numColThreads + colIdx;
          PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
          
          // Perform conversion (writes scale to sf_out) and get FP4 output
          auto out_val = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, global_scale, sf_out);
          
          // Write FP4 output using UNPADDED stride (buffer has original dimensions)
          out[inOffset] = out_val;
          
          // DEBUG LOG
          if (blockIdx.x == 0 && threadIdx.x == 0 && rowIdx == 0 && colIdx > numColThreads - 5) {
               printf("[FP4_DEBUG] Writing VALID: row=%d, col=%d, inOffset=%ld\n", rowIdx, colIdx, inOffset);
          }
      }
    }
  }
}

template <typename T>
void invokeFP4Quantization(int m, int n, T const* input, float const* SFScale,
                           int64_t* output, int32_t* SFOuput, bool useUE8M0,
                           int multiProcessorCount, cudaStream_t stream) {
  // Grid, Block size.
  // Each thread converts 8 values.
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
  // Get number of blocks per SM
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  // Launch the cvt kernel.
  if (useUE8M0) {
    cvt_fp16_to_fp4<T, true><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(SFOuput));
  } else {
    cvt_fp16_to_fp4<T, false><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(SFOuput));
  }
}

// Instantiate the function.
template void invokeFP4Quantization(int m, int n, half const* input,
                                    float const* SFScale, int64_t* output,
                                    int32_t* SFOuput, bool useUE8M0,
                                    int multiProcessorCount,
                                    cudaStream_t stream);

template void invokeFP4Quantization(int m, int n, __nv_bfloat16 const* input,
                                    float const* SFScale, int64_t* output,
                                    int32_t* SFOuput, bool useUE8M0,
                                    int multiProcessorCount,
                                    cudaStream_t stream);

}  // namespace vllm

void scaled_fp4_quant_sm1xxa(torch::Tensor const& output,
                             torch::Tensor const& input,
                             torch::Tensor const& output_sf,
                             torch::Tensor const& input_sf) {
  int32_t m = input.size(0);
  int32_t n = input.size(1);

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

  // We don't support e8m0 scales at this moment.
  bool useUE8M0 = false;

  VLLM_DISPATCH_HALF_TYPES(input.scalar_type(), "nvfp4_quant_kernel", [&] {
    using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
    auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
    vllm::invokeFP4Quantization(m, n, input_ptr, input_sf_ptr, output_ptr,
                                sf_out, useUE8M0, multiProcessorCount, stream);
  });
}
