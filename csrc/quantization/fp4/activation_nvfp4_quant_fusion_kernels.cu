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

#include <torch/csrc/stable/tensor.h>
#include "torch_utils.h"
#include "dispatch_utils.h"
#include "../../cuda_vec_utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cuda_fp8.h>

#include "cuda_utils.h"
#include "launch_bounds_utils.h"

// Define before including nvfp4_utils.cuh so the header
// can use this macro during compilation.
#define NVFP4_ENABLE_ELTS16 1
#include "nvfp4_utils.cuh"

namespace vllm {

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    silu_mul_cvt_fp16_to_fp4(int32_t numRows, int32_t numCols,
                             int32_t num_packed_cols, int32_t num_padded_cols,
                             Type const* __restrict__ in,
                             float const* __restrict__ SFScale,
                             uint32_t* __restrict__ out,
                             uint32_t* __restrict__ SFout) {
  using PackedVec = vllm::PackedVec<Type, CVT_FP4_PACK16>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;

  int const sf_m = round_up<int>(numRows, 128);

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const SFScaleVal = (SFScale == nullptr) ? 1.0f : SFScale[0];

  int32_t const colIdx = blockDim.x * blockIdx.y + threadIdx.x;
  int elem_idx = colIdx * CVT_FP4_ELTS_PER_THREAD;

  // Iterate over all rows and cols including padded ones -
  //  ensures we visit every single scale factor address to initialize it.
  for (int rowIdx = blockIdx.x; rowIdx < sf_m; rowIdx += gridDim.x) {
    if (colIdx < num_padded_cols) {
      PackedVec in_vec;
      PackedVec in_vec2;
      int64_t inOffset =
          rowIdx * (numCols * 2 / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      int64_t inOffset2 = rowIdx * (numCols * 2 / CVT_FP4_ELTS_PER_THREAD) +
                          numCols / CVT_FP4_ELTS_PER_THREAD + colIdx;

      // If we are outside valid rows/columns, feed zeros
      bool valid = (rowIdx < numRows) && (colIdx < num_packed_cols);
      if constexpr (CVT_FP4_PACK16) {
        ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 8],
                         valid);
        ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec2),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset2 * 8],
                         valid);
      } else {
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 4],
                         valid);
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec2),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset2 * 4],
                         valid);
      }

      // Compute silu and mul
      PackedVec out_silu_mul = compute_silu_mul<Type>(in_vec, in_vec2);

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numKTiles, SFout);

      fp4_packed_t out_val{};
      if (valid) {
        out_val =
            cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
                out_silu_mul, SFScaleVal, sf_out);
      } else if (sf_out != nullptr) {
        // The swizzled scale tensor may be allocated with torch.empty.
        // Explicitly initialize padded rows and columns in this kernel so
        // downstream NVFP4 GEMMs never observe stale scale factors.
        *sf_out = 0;
      }

      if (valid) {
        if constexpr (CVT_FP4_PACK16) {
          int64_t outOffset = rowIdx * (numCols / 8) + colIdx * 2;
          uint64_t packed64 =
              (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
          reinterpret_cast<uint64_t*>(out)[outOffset >> 1] = packed64;
        } else {
          int64_t outOffset =
              rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
          reinterpret_cast<fp4_packed_t*>(out)[outOffset] = out_val;
        }
      }
    }
  }
}

}  // namespace vllm

void silu_and_mul_nvfp4_quant_sm1xxa(
    torch::stable::Tensor& output,  // [..., d]
    torch::stable::Tensor& output_sf,
    torch::stable::Tensor& input,  // [..., 2 * d]
    torch::stable::Tensor& input_sf) {
  int32_t m = input.size(0);
  int32_t n = input.size(1) / 2;

  STD_TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Unsupported input data type for quantize_to_fp4.");

  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  auto stream = get_current_cuda_stream(input.get_device_index());
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));

  int num_packed_cols = int(n / CVT_FP4_ELTS_PER_THREAD);

  // The swizzled scale tensor is padded to 128 rows x 4 scale columns; the
  // grid covers the padding so the kernel can initialize it.
  int sf_n_int = int(vllm::round_up(n / CVT_FP4_SF_VEC_SIZE, 4) / 4);
  int num_padded_cols =
      sf_n_int * 4 * CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;

  int grid_y = vllm::div_round_up(num_padded_cols, static_cast<int>(block.x));
  int grid_x =
      std::min(vllm::computeEffectiveRows(m),
               std::max(1, (multiProcessorCount * numBlocksPerSM) / grid_y));
  dim3 grid(grid_x, grid_y);

  // PDL launch pays off at small M; at large M the early-launched blocks
  // slow the chain down, so keep the serialized launch there.
  bool const use_pdl =
      int64_t(m) * num_packed_cols <= int64_t(multiProcessorCount) * 4096;

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "silu_and_mul_nvfp4_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        config.numAttrs = use_pdl ? 1 : 0;
        config.attrs = attrs;
        cudaLaunchKernelEx(&config, vllm::silu_mul_cvt_fp16_to_fp4<cuda_type>,
                           m, n, num_packed_cols, num_padded_cols, input_ptr,
                           input_sf_ptr,
                           reinterpret_cast<uint32_t*>(output_ptr),
                           reinterpret_cast<uint32_t*>(sf_out));
      });
}
