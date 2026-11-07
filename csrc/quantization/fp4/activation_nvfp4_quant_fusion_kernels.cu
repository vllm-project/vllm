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

// The activations fused in front of NVFP4 quantization. A gated activation
// reads the halves x, y of a [m, 2 * n] input; BiasGelu reads a [m, n] input x
// and an optional [n] bias y.
struct SiluAndMul {
  static constexpr bool kGated = true;

  template <class Type>
  static __device__ __forceinline__ PackedVec<Type, CVT_FP4_PACK16> apply(
      const PackedVec<Type, CVT_FP4_PACK16>& x,
      const PackedVec<Type, CVT_FP4_PACK16>& y) {
    return compute_silu_mul<Type>(x, y);
  }
};

struct BiasGelu {
  static constexpr bool kGated = false;

  template <class Type>
  static __device__ __forceinline__ PackedVec<Type, CVT_FP4_PACK16> apply(
      const PackedVec<Type, CVT_FP4_PACK16>& x,
      const PackedVec<Type, CVT_FP4_PACK16>& y) {
    return compute_bias_gelu<Type>(x, y);
  }
};

// Use UE4M3 by default.
template <class Type, class Act, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    act_cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, int32_t outputCols,
                        int32_t num_packed_cols, int32_t num_padded_cols,
                        Type const* __restrict__ in,
                        Type const* __restrict__ bias,
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
  int32_t const numKTiles = (outputCols + 63) / 64;

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
      // The second half of a gated input, else the bias (zeros without one).
      PackedVec in_vec2;
      int64_t const inCols = Act::kGated ? numCols * 2 : numCols;
      int64_t inOffset = rowIdx * (inCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      int64_t inOffset2 =
          Act::kGated ? inOffset + numCols / CVT_FP4_ELTS_PER_THREAD : colIdx;
      Type const* in2 = (Act::kGated || bias == nullptr) ? in : bias;

      // If we are outside valid rows/columns, feed zeros
      bool valid = (rowIdx < numRows) && (colIdx < num_packed_cols);
      bool valid2 = valid && (Act::kGated || bias != nullptr);
      if constexpr (CVT_FP4_PACK16) {
        ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 8],
                         valid);
        ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec2),
                         &reinterpret_cast<const uint32_t*>(in2)[inOffset2 * 8],
                         valid2);
      } else {
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 4],
                         valid);
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec2),
                         &reinterpret_cast<const uint32_t*>(in2)[inOffset2 * 4],
                         valid2);
      }

      PackedVec out_act = Act::template apply<Type>(in_vec, in_vec2);

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numKTiles, SFout);

      fp4_packed_t out_val{};
      if (valid) {
        out_val =
            cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
                out_act, SFScaleVal, sf_out);
      } else if (sf_out != nullptr) {
        // The swizzled scale tensor may be allocated with torch.empty.
        // Explicitly initialize padded rows and columns in this kernel so
        // downstream NVFP4 GEMMs never observe stale scale factors.
        *sf_out = 0;
      }

      // Columns from numCols to outputCols are the K padding of the output
      // (scaled_fp4_quant's padded_n); out_val is zero there.
      if (rowIdx < numRows && elem_idx < outputCols) {
        if constexpr (CVT_FP4_PACK16) {
          int64_t outOffset = rowIdx * (outputCols / 8) + colIdx * 2;
          uint64_t packed64 =
              (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
          reinterpret_cast<uint64_t*>(out)[outOffset >> 1] = packed64;
        } else {
          int64_t outOffset =
              rowIdx * (outputCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
          reinterpret_cast<fp4_packed_t*>(out)[outOffset] = out_val;
        }
      }
    }
  }
}

}  // namespace vllm

template <class Act>
static void act_and_nvfp4_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_sf,
    torch::stable::Tensor const& input, torch::stable::Tensor const& input_sf,
    std::optional<torch::stable::Tensor> const& bias) {
  int32_t m = input.size(0);
  int32_t n = Act::kGated ? input.size(1) / 2 : input.size(1);
  int32_t output_n = output.size(1) * 2;

  STD_TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");
  STD_TORCH_CHECK(output_n % 16 == 0 && output_n >= n,
                  "The output tensor width must be a multiple of 16 and hold "
                  "the input.");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Unsupported input data type for quantize_to_fp4.");
  if (bias.has_value()) {
    STD_TORCH_CHECK(bias->scalar_type() == input.scalar_type() &&
                        bias->dim() == 1 && bias->size(0) == n &&
                        bias->is_contiguous(),
                    "The bias must be a contiguous [N] tensor of the input "
                    "data type.");
  }

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
  int sf_n_int = int(vllm::round_up(output_n / CVT_FP4_SF_VEC_SIZE, 4) / 4);
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
      input.scalar_type(), "act_and_nvfp4_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
        auto bias_ptr = bias.has_value()
                            ? static_cast<cuda_type const*>(bias->data_ptr())
                            : nullptr;
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
        cudaLaunchKernelEx(&config, vllm::act_cvt_fp16_to_fp4<cuda_type, Act>,
                           m, n, output_n, num_packed_cols, num_padded_cols,
                           input_ptr, bias_ptr, input_sf_ptr,
                           reinterpret_cast<uint32_t*>(output_ptr),
                           reinterpret_cast<uint32_t*>(sf_out));
      });
}

void silu_and_mul_nvfp4_quant_sm1xxa(
    torch::stable::Tensor& output,  // [..., d]
    torch::stable::Tensor& output_sf,
    torch::stable::Tensor& input,  // [..., 2 * d]
    torch::stable::Tensor& input_sf) {
  act_and_nvfp4_quant<vllm::SiluAndMul>(output, output_sf, input, input_sf,
                                        std::nullopt);
}

void gelu_nvfp4_quant_sm1xxa(
    torch::stable::Tensor& output,  // [..., d]
    torch::stable::Tensor& output_sf,
    torch::stable::Tensor& input,  // [..., d]
    torch::stable::Tensor& input_sf,
    std::optional<torch::stable::Tensor> const& bias) {  // [d]
  STD_TORCH_CHECK(input.is_contiguous(), "The input must be contiguous.");
  act_and_nvfp4_quant<vllm::BiasGelu>(output, output_sf, input, input_sf, bias);
}
