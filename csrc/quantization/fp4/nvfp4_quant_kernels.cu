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

// Define before including nvfp4_utils.cuh so the header
// can use this macro during compilation.
#define NVFP4_ENABLE_ELTS16 1
#include "nvfp4_utils.cuh"

namespace vllm {

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, int32_t num_padded_cols,
                    Type const* __restrict__ in,
                    float const* __restrict__ SFScale,
                    uint32_t* __restrict__ out, uint32_t* __restrict__ SFout) {
  using PackedVec = vllm::PackedVec<Type, CVT_FP4_PACK16>;

  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;

  int sf_m = round_up<int>(numRows, 128);
  int32_t const colIdx = blockDim.x * blockIdx.y + threadIdx.x;
  int elem_idx = colIdx * CVT_FP4_ELTS_PER_THREAD;

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const global_scale = (SFScale == nullptr) ? 1.0f : SFScale[0];

  // Iterate over all rows and cols including padded ones -
  //  ensures we visit every single scale factor address to initialize it.
  for (int rowIdx = blockIdx.x; rowIdx < sf_m; rowIdx += gridDim.x) {
    if (colIdx < num_padded_cols) {
      PackedVec in_vec;
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;

      // If we are outside valid rows OR outside valid columns -> Use Zeros
      bool valid = (rowIdx < numRows) && (elem_idx < numCols);
      if constexpr (CVT_FP4_PACK16) {
        ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 8],
                         valid);
      } else {
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 4],
                         valid);
      }

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numKTiles, SFout);

      auto out_val =
          cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
              in_vec, global_scale, sf_out);

      // We do NOT write output for padding because the 'out' tensor is not
      // padded.
      if (valid) {
        if constexpr (CVT_FP4_PACK16) {
          int64_t outOffset = rowIdx * (numCols / 8) + colIdx * 2;
          uint64_t packed64 =
              (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
          reinterpret_cast<uint64_t*>(out)[outOffset >> 1] = packed64;
        } else {
          out[inOffset] = out_val;
        }
      }
    }
  }
}

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4_sf_major(int32_t numRows, int32_t numCols,
                             int32_t sf_n_unpadded, int32_t num_packed_cols,
                             Type const* __restrict__ in,
                             float const* __restrict__ SFScale,
                             uint32_t* __restrict__ out,
                             uint32_t* __restrict__ SFout) {
  using PackedVec = PackedVec<Type, CVT_FP4_PACK16>;

  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  int32_t const colIdx = blockDim.x * blockIdx.y + threadIdx.x;
  int elem_idx = colIdx * CVT_FP4_ELTS_PER_THREAD;

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const global_scale = (SFScale == nullptr) ? 1.0f : SFScale[0];

  // Iterate over all rows and cols including padded ones -
  //  ensures we visit every single scale factor address to initialize it.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    if (colIdx < num_packed_cols) {
      PackedVec in_vec;
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;

      // If we are outside valid rows OR outside valid columns -> Use Zeros
      bool valid = (rowIdx < numRows) && (elem_idx < numCols);
      if constexpr (CVT_FP4_PACK16) {
        ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 8],
                         valid);
      } else {
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 4],
                         valid);
      }

      auto sf_out =
          sf_out_rowmajor_u8<uint32_t>(rowIdx, colIdx, sf_n_unpadded, SFout);

      auto out_val =
          cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
              in_vec, global_scale, sf_out);

      // We do NOT write output for padding because the 'out' tensor is not
      // padded.
      if (valid) {
        if constexpr (CVT_FP4_PACK16) {
          int64_t outOffset = rowIdx * (numCols / 8) + colIdx * 2;
          uint64_t packed64 =
              (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
          reinterpret_cast<uint64_t*>(out)[outOffset >> 1] = packed64;
        } else {
          out[inOffset] = out_val;
        }
      }
    }
  }
}

// ============================================================================
// SM103 (B300) activation quantization kernel.
//
// Identical to the SM100 cvt_fp16_to_fp4 except it writes scale factors
// in the SM103 swizzled layout (Sm103BlockScaledConfig).
// ============================================================================
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4_sm103(int32_t numRows, int32_t numCols,
                          int32_t num_padded_cols,
                          Type const* __restrict__ in,
                          float const* __restrict__ SFScale,
                          uint32_t* __restrict__ out,
                          uint32_t* __restrict__ SFout) {
  using PackedVec = vllm::PackedVec<Type, CVT_FP4_PACK16>;

  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  int32_t const numKTiles = (numCols + 63) / 64;

  int sf_m = round_up<int>(numRows, 128);
  int32_t const colIdx = blockDim.x * blockIdx.y + threadIdx.x;
  int elem_idx = colIdx * CVT_FP4_ELTS_PER_THREAD;

  float const global_scale = (SFScale == nullptr) ? 1.0f : SFScale[0];

  for (int rowIdx = blockIdx.x; rowIdx < sf_m; rowIdx += gridDim.x) {
    if (colIdx < num_padded_cols) {
      PackedVec in_vec;
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;

      bool valid = (rowIdx < numRows) && (elem_idx < numCols);
      if constexpr (CVT_FP4_PACK16) {
        ld256_cg_or_zero(reinterpret_cast<u32x8_t&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 8],
                         valid);
      } else {
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 4],
                         valid);
      }

      // SM103: Use SM103-specific SF offset function
      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset_sm103<uint32_t,
                                                   CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numKTiles, SFout);

      auto out_val =
          cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
              in_vec, global_scale, sf_out);

      if (valid) {
        if constexpr (CVT_FP4_PACK16) {
          int64_t outOffset = rowIdx * (numCols / 8) + colIdx * 2;
          uint64_t packed64 =
              (uint64_t(out_val.hi) << 32) | uint64_t(out_val.lo);
          reinterpret_cast<uint64_t*>(out)[outOffset >> 1] = packed64;
        } else {
          out[inOffset] = out_val;
        }
      }
    }
  }
}

// ============================================================================
// Scale factor layout conversion: SM100 <-> SM103
//
// Converts an already-swizzled SF tensor between SM100 and SM103 layouts.
// Both layouts use the same 512-byte tile structure (128 M-rows x 4 K-cols)
// but arrange bytes differently within each tile.
//
// SM100 offset: outerM(=mIdx%32)*16 + innerM(=(mIdx/32)%4)*4 + innerK
// SM103 offset: m8(=(mIdx/16)%8)*16 + m4a(=(mIdx/4)%4)*128 + m4b(=mIdx%4)*4
//               + innerK
// ============================================================================
__global__ void convert_sf_sm100_to_sm103_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int32_t numMTiles,
    int32_t numKTiles) {
  // Each thread converts one byte (one SF value).
  // Grid: numMTiles * numKTiles blocks, 512 threads per block.
  int32_t tile_idx = blockIdx.x;
  int32_t mTileIdx = tile_idx / numKTiles;
  int32_t kTileIdx = tile_idx % numKTiles;

  // Each tile is 512 bytes: 128 M-positions x 4 K-positions.
  int32_t local_idx = threadIdx.x;  // 0..511
  if (mTileIdx >= numMTiles) return;

  int64_t tile_base = static_cast<int64_t>(tile_idx) << 9;

  // Decode this thread's (mLocal, kLocal) from a simple linear index.
  int32_t mLocal = local_idx >> 2;   // 0..127
  int32_t kLocal = local_idx & 3;    // 0..3

  // Compute SM100 source offset within tile.
  int32_t outerMIdx = mLocal & 31;
  int32_t innerMIdx = (mLocal >> 5) & 3;
  int32_t sm100_off = (outerMIdx << 4) | (innerMIdx << 2) | kLocal;

  // Compute SM103 destination offset within tile.
  int32_t m4b = mLocal & 3;
  int32_t m4a = (mLocal >> 2) & 3;
  int32_t m8  = (mLocal >> 4) & 7;
  int32_t sm103_off = (m8 << 4) | (m4a << 7) | (m4b << 2) | kLocal;

  dst[tile_base + sm103_off] = src[tile_base + sm100_off];
}

__global__ void convert_sf_sm103_to_sm100_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int32_t numMTiles,
    int32_t numKTiles) {
  int32_t tile_idx = blockIdx.x;
  int32_t mTileIdx = tile_idx / numKTiles;
  if (mTileIdx >= numMTiles) return;

  int32_t local_idx = threadIdx.x;
  int64_t tile_base = static_cast<int64_t>(tile_idx) << 9;

  int32_t mLocal = local_idx >> 2;
  int32_t kLocal = local_idx & 3;

  // SM103 source offset
  int32_t m4b = mLocal & 3;
  int32_t m4a = (mLocal >> 2) & 3;
  int32_t m8  = (mLocal >> 4) & 7;
  int32_t sm103_off = (m8 << 4) | (m4a << 7) | (m4b << 2) | kLocal;

  // SM100 destination offset
  int32_t outerMIdx = mLocal & 31;
  int32_t innerMIdx = (mLocal >> 5) & 3;
  int32_t sm100_off = (outerMIdx << 4) | (innerMIdx << 2) | kLocal;

  dst[tile_base + sm100_off] = src[tile_base + sm103_off];
}

}  // namespace vllm

// ============================================================================
// Host entry: SM103 activation quantization
// ============================================================================
void scaled_fp4_quant_sm103a(torch::Tensor const& output,
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

  int sf_n_unpadded = int(n / CVT_FP4_SF_VEC_SIZE);

  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));

  // SM103 always uses swizzled layout (the SM103 variant)
  int sf_n_int = int(vllm::round_up(sf_n_unpadded, 4) / 4);
  int32_t num_padded_cols =
      sf_n_int * 4 * CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;

  int grid_y = vllm::div_round_up(num_padded_cols, static_cast<int>(block.x));
  int grid_x =
      std::min(vllm::computeEffectiveRows(m),
               std::max(1, (multiProcessorCount * numBlocksPerSM) / grid_y));
  dim3 grid(grid_x, grid_y);

  VLLM_DISPATCH_HALF_TYPES(input.scalar_type(), "nvfp4_quant_sm103", [&] {
    using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
    auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
    vllm::cvt_fp16_to_fp4_sm103<cuda_type, false><<<grid, block, 0, stream>>>(
        m, n, num_padded_cols, input_ptr, input_sf_ptr,
        reinterpret_cast<uint32_t*>(output_ptr),
        reinterpret_cast<uint32_t*>(sf_out));
  });
}

// ============================================================================
// Host entry: SF layout conversion SM100 <-> SM103
// ============================================================================
void convert_sf_layout_sm100_to_sm103(torch::Tensor& dst,
                                      torch::Tensor const& src) {
  TORCH_CHECK(src.is_contiguous(), "Source SF tensor must be contiguous");
  TORCH_CHECK(dst.is_contiguous(), "Destination SF tensor must be contiguous");
  TORCH_CHECK(src.numel() == dst.numel(),
              "Source and destination must have the same number of elements");

  // SF tensors are stored as int32 with shape (rounded_m, rounded_k / 4)
  // Total bytes = rounded_m * (rounded_k / 4) * 4 = rounded_m * rounded_k
  int64_t total_bytes = src.numel() * src.element_size();
  int32_t numMTiles = src.size(0) / 128;
  int32_t numKTiles = total_bytes / (numMTiles * 512);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(src));
  auto stream = at::cuda::getCurrentCUDAStream(src.get_device());

  int32_t num_tiles = numMTiles * numKTiles;
  dim3 grid(num_tiles);
  dim3 block(512);

  vllm::convert_sf_sm100_to_sm103_kernel<<<grid, block, 0, stream>>>(
      static_cast<const uint8_t*>(src.data_ptr()),
      static_cast<uint8_t*>(dst.data_ptr()),
      numMTiles, numKTiles);
}

void convert_sf_layout_sm103_to_sm100(torch::Tensor& dst,
                                      torch::Tensor const& src) {
  TORCH_CHECK(src.is_contiguous() && dst.is_contiguous());
  TORCH_CHECK(src.numel() == dst.numel());

  int64_t total_bytes = src.numel() * src.element_size();
  int32_t numMTiles = src.size(0) / 128;
  int32_t numKTiles = total_bytes / (numMTiles * 512);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(src));
  auto stream = at::cuda::getCurrentCUDAStream(src.get_device());

  int32_t num_tiles = numMTiles * numKTiles;
  vllm::convert_sf_sm103_to_sm100_kernel<<<dim3(num_tiles), dim3(512), 0, stream>>>(
      static_cast<const uint8_t*>(src.data_ptr()),
      static_cast<uint8_t*>(dst.data_ptr()),
      numMTiles, numKTiles);
}

// ============================================================================
// Original SM100 host entry
// ============================================================================
void scaled_fp4_quant_sm1xxa(torch::Tensor const& output,
                             torch::Tensor const& input,
                             torch::Tensor const& output_sf,
                             torch::Tensor const& input_sf,
                             bool is_sf_swizzled_layout) {
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

  int sf_n_unpadded = int(n / CVT_FP4_SF_VEC_SIZE);

  // Grid, Block size. Each thread converts 8 values.
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));

  if (is_sf_swizzled_layout) {
    int sf_n_int = int(vllm::round_up(sf_n_unpadded, 4) / 4);
    int32_t num_padded_cols =
        sf_n_int * 4 * CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;

    int grid_y = vllm::div_round_up(num_padded_cols, static_cast<int>(block.x));
    int grid_x =
        std::min(vllm::computeEffectiveRows(m),
                 std::max(1, (multiProcessorCount * numBlocksPerSM) / grid_y));
    dim3 grid(grid_x, grid_y);

    VLLM_DISPATCH_HALF_TYPES(input.scalar_type(), "nvfp4_quant_kernel", [&] {
      using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
      auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
      // NOTE: We don't support e8m0 scales at this moment.
      vllm::cvt_fp16_to_fp4<cuda_type, false><<<grid, block, 0, stream>>>(
          m, n, num_padded_cols, input_ptr, input_sf_ptr,
          reinterpret_cast<uint32_t*>(output_ptr),
          reinterpret_cast<uint32_t*>(sf_out));
    });
  } else {
    int num_packed_cols = n / CVT_FP4_ELTS_PER_THREAD;
    int grid_y = vllm::div_round_up(num_packed_cols, static_cast<int>(block.x));
    int grid_x = std::min(
        m, std::max(1, (multiProcessorCount * numBlocksPerSM) / grid_y));
    dim3 grid(grid_x, grid_y);

    VLLM_DISPATCH_HALF_TYPES(input.scalar_type(), "nvfp4_quant_kernel", [&] {
      using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
      auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
      // NOTE: We don't support e8m0 scales at this moment.
      vllm::cvt_fp16_to_fp4_sf_major<cuda_type, false>
          <<<grid, block, 0, stream>>>(m, n, sf_n_unpadded, num_packed_cols,
                                       input_ptr, input_sf_ptr,
                                       reinterpret_cast<uint32_t*>(output_ptr),
                                       reinterpret_cast<uint32_t*>(sf_out));
    });
  }
}
