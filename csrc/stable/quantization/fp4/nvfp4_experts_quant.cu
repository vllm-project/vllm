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

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include "../../torch_utils.h"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cuda_fp8.h>
#include "../../dispatch_utils.h"

#include "cuda_utils.h"
#include "nvfp4_utils.cuh"
#include "launch_bounds_utils.h"

namespace vllm {

// NVFP4 quantization kernel for experts (low-latency path).
// When FUSE_SILU_MUL=true, expects input with gate||up layout and fuses
// SiLU(gate)*up before quantization.
// Use UE4M3 by default.
template <class Type, bool FUSE_SILU_MUL = false, bool UE8M0_SF = false,
          bool SMALL_NUM_EXPERTS = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                    float const* SFScale, uint32_t* out, uint32_t* SFout,
                    uint32_t* input_offset_by_experts,
                    uint32_t* output_scale_offset_by_experts, int n_experts,
                    bool low_latency) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  // When fusing SiLU+Mul, input has gate || up layout (doubled width)
  int inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;

  // Each global thread processes one element
  for (int globalIdx = tid; globalIdx < numRows * colsPerRow;
       globalIdx += gridDim.x * blockDim.x) {
    // Calculate which row and column this global thread should process
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // Find index within the experts using different strategies based on expert
    // count
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    if constexpr (SMALL_NUM_EXPERTS) {
      for (int i = 0; i < n_experts; i++) {
        uint32_t current_offset = __ldca(&input_offset_by_experts[i]);
        uint32_t next_offset = __ldca(&input_offset_by_experts[i + 1]);
        if (rowIdx >= current_offset && rowIdx < next_offset) {
          rowIdx_in_expert = rowIdx - current_offset;
          expert_idx = i;
          break;
        }
      }
    } else {
      // Load input offsets into registers first, then do the computation.
      // Local array size set to 17 because of register limit.
      uint32_t local_offsets[17];
      for (int chunk_start = 0; chunk_start < n_experts; chunk_start += 16) {
        *reinterpret_cast<int4*>(local_offsets) =
            __ldca(reinterpret_cast<const int4*>(
                &input_offset_by_experts[chunk_start]));
        *reinterpret_cast<int4*>(local_offsets + 4) =
            __ldca(reinterpret_cast<const int4*>(
                &input_offset_by_experts[chunk_start + 4]));
        *reinterpret_cast<int4*>(local_offsets + 8) =
            __ldca(reinterpret_cast<const int4*>(
                &input_offset_by_experts[chunk_start + 8]));
        *reinterpret_cast<int4*>(local_offsets + 12) =
            __ldca(reinterpret_cast<const int4*>(
                &input_offset_by_experts[chunk_start + 12]));
        local_offsets[16] = __ldca(&input_offset_by_experts[chunk_start + 16]);

// Check against the 16 loaded offsets
#pragma unroll
        for (int i = 0; i < 16; i++) {
          if (rowIdx >= local_offsets[i] && rowIdx < local_offsets[i + 1]) {
            rowIdx_in_expert = rowIdx - local_offsets[i];
            expert_idx = chunk_start + i;
            break;
          }
        }
      }
    }

    // Load input and optionally apply fused SiLU+Mul
    int64_t inOffset = rowIdx * inColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    PackedVec quant_input;
    if constexpr (FUSE_SILU_MUL) {
      PackedVec in_vec_up =
          reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      quant_input = compute_silu_mul(in_vec, in_vec_up);
    } else {
      quant_input = in_vec;
    }

    // Get the output tensor offset.
    // Same as inOffset because 8 elements are packed into one uint32_t.
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // Get the global scaling factor, which will be applied to the SF.
    // Note SFScale is the same as next GEMM's alpha, which is
    // (448.f / (Alpha_A / 6.f)).
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    uint32_t* SFout_in_expert =
        SFout + output_scale_offset_by_experts[expert_idx] * numKTiles;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);

    out_pos = cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
        quant_input, SFScaleVal, sf_out);
  }
}

// NVFP4 quantization kernel for LARGE_M_TOPK = true (large m_topk optimized
// version). When FUSE_SILU_MUL=true, expects input with gate||up layout and
// fuses SiLU(gate)*up before quantization.
template <class Type, bool FUSE_SILU_MUL = false, bool UE8M0_SF = false,
          bool SMALL_NUM_EXPERTS = false>
__global__ void __launch_bounds__(1024, VLLM_BLOCKS_PER_SM(1024))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                    float const* SFScale, uint32_t* out, uint32_t* SFout,
                    uint32_t* input_offset_by_experts,
                    uint32_t* output_scale_offset_by_experts, int n_experts) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;

  extern __shared__ uint32_t shared_input_offsets[];

  // Load input offsets into shared memory.
  // If n_experts is larger than 4, use vectorized int4 to save instructions.
  // If n_experts is smaller than 4, read directly.
  if constexpr (SMALL_NUM_EXPERTS) {
    for (int i = threadIdx.x; i < n_experts + 1; i += blockDim.x) {
      shared_input_offsets[i] = input_offset_by_experts[i];
    }
  } else {
    for (int i = threadIdx.x * 4; i < n_experts; i += blockDim.x * 4) {
      *reinterpret_cast<int4*>(&shared_input_offsets[i]) =
          *reinterpret_cast<const int4*>(&input_offset_by_experts[i]);
    }
    if (threadIdx.x == 0) {
      shared_input_offsets[n_experts] = input_offset_by_experts[n_experts];
    }
  }

  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  // When fusing SiLU+Mul, input has gate || up layout (doubled width)
  int inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;

  // Each global thread processes one element
  for (int globalIdx = tid; globalIdx < numRows * colsPerRow;
       globalIdx += gridDim.x * blockDim.x) {
    // Calculate which row and column this global thread should process
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // Find expert using binary search for better performance with large m_topk
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    // Binary search through experts using shared memory
    int left = 0, right = n_experts - 1;
    while (left <= right) {
      int mid = (left + right) / 2;
      // Get offsets: shared_input_offsets[i] corresponds to
      // input_offset_by_experts[i]
      uint32_t mid_offset = shared_input_offsets[mid];
      uint32_t next_offset = shared_input_offsets[mid + 1];

      if (rowIdx >= mid_offset && rowIdx < next_offset) {
        rowIdx_in_expert = rowIdx - mid_offset;
        expert_idx = mid;
        break;
      } else if (rowIdx < mid_offset) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }

    // Load input and optionally apply fused SiLU+Mul
    int64_t inOffset = rowIdx * inColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    PackedVec quant_input;
    if constexpr (FUSE_SILU_MUL) {
      PackedVec in_vec_up =
          reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      quant_input = compute_silu_mul(in_vec, in_vec_up);
    } else {
      quant_input = in_vec;
    }

    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    uint32_t* SFout_in_expert =
        SFout + output_scale_offset_by_experts[expert_idx] * numKTiles;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);

    out_pos = cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
        quant_input, SFScaleVal, sf_out);
  }
}

template <typename T, bool FUSE_SILU_MUL = false>
void quant_impl(void* output, void* output_scale, void* input,
                void* input_global_scale, void* input_offset_by_experts,
                void* output_scale_offset_by_experts, int m_topk, int k,
                int n_experts, cudaStream_t stream) {
  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  // Grid, Block size.
  // Each thread converts 8 values.
  int const workSizePerRow = k / ELTS_PER_THREAD;
  int const totalWorkSize = m_topk * workSizePerRow;
  dim3 block(std::min(workSizePerRow, 512));
  // Get number of blocks per SM
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));
  dim3 grid(std::min(static_cast<int>((totalWorkSize + block.x - 1) / block.x),
                     multiProcessorCount * numBlocksPerSM));
  while (grid.x <= multiProcessorCount && block.x > 64) {
    grid.x *= 2;
    block.x = (block.x + 1) / 2;
  }

  int const blockRepeat =
      (totalWorkSize + block.x * grid.x - 1) / (block.x * grid.x);
  if (blockRepeat > 1) {
    size_t shared_mem_size = (n_experts + 1) * sizeof(uint32_t);
    if (n_experts >= 4) {
      cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, false>
          <<<grid, block, shared_mem_size, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<float*>(input_global_scale),
              reinterpret_cast<uint32_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts);
    } else {
      cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, true>
          <<<grid, block, shared_mem_size, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<float*>(input_global_scale),
              reinterpret_cast<uint32_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts);
    }
  } else {
    if (n_experts >= 16) {
      cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, false>
          <<<grid, block, 0, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<float*>(input_global_scale),
              reinterpret_cast<uint32_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts, /* bool low_latency */ true);
    } else {
      cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, true>
          <<<grid, block, 0, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<float*>(input_global_scale),
              reinterpret_cast<uint32_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts, /* bool low_latency */ true);
    }
  }
}

}  // namespace vllm

/*Quantization entry for fp4 experts quantization*/
#define CHECK_TH_CUDA(x, m) \
  STD_TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  STD_TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, m) \
  CHECK_TH_CUDA(x, m);    \
  CHECK_CONTIGUOUS(x, m);

constexpr auto HALF = torch::headeronly::ScalarType::Half;
constexpr auto BF16 = torch::headeronly::ScalarType::BFloat16;
constexpr auto FLOAT = torch::headeronly::ScalarType::Float;
constexpr auto INT = torch::headeronly::ScalarType::Int;
constexpr auto UINT8 = torch::headeronly::ScalarType::Byte;

// Common validation for fp4 experts quantization entry points.
static void validate_fp4_experts_quant_inputs(
    torch::stable::Tensor const& output,
    torch::stable::Tensor const& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts, int64_t m_topk,
    int64_t k) {
  CHECK_INPUT(output, "output");
  CHECK_INPUT(output_scale, "output_scale");
  CHECK_INPUT(input, "input");
  CHECK_INPUT(input_global_scale, "input_global_scale");
  CHECK_INPUT(input_offset_by_experts, "input_offset_by_experts");
  CHECK_INPUT(output_scale_offset_by_experts, "output_scale_offset_by_experts");

  STD_TORCH_CHECK(output.dim() == 2);
  STD_TORCH_CHECK(output_scale.dim() == 2);
  STD_TORCH_CHECK(input.dim() == 2);
  STD_TORCH_CHECK(input_global_scale.dim() == 1);
  STD_TORCH_CHECK(input_offset_by_experts.dim() == 1);
  STD_TORCH_CHECK(output_scale_offset_by_experts.dim() == 1);

  STD_TORCH_CHECK(input.scalar_type() == HALF || input.scalar_type() == BF16);
  STD_TORCH_CHECK(input_global_scale.scalar_type() == FLOAT);
  STD_TORCH_CHECK(input_offset_by_experts.scalar_type() == INT);
  STD_TORCH_CHECK(output_scale_offset_by_experts.scalar_type() == INT);
  // output is uint8 (two nvfp4 values are packed into one uint8)
  // output_scale is int32 (four fp8 values are packed into one int32)
  STD_TORCH_CHECK(output.scalar_type() == UINT8);
  STD_TORCH_CHECK(output_scale.scalar_type() == INT);

  const int BLOCK_SIZE = 16;
  STD_TORCH_CHECK(k % BLOCK_SIZE == 0, "k must be a multiple of 16");
  auto n_experts = input_global_scale.size(0);
  STD_TORCH_CHECK(input_offset_by_experts.size(0) == n_experts + 1);
  STD_TORCH_CHECK(output_scale_offset_by_experts.size(0) == n_experts + 1);
  STD_TORCH_CHECK(output.size(0) == m_topk);
  STD_TORCH_CHECK(output.size(1) == k / 2);
  int scales_k = k / BLOCK_SIZE;
  // 4 means the swizzle requirement by nvidia nvfp4.
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
  // 4 means 4 fp8 values are packed into one int32
  STD_TORCH_CHECK(output_scale.size(1) * 4 == padded_k);
}

void scaled_fp4_experts_quant_sm1xxa(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts) {
  auto m_topk = input.size(0);
  auto k = input.size(1);

  validate_fp4_experts_quant_inputs(output, output_scale, input,
                                    input_global_scale, input_offset_by_experts,
                                    output_scale_offset_by_experts, m_topk, k);

  auto n_experts = input_global_scale.size(0);
  torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "nvfp4_experts_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        vllm::quant_impl<cuda_type, /*FUSE_SILU_MUL=*/false>(
            output.data_ptr(), output_scale.data_ptr(), input.data_ptr(),
            input_global_scale.data_ptr(), input_offset_by_experts.data_ptr(),
            output_scale_offset_by_experts.data_ptr(), m_topk, k, n_experts,
            stream);
      });
}

void silu_and_mul_scaled_fp4_experts_quant_sm1xxa(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts) {
  auto m_topk = input.size(0);
  // Input has gate || up layout, so k = input.size(1) / 2
  auto k_times_2 = input.size(1);
  STD_TORCH_CHECK(k_times_2 % 2 == 0, "input width must be even (gate || up)");
  auto k = k_times_2 / 2;

  validate_fp4_experts_quant_inputs(output, output_scale, input,
                                    input_global_scale, input_offset_by_experts,
                                    output_scale_offset_by_experts, m_topk, k);

  auto n_experts = input_global_scale.size(0);
  torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "silu_mul_nvfp4_experts_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        vllm::quant_impl<cuda_type, /*FUSE_SILU_MUL=*/true>(
            output.data_ptr(), output_scale.data_ptr(), input.data_ptr(),
            input_global_scale.data_ptr(), input_offset_by_experts.data_ptr(),
            output_scale_offset_by_experts.data_ptr(), m_topk, k, n_experts,
            stream);
      });
}
