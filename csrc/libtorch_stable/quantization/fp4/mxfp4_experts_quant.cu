/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * MXFP4 activation quantization kernel for MoE experts.
 * Quantizes BF16/FP16 activations to MXFP4: E2M1 values with E8M0 block scales
 * over 32-element groups.
 *
 * Uses PACK16 E2M1 conversion helpers (nvfp4_utils.cuh) configured for:
 *   - Block size 32 (2 threads per SF in PACK16 mode)
 *   - E8M0 (power-of-two) scale factors
 *   - SF layout: [numMTiles, numKTiles, 32, 4, 4] where numKTiles=ceil(K/128)
 */

// MXFP4 requires PACK16 mode (16 elements per thread) so that
// 2 threads cover 32-element blocks. This requires CUDA >= 12.9.
// Must be defined before any header that (transitively) includes
// nvfp4_utils.cuh.
#define NVFP4_ENABLE_ELTS16 1

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/torch_utils.h"
#include "libtorch_stable/dispatch_utils.h"
#include "cuda_vec_utils.cuh"
#include "cuda_utils.h"

#include "nvfp4_utils.cuh"
static_assert(CVT_FP4_ELTS_PER_THREAD == 16,
              "MXFP4 experts quant requires PACK16 mode (CUDA >= 12.9)");

#include "launch_bounds_utils.h"

namespace vllm {

// MXFP4 block size constants
static constexpr int MXFP4_SF_VEC_SIZE = 32;

// For PACK16 mode (CVT_FP4_ELTS_PER_THREAD=16): 2 threads per SF
// For PACK8 mode (CVT_FP4_ELTS_PER_THREAD=8): 4 threads per SF
static constexpr int MXFP4_NUM_THREADS_PER_SF =
    MXFP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;

// MXFP4 quantization kernel for experts.
// Uses 32-element blocks with E8M0 (UE8M0) scale factors.
// When FUSE_SILU_MUL=true, expects input with gate||up layout and fuses
// SiLU(gate)*up before quantization.
template <class Type, bool FUSE_SILU_MUL = false,
          bool SMALL_NUM_EXPERTS = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    mxfp4_cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                          fp4_packed_t* out, uint32_t* SFout,
                          uint32_t* input_offset_by_experts,
                          uint32_t* output_scale_offset_by_experts,
                          int n_experts, bool low_latency) {
  using PackedVec = PackedVec<Type, CVT_FP4_PACK16>;
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // MXFP4: numKTiles = ceil(numCols / 128) since block_size=32, 4 SFs/tile
  int32_t const numKTiles = (numCols + 127) / 128;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  int inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;

  for (int globalIdx = tid; globalIdx < numRows * colsPerRow;
       globalIdx += gridDim.x * blockDim.x) {
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

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

    // In PACK16 mode, each thread outputs 16 E2M1 values = u32x2
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    uint32_t* SFout_in_expert =
        SFout + output_scale_offset_by_experts[expert_idx] * numKTiles;

    // Use MXFP4_NUM_THREADS_PER_SF (2 for PACK16) for 32-element blocks
    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t, MXFP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);

    // Block E8M0 scales only; no extra tensor-level scale in this path
    constexpr float SFScaleVal = 1.0f;
    // UE8M0_SF=true for MXFP4 E8M0 scale factors
    out_pos =
        cvt_warp_fp16_to_fp4<Type, MXFP4_NUM_THREADS_PER_SF, /*UE8M0_SF=*/true>(
            quant_input, SFScaleVal, sf_out);
  }
}

// Large M_topk variant using shared memory for expert offsets
template <class Type, bool FUSE_SILU_MUL = false,
          bool SMALL_NUM_EXPERTS = false>
__global__ void __launch_bounds__(1024, VLLM_BLOCKS_PER_SM(1024))
    mxfp4_cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                          fp4_packed_t* out, uint32_t* SFout,
                          uint32_t* input_offset_by_experts,
                          uint32_t* output_scale_offset_by_experts,
                          int n_experts) {
  using PackedVec = PackedVec<Type, CVT_FP4_PACK16>;
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // MXFP4: numKTiles = ceil(numCols / 128)
  int32_t const numKTiles = (numCols + 127) / 128;

  extern __shared__ uint32_t shared_input_offsets[];

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
  int inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;

  for (int globalIdx = tid; globalIdx < numRows * colsPerRow;
       globalIdx += gridDim.x * blockDim.x) {
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    // Binary search through experts using shared memory
    int left = 0, right = n_experts - 1;
    while (left <= right) {
      int mid = (left + right) / 2;
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

    // MXFP4 has no global scale - only block-level E8M0 scale factors
    constexpr float SFScaleVal = 1.0f;

    uint32_t* SFout_in_expert =
        SFout + output_scale_offset_by_experts[expert_idx] * numKTiles;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t, MXFP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);

    out_pos =
        cvt_warp_fp16_to_fp4<Type, MXFP4_NUM_THREADS_PER_SF, /*UE8M0_SF=*/true>(
            quant_input, SFScaleVal, sf_out);
  }
}

template <typename T, bool FUSE_SILU_MUL = false>
void mxfp4_quant_impl(void* output, void* output_scale, void* input,
                      void* input_offset_by_experts,
                      void* output_scale_offset_by_experts, int m_topk, int k,
                      int n_experts, cudaStream_t stream) {
  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  int const workSizePerRow = k / ELTS_PER_THREAD;
  int const totalWorkSize = m_topk * workSizePerRow;
  dim3 block(std::min(workSizePerRow, 512));
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
      mxfp4_cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false>
          <<<grid, block, shared_mem_size, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<fp4_packed_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts);
    } else {
      mxfp4_cvt_fp16_to_fp4<T, FUSE_SILU_MUL, true>
          <<<grid, block, shared_mem_size, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<fp4_packed_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts);
    }
  } else {
    if (n_experts >= 16) {
      mxfp4_cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false>
          <<<grid, block, 0, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<fp4_packed_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts, /* bool low_latency */ true);
    } else {
      mxfp4_cvt_fp16_to_fp4<T, FUSE_SILU_MUL, true><<<grid, block, 0, stream>>>(
          m_topk, k, reinterpret_cast<T*>(input),
          reinterpret_cast<fp4_packed_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          n_experts, /* bool low_latency */ true);
    }
  }
}

}  // namespace vllm

/*Quantization entry for mxfp4 experts quantization*/
#define CHECK_TH_CUDA(x, m) \
  STD_TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  STD_TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, m) \
  CHECK_TH_CUDA(x, m);    \
  CHECK_CONTIGUOUS(x, m);

constexpr auto HALF = torch::headeronly::ScalarType::Half;
constexpr auto BF16 = torch::headeronly::ScalarType::BFloat16;
constexpr auto INT = torch::headeronly::ScalarType::Int;
constexpr auto UINT8 = torch::headeronly::ScalarType::Byte;

static constexpr int MXFP4_BLOCK_SIZE = 32;

static void validate_mxfp4_experts_quant_inputs(
    torch::stable::Tensor const& output,
    torch::stable::Tensor const& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts,
    int64_t n_experts, int64_t m_topk, int64_t k) {
  CHECK_INPUT(output, "output");
  CHECK_INPUT(output_scale, "output_scale");
  CHECK_INPUT(input, "input");
  CHECK_INPUT(input_offset_by_experts, "input_offset_by_experts");
  CHECK_INPUT(output_scale_offset_by_experts, "output_scale_offset_by_experts");

  STD_TORCH_CHECK(output.dim() == 2);
  STD_TORCH_CHECK(output_scale.dim() == 2);
  STD_TORCH_CHECK(input.dim() == 2);
  STD_TORCH_CHECK(input_offset_by_experts.dim() == 1);
  STD_TORCH_CHECK(output_scale_offset_by_experts.dim() == 1);

  STD_TORCH_CHECK(input.scalar_type() == HALF || input.scalar_type() == BF16);
  STD_TORCH_CHECK(input_offset_by_experts.scalar_type() == INT);
  STD_TORCH_CHECK(output_scale_offset_by_experts.scalar_type() == INT);
  // output is uint8 (two mxfp4 values packed into one uint8)
  // output_scale is int32 (four E8M0 values packed into one int32)
  STD_TORCH_CHECK(output.scalar_type() == UINT8);
  STD_TORCH_CHECK(output_scale.scalar_type() == INT);

  STD_TORCH_CHECK(k % MXFP4_BLOCK_SIZE == 0, "k must be a multiple of 32");
  STD_TORCH_CHECK(input_offset_by_experts.size(0) == n_experts + 1);
  STD_TORCH_CHECK(output_scale_offset_by_experts.size(0) == n_experts + 1);
  STD_TORCH_CHECK(output.size(0) == m_topk);
  STD_TORCH_CHECK(output.size(1) == k / 2);
  int scales_k = k / MXFP4_BLOCK_SIZE;
  // K-dimension scale columns padded to a multiple of 4 for swizzle layout
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
  // 4 = 4 E8M0 values packed into one int32
  STD_TORCH_CHECK(output_scale.size(1) * 4 == padded_k);
}

void mxfp4_experts_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts,
    int64_t n_experts) {
  auto m_topk = input.size(0);
  auto k = input.size(1);

  validate_mxfp4_experts_quant_inputs(
      output, output_scale, input, input_offset_by_experts,
      output_scale_offset_by_experts, n_experts, m_topk, k);

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "mxfp4_experts_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        vllm::mxfp4_quant_impl<cuda_type, /*FUSE_SILU_MUL=*/false>(
            output.data_ptr(), output_scale.data_ptr(), input.data_ptr(),
            input_offset_by_experts.data_ptr(),
            output_scale_offset_by_experts.data_ptr(), m_topk, k, n_experts,
            stream);
      });
}

void silu_and_mul_mxfp4_experts_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts,
    int64_t n_experts) {
  auto m_topk = input.size(0);
  auto k_times_2 = input.size(1);
  STD_TORCH_CHECK(k_times_2 % 2 == 0, "input width must be even (gate || up)");
  auto k = k_times_2 / 2;

  validate_mxfp4_experts_quant_inputs(
      output, output_scale, input, input_offset_by_experts,
      output_scale_offset_by_experts, n_experts, m_topk, k);

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "silu_mul_mxfp4_experts_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        vllm::mxfp4_quant_impl<cuda_type, /*FUSE_SILU_MUL=*/true>(
            output.data_ptr(), output_scale.data_ptr(), input.data_ptr(),
            input_offset_by_experts.data_ptr(),
            output_scale_offset_by_experts.data_ptr(), m_topk, k, n_experts,
            stream);
      });
}

// Registered here (not torch_bindings.cpp) because VLLM_GPU_FLAGS is applied
// only under COMPILE_LANGUAGE:CUDA, so ENABLE_NVFP4_SM100 is invisible to
// .cpp files and cannot gate the registration from there.
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("mxfp4_experts_quant", TORCH_BOX(&mxfp4_experts_quant));
  m.impl("silu_and_mul_mxfp4_experts_quant",
         TORCH_BOX(&silu_and_mul_mxfp4_experts_quant));
}
