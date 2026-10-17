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

#define NVFP4_ENABLE_ELTS16

#include <torch/csrc/stable/tensor.h>
#include "torch_utils.h"
#include "dispatch_utils.h"
#include "cuda_vec_utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cuda_fp8.h>

#include "cuda_utils.h"
#include "nvfp4_utils.cuh"
#include "launch_bounds_utils.h"
#include "core/cuda_compat.h"

#define CVT_FP4_MAX_THREADS_PER_BLOCK (512)
#define CVT_FP4_EXPERT_SMEM_MIN_EXPERTS (32)

namespace vllm {

template <class PackedVecT>
__device__ __forceinline__ PackedVecT LoadPackedVec(PackedVecT const* ptr) {
  if constexpr (VLLM_256B_PTX_ENABLED &&
                sizeof(PackedVecT) == sizeof(u32x8_t)) {
    PackedVecT vec;
    ld256(vec, ptr);
    return vec;
  } else {
    return *ptr;
  }
}

// Entries between the per-expert shared tables: n_experts + 1 rounded up to 4
// so every table starts 16B aligned for the int4/float4 stores.
__host__ __device__ __forceinline__ int expert_table_stride(int n_experts) {
  return (n_experts + 1 + 3) & ~3;
}

// Shared bytes for the per-expert input offsets, SF offsets and global scales.
inline size_t expert_tables_smem_size(int n_experts) {
  return (2 * expert_table_stride(n_experts) + n_experts) * sizeof(uint32_t);
}

// Copy the per-expert input offsets and SF offsets (n_experts + 1 entries
// each) and global scales (n_experts entries) into shared memory. The global
// tables must be 16B aligned: whole 4-expert chunks are loaded vectorized and
// the remainder, including the [n_experts] sentinel, scalar, so any n_experts
// is supported. The scalar entries go to the block's last threads and are
// loaded before the vector chunks, which start from thread 0, so every load is
// in flight before any store waits on one: up to 4 * blockDim experts cost one
// global round trip rather than one per kind of entry.
__device__ __forceinline__ void load_expert_tables(
    uint32_t* __restrict__ shared_input_offsets,
    uint32_t* __restrict__ shared_output_scale_offsets,
    float* __restrict__ shared_SFScale,
    uint32_t const* __restrict__ input_offset_by_experts,
    uint32_t const* __restrict__ output_scale_offset_by_experts,
    float const* __restrict__ SFScale, int n_experts) {
  float4 const ones = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  int const tail_begin = n_experts & ~3;

  // At most 4 scalar entries: up to 3 tail experts plus the sentinel.
  int const scalar_i = tail_begin + (blockDim.x - 1 - threadIdx.x);
  uint32_t input_offset, output_scale_offset;
  float scale;
  if (scalar_i <= n_experts) {
    input_offset = input_offset_by_experts[scalar_i];
    output_scale_offset = output_scale_offset_by_experts[scalar_i];
  }
  if (scalar_i < n_experts) {
    scale = SFScale == nullptr ? 1.0f : SFScale[scalar_i];
  }

  for (int i = threadIdx.x * 4; i < tail_begin; i += blockDim.x * 4) {
    *reinterpret_cast<int4*>(&shared_input_offsets[i]) =
        *reinterpret_cast<int4 const*>(&input_offset_by_experts[i]);
    *reinterpret_cast<int4*>(&shared_output_scale_offsets[i]) =
        *reinterpret_cast<int4 const*>(&output_scale_offset_by_experts[i]);
    *reinterpret_cast<float4*>(&shared_SFScale[i]) =
        SFScale == nullptr ? ones
                           : *reinterpret_cast<float4 const*>(&SFScale[i]);
  }

  if (scalar_i <= n_experts) {
    shared_input_offsets[scalar_i] = input_offset;
    shared_output_scale_offsets[scalar_i] = output_scale_offset;
  }
  if (scalar_i < n_experts) {
    shared_SFScale[scalar_i] = scale;
  }
}

// Find the last expert starting at or before row, walking forward from
// expert_idx; the caller's rows only advance and
// row < input_offsets[n_experts]. The expert is usually unchanged, which one
// read confirms. Otherwise, with gallop, experts 1, 2, 4, ... further ahead are
// probed while that beats bisecting the rest of the table, and the range left,
// below the first probe past row or else the rest of the table, is bisected.
// Empty experts share their offset with the next one and are stepped over.
//
// Where galloping pays, for a jump of d experts (the row's expert minus
// expert_idx) out of n = n_experts:
//   bisecting the rest of the table takes ~log2(n) dependent reads;
//   galloping takes ~log2(d) probes to pass row plus ~log2(d) to narrow the
//   last step, ~2 * log2(d) + 1 = log2(2 * d^2) reads.
// So galloping wins while 2 * d^2 < n, i.e. d < sqrt(n / 2), which caps its
// step; past the cap the probes are wasted and the rest is bisected.
__device__ __forceinline__ int search_expert(uint32_t const* input_offsets,
                                             int n_experts, int expert_idx,
                                             uint32_t row, bool gallop) {
  if (input_offsets[expert_idx + 1] > row) {
    return expert_idx;
  }
  // The expert lies in [lo, hi] and input_offsets[lo] <= row.
  int lo = expert_idx + 1, hi = n_experts - 1;
  if (gallop) {
    // The cap: the largest power-of-two step with 2 * step^2 <= n_experts; it
    // depends only on n_experts, so it is computed once rather than tested per
    // probe.
    int const max_step = 1 << ((31 - __clz(max(n_experts >> 1, 1))) >> 1);
    for (int step = 1; step <= max_step && lo + step <= hi; step <<= 1) {
      if (input_offsets[lo + step] > row) {
        hi = lo + step - 1;
        break;
      }
      lo += step;
    }
  }
  while (lo < hi) {
    int const mid = (lo + hi + 1) >> 1;
    bool const le = input_offsets[mid] <= row;
    lo = le ? mid : lo;
    hi = le ? hi : mid - 1;
  }
  return lo;
}

// Block and grid sizes for one wave of blocks grid-striding over work_items,
// one per thread. The block shrinks from the largest size while its blocks
// cannot fill every resident block slot, so small inputs still spread over
// all SMs, but stops at the minimum block size or before exceeding one wave
// of full-size blocks. Only multiples of the warp size dividing the SM's
// thread capacity are taken, so the resident blocks fill it exactly.
template <class Kernel>
std::pair<int, int> one_wave_launch_dims(Kernel kernel, int64_t work_items,
                                         size_t shared_mem_size) {
  static constexpr int kWarpSize = WARP_SIZE;
  // Each block stages the expert tables once, so a block keeps at least this
  // many threads to share that cost.
  static constexpr int kMinThreadsPerBlock = 128;
  cudaDeviceProp const* const prop = get_device_prop();
  int const multiProcessorCount = prop->multiProcessorCount;
  int const maxThreadsPerSM = prop->maxThreadsPerMultiProcessor;

  // Resident block slots on the device for a block of the given size.
  int numBlocksPerSM = 0;
  int block_size = CVT_FP4_MAX_THREADS_PER_BLOCK;
  STD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSM, kernel, block_size, shared_mem_size));
  // We need to iteratively compute the number of blocks per SM because it
  // depends on the block size, which we adjust in the loop below.
  // The initial computation gives us the number of block slots for the maximum
  // block size, and we will adjust the block size downwards to ensure small
  // input size still utilize all the SMs.
  int64_t block_slots =
      (int64_t)multiProcessorCount * std::max(numBlocksPerSM, 1);
  int64_t const max_blocks = block_slots;

  while (block_size > kMinThreadsPerBlock &&
         div_round_up(work_items, (int64_t)block_size) < block_slots) {
    int next_block_size = block_size - kWarpSize;
    while (next_block_size > kMinThreadsPerBlock &&
           maxThreadsPerSM % next_block_size != 0) {
      next_block_size -= kWarpSize;
    }
    if (next_block_size < kMinThreadsPerBlock ||
        maxThreadsPerSM % next_block_size != 0 ||
        div_round_up(work_items, (int64_t)next_block_size) > max_blocks) {
      break;
    }
    block_size = next_block_size;
    STD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM, kernel, block_size, shared_mem_size));
    block_slots = (int64_t)multiProcessorCount * std::max(numBlocksPerSM, 1);
  }
  int const grid_size = static_cast<int>(std::min<int64_t>(
      div_round_up(work_items, (int64_t)block_size), block_slots));
  return {block_size, grid_size};
}

// NVFP4 quantization kernel for LARGE_M_TOPK = true (large m_topk optimized
// version). When FUSE_SILU_MUL=true, expects input with gate||up layout and
// fuses SiLU(gate)*up before quantization.
template <class Type, bool FUSE_SILU_MUL = false, bool UE8M0_SF = false,
          bool STAGE_EXPERT_TABLES = true>
__global__ void __launch_bounds__(
    CVT_FP4_MAX_THREADS_PER_BLOCK,
    VLLM_BLOCKS_PER_SM(CVT_FP4_MAX_THREADS_PER_BLOCK))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols,
                    Type const* __restrict__ in,
                    float const* __restrict__ SFScale,
                    uint32_t* __restrict__ out, uint32_t* __restrict__ SFout,
                    uint32_t* __restrict__ input_offset_by_experts,
                    uint32_t* __restrict__ output_scale_offset_by_experts,
                    int n_experts, int32_t colsPerRow, int32_t numKTiles,
                    uint32_t rowStride, int gridStride) {
  using PackedVec = PackedVec<Type, CVT_FP4_PACK16>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  // When fusing SiLU+Mul, input has gate || up layout (doubled width)
  int const inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;

  PackedVec const* const in_vecs = reinterpret_cast<PackedVec const*>(in);

  // Per-expert tables, read from global memory unless staged in shared memory
  // (see CVT_FP4_EXPERT_SMEM_MIN_EXPERTS). STAGE_EXPERT_TABLES is a template
  // parameter so each instantiation's table pointers have one known address
  // space: staged lookups compile to LDS and unstaged ones to LDG.
  uint32_t const* input_offsets = input_offset_by_experts;
  uint32_t const* output_scale_offset = output_scale_offset_by_experts;
  float const* sf_scales = SFScale;
  if constexpr (STAGE_EXPERT_TABLES) {
    extern __shared__ __align__(16) uint32_t shared_expert_tables[];
    int const table_stride = expert_table_stride(n_experts);
    uint32_t* const shared_input_offsets = shared_expert_tables;
    uint32_t* const shared_sf_offsets = shared_expert_tables + table_stride;
    float* const shared_SFScale =
        reinterpret_cast<float*>(shared_expert_tables + 2 * table_stride);
    load_expert_tables(shared_input_offsets, shared_sf_offsets, shared_SFScale,
                       input_offset_by_experts, output_scale_offset_by_experts,
                       SFScale, n_experts);
    __syncthreads();
    input_offsets = shared_input_offsets;
    output_scale_offset = shared_sf_offsets;
    sf_scales = shared_SFScale;
  }

  int32_t const numValidRows =
      min(numRows, static_cast<int32_t>(input_offsets[n_experts]));
  int const numValidVecs = numValidRows * colsPerRow;

  // A thread's rows only advance, so its expert is carried across iterations.
  int expert_idx = 0;

  // Whether a thread's lookups after its first gallop (see search_expert).
  // A thread advances by the same stride from the start of the rows to past
  // the end, so over its N iterations it crosses about all n = n_experts
  // expert boundaries, a mean jump of d = n / N experts per lookup. Galloping
  // wins below d = sqrt(n / 2):
  //   n / N < sqrt(n / 2)  <=>  n^2 / N^2 < n / 2  <=>  N^2 > 2 * n.
  // The first lookup starts from expert 0, about n / 2 experts short of its
  // row, so it always bisects.
  int const N = div_round_up(numValidVecs, gridStride);
  bool const search_use_gallop = (int64_t)N * N > 2 * n_experts;
  // simplify "search_use_gallop && globalIdx >= gridStride" to "globalIdx >=
  // gallopFrom"
  int const gallopFrom = search_use_gallop ? gridStride : INT_MAX;

  // The row advances by rowStride per iteration, plus at most one carry.
  uint32_t rowIdx = tid / colsPerRow;

  // Each global thread processes one element
  for (int globalIdx = tid; globalIdx < numValidVecs;
       globalIdx += gridStride, rowIdx += rowStride) {
    uint32_t colIdx = globalIdx - rowIdx * colsPerRow;
    if (colIdx >= static_cast<uint32_t>(colsPerRow)) {
      ++rowIdx;
      colIdx -= colsPerRow;
    }

    // The input is loaded before the expert lookup so the lookup overlaps it.
    PackedVec in_vec, in_vec_up;
    if constexpr (FUSE_SILU_MUL) {
      // Gate half of row rowIdx (2 * colsPerRow vectors per row), column
      // colIdx.
      PackedVec const* const in_vec_ptr =
          in_vecs + ((int64_t)globalIdx + (int64_t)rowIdx * colsPerRow);
      in_vec = LoadPackedVec(in_vec_ptr);
      in_vec_up = LoadPackedVec(in_vec_ptr + colsPerRow);
    } else {
      in_vec = LoadPackedVec(in_vecs + globalIdx);
    }

    // Gallop as decided above (see gallopFrom). Tables too small to stage
    // always bisect. SiLU always bisects: the benefit of galloping is too small
    // compared to just relying on bisect.
    // globalIdx >= gallopFrom is a result of merging these two boolean
    // expressions: search_use_gallop && globalIdx >= gridStride
    expert_idx = search_expert(
        input_offsets, n_experts, expert_idx, rowIdx,
        !FUSE_SILU_MUL && STAGE_EXPERT_TABLES && globalIdx >= gallopFrom);

    uint32_t const rowIdx_in_expert = rowIdx - input_offsets[expert_idx];

    // Optionally apply fused SiLU+Mul
    if constexpr (FUSE_SILU_MUL) {
      // On B200 the memory is fast enough that this kernel turns compute
      // bound on large n_expert, and SiLU bottlenecks on the XU pipe: silu()
      // issues two MUFU ops per element (ex2 for __expf, rcp for __fdividef).
      // The tanh form silu(x) = 0.5x * (1 + tanh.approx(0.5x)) needs only one,
      // halving XU pressure, but it is not bit-identical to the current silu().
      in_vec = compute_silu_mul(in_vec, in_vec_up);
    }

    // no need to check for nullptr if we staged the expert SFScale in shared
    // memory
    float const SFScaleVal = (STAGE_EXPERT_TABLES || sf_scales != nullptr)
                                 ? sf_scales[expert_idx]
                                 : 1.0f;

    uint32_t* SFout_in_expert =
        SFout + output_scale_offset[expert_idx] * numKTiles;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);

    u32x2 const o =
        cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
            in_vec, SFScaleVal, sf_out);

    // Each thread packs 16 FP4 values into one 64-bit store, so the output is
    // indexed by globalIdx like the input vectors.
    static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1,
                  "we can only index output with globalIdx when "
                  "CVT_FP4_NUM_THREADS_PER_SF is equal to 1");
    st64_cs(reinterpret_cast<int64_t*>(out) + globalIdx,
            static_cast<int64_t>(static_cast<uint64_t>(o.hi) << 32 | o.lo));
  }
}

template <typename T, bool FUSE_SILU_MUL = false>
void quant_impl(void* output, void* output_scale, void* input,
                void* input_global_scale, void* input_offset_by_experts,
                void* output_scale_offset_by_experts, int m_topk, int k,
                int n_experts, cudaStream_t stream) {
  if (m_topk == 0) return;

  int const cols_per_row = k / ELTS_PER_THREAD;
  // SF layout: 64-column K tiles per row.
  int const num_k_tiles = (k + 63) / 64;
  int64_t const total_thread_work = (int64_t)m_topk * cols_per_row;
  if (n_experts >= CVT_FP4_EXPERT_SMEM_MIN_EXPERTS) {
    size_t const shared_mem_size = expert_tables_smem_size(n_experts);
    auto const kernel = cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, true>;
    auto const [threads, blocks] =
        one_wave_launch_dims(kernel, total_thread_work, shared_mem_size);
    int const grid_stride = threads * blocks;
    kernel<<<blocks, threads, shared_mem_size, stream>>>(
        m_topk, k, reinterpret_cast<T*>(input),
        reinterpret_cast<float*>(input_global_scale),
        reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(output_scale),
        reinterpret_cast<uint32_t*>(input_offset_by_experts),
        reinterpret_cast<uint32_t*>(output_scale_offset_by_experts), n_experts,
        cols_per_row, num_k_tiles, grid_stride / cols_per_row, grid_stride);
  } else {
    auto const kernel = cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, false>;
    auto const [threads, blocks] =
        one_wave_launch_dims(kernel, total_thread_work, 0);
    int const grid_stride = threads * blocks;
    kernel<<<blocks, threads, 0, stream>>>(
        m_topk, k, reinterpret_cast<T*>(input),
        reinterpret_cast<float*>(input_global_scale),
        reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(output_scale),
        reinterpret_cast<uint32_t*>(input_offset_by_experts),
        reinterpret_cast<uint32_t*>(output_scale_offset_by_experts), n_experts,
        cols_per_row, num_k_tiles, grid_stride / cols_per_row, grid_stride);
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
  const torch::stable::accelerator::DeviceGuard device_guard(
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
  const torch::stable::accelerator::DeviceGuard device_guard(
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
