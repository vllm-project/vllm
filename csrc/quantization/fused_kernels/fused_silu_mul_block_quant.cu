// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../torch_utils.h"

#include "../../dispatch_utils.h"
#include "../../core/cuda_compat.h"
#include "quant_conversions.cuh"

#include <algorithm>
#include <cstring>

#if defined(__gfx1200__) || defined(__gfx1201__)
  #define VLLM_SILU_QUANT_RDNA4 1
#endif

namespace vllm {

// RDNA4 runs the vectorized kernel below VALU-bound, so there both kernels use
// the hardware exp/rcp for the sigmoid and quantize OCP fp8 with a reciprocal
// group scale and the packed hardware cvt. The kernels stay bitwise identical.
__device__ __forceinline__ float silu_and_mul_fn(float const gate,
                                                 float const up) {
#ifdef VLLM_SILU_QUANT_RDNA4
  float const sigmoid_gate = __builtin_amdgcn_rcpf(1.0f + __expf(-gate));
#else
  float const sigmoid_gate = 1.0f / (1.0f + expf(-gate));
#endif
  return gate * sigmoid_gate * up;
}

#ifdef VLLM_SILU_QUANT_RDNA4
// Two values to OCP e4m3 (round to nearest even), low half of the result.
__device__ __forceinline__ uint32_t scaled_cvt_pk_fp8(float const a,
                                                      float const b,
                                                      float const inv_scale) {
  float const qmax = quant_type_max_v<c10::Float8_e4m3fn>;
  return __builtin_amdgcn_cvt_pk_fp8_f32(
      fmaxf(-qmax, fminf(a * inv_scale, qmax)),
      fmaxf(-qmax, fminf(b * inv_scale, qmax)), 0, false);
}
#endif

template <typename scalar_out_t>
__device__ __forceinline__ scalar_out_t quant_fn(float const x,
                                                 float const group_scale) {
#ifdef VLLM_SILU_QUANT_RDNA4
  if constexpr (std::is_same_v<scalar_out_t, c10::Float8_e4m3fn>) {
    return scalar_out_t(
        static_cast<uint8_t>(scaled_cvt_pk_fp8(x, x, 1.0f / group_scale)),
        scalar_out_t::from_bits());
  }
#endif
  return ScaledQuant<scalar_out_t, false>::quant_fn(x, group_scale);
}

// Logic: one thread block per (token, group) pair

template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          int32_t group_size>
__global__ void silu_and_mul_per_block_quant_kernel(
    scalar_out_t* __restrict__ out,  // Output: [num_tokens, hidden_size] in
                                     // FP8/INT8
    float* __restrict__ scales,      // Output: [num_tokens, hidden_size /
                                 // group_size] or [hidden_size / group_size,
                                 // num_tokens]
    scalar_t const* __restrict__ input,  // Input: [num_tokens, hidden_size * 2]
    float const* scale_ub,               // Optional scale upper bound
    int32_t const hidden_size  // Output hidden size (input is 2x this)
) {
  static_assert((group_size & (group_size - 1)) == 0,
                "group_size must be a power of 2 for correct reduction");

  // Grid: (num_tokens, num_groups)
  int64_t const token_idx = blockIdx.x;
  int const group_idx = blockIdx.y;
  int const tid = threadIdx.x;  // tid in [0, group_size)
  int const num_tokens = gridDim.x;

  // Input layout: [gate || up] concatenated along last dimension
  int const input_stride = hidden_size * 2;
  int const group_start = group_idx * group_size;

  // Pointers to this token's data
  scalar_t const* token_input_gate =
      input + token_idx * input_stride + group_start;
  scalar_t const* token_input_up = token_input_gate + hidden_size;
  scalar_out_t* token_output = out + token_idx * hidden_size + group_start;

  // Scale pointer for this group
  int const num_groups = gridDim.y;
  float* group_scale_ptr = is_scale_transposed
                               ? scales + group_idx * num_tokens + token_idx
                               : scales + token_idx * num_groups + group_idx;

  // Shared memory for reduction (compile-time sized)
  __shared__ float shared_max[group_size];

  // Step 1: Each thread loads one element, computes SiLU, stores in register
  float gate = static_cast<float>(token_input_gate[tid]);
  float up = static_cast<float>(token_input_up[tid]);

  // Compute SiLU(gate) * up
  float result = silu_and_mul_fn(gate, up);  // Keep in register

  // Step 2: Reduce to find group max
  shared_max[tid] = fabsf(result);
  __syncthreads();

// Power-of-2 reduction (group_size guaranteed to be power of 2)
#pragma unroll
  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }

  // Step 3: Compute scale (thread 0), broadcast via shared memory
  if (tid == 0) {
    float group_max = shared_max[0];

    float const quant_range = quant_type_max_v<scalar_out_t>;
    float group_scale = group_max / quant_range;

    // Apply scale upper bound if provided
    if (scale_ub != nullptr) {
      group_scale = fminf(group_scale, *scale_ub);
    }

    // Use minimum safe scaling factor
    group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());

    // Store scale to global memory
    *group_scale_ptr = group_scale;

    // Reuse shared_max[0] to broadcast scale
    shared_max[0] = group_scale;
  }
  __syncthreads();

  float group_scale = shared_max[0];

  // Step 4: Quantize and write output
  token_output[tid] = quant_fn<scalar_out_t>(result, group_scale);
}

// Vectorized variant: each group is handled by group_size / 8 lanes of a
// warp, each loading 8 contiguous 16-bit elements of gate and up (16B loads)
// and storing 8 quantized bytes. The group max is reduced with warp shuffles,
// so there is no shared memory or __syncthreads. Requires 16B-aligned input
// and 8B-aligned output rows. Same arithmetic as the kernel above.
template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          int32_t group_size>
__global__ void silu_and_mul_per_block_quant_vec_kernel(
    scalar_out_t* __restrict__ out, float* __restrict__ scales,
    scalar_t const* __restrict__ input, float const* scale_ub,
    int32_t const hidden_size, int32_t const num_groups,
    int64_t const num_tokens) {
  constexpr int kVec = 16 / sizeof(scalar_t);
  constexpr int kLanes = group_size / kVec;
  static_assert(kVec == 8, "expects 16-bit input");
  static_assert(kLanes > 0 && kLanes <= 32 && (32 % kLanes) == 0);

  int const lane = threadIdx.x % kLanes;
#ifdef VLLM_SILU_QUANT_RDNA4
  // A HIP grid has at most 2^32 work-items.
  uint32_t const gid = (blockIdx.x * blockDim.x + threadIdx.x) / kLanes;
  bool const valid = gid < static_cast<uint32_t>(num_tokens) * num_groups;
  int64_t const token_idx = valid ? gid / static_cast<uint32_t>(num_groups) : 0;
  int const group_idx = valid ? gid % static_cast<uint32_t>(num_groups) : 0;
#else
  int64_t const gid =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kLanes;
  bool const valid = gid < num_tokens * num_groups;
  int64_t const token_idx = valid ? gid / num_groups : 0;
  int const group_idx = valid ? static_cast<int>(gid % num_groups) : 0;
#endif

  int64_t const in_off = token_idx * (2 * static_cast<int64_t>(hidden_size)) +
                         group_idx * group_size + lane * kVec;
  alignas(16) scalar_t g[kVec];
  alignas(16) scalar_t u[kVec];
  if (valid) {
    *reinterpret_cast<uint4*>(g) =
        *reinterpret_cast<uint4 const*>(input + in_off);
    *reinterpret_cast<uint4*>(u) =
        *reinterpret_cast<uint4 const*>(input + in_off + hidden_size);
  } else {
    *reinterpret_cast<uint4*>(g) = make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(u) = make_uint4(0, 0, 0, 0);
  }

  float r[kVec];
  float local_max = 0.f;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    r[i] = silu_and_mul_fn(static_cast<float>(g[i]), static_cast<float>(u[i]));
    local_max = fmaxf(local_max, fabsf(r[i]));
  }
#pragma unroll
  for (int off = kLanes / 2; off > 0; off >>= 1) {
    local_max =
        fmaxf(local_max, VLLM_SHFL_XOR_SYNC_WIDTH(local_max, off, kLanes));
  }

  float group_scale = local_max / quant_type_max_v<scalar_out_t>;
  if (scale_ub != nullptr) {
    group_scale = fminf(group_scale, *scale_ub);
  }
  group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());
  if (!valid) {
    return;
  }

  alignas(8) scalar_out_t q[kVec];
#ifdef VLLM_SILU_QUANT_RDNA4
  if constexpr (std::is_same_v<scalar_out_t, c10::Float8_e4m3fn>) {
    float const inv_scale = 1.0f / group_scale;
  #pragma unroll
    for (int i = 0; i < kVec; i += 2) {
      reinterpret_cast<uint16_t*>(q)[i / 2] =
          static_cast<uint16_t>(scaled_cvt_pk_fp8(r[i], r[i + 1], inv_scale));
    }
  } else
#endif
  {
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      q[i] = quant_fn<scalar_out_t>(r[i], group_scale);
    }
  }
  *reinterpret_cast<uint2*>(out + token_idx * hidden_size +
                            group_idx * group_size + lane * kVec) =
      *reinterpret_cast<uint2 const*>(q);
  if (lane == 0) {
    float* scale_ptr = is_scale_transposed
                           ? scales + group_idx * num_tokens + token_idx
                           : scales + token_idx * num_groups + group_idx;
    *scale_ptr = group_scale;
  }
}

}  // namespace vllm

#ifdef USE_ROCM
// Full-die gfx1201 holds back the end of a short dispatch by ~10us when, at
// full occupancy, its waves fill the last max(16, 4 * waves per block) wave
// slots below a multiple of the device's resident-wave capacity
// (ROCm/TheRock#8634). Blocks past the multiple, even ones that exit at once,
// avoid it, so pad such grids by one block.
static unsigned pad_grid_off_full_fill(unsigned blocks, int block_size) {
  cudaDeviceProp const* prop = get_device_prop();
  int const wave_size = prop->warpSize;
  int const waves_per_block = block_size / wave_size;
  int64_t const capacity = static_cast<int64_t>(prop->multiProcessorCount) *
                           prop->maxThreadsPerMultiProcessor / wave_size;
  int64_t const waves = static_cast<int64_t>(blocks) * waves_per_block;
  int64_t const rem = waves % capacity;
  if (capacity % waves_per_block != 0 ||
      (rem != 0 && rem <= capacity - std::max(16, 4 * waves_per_block))) {
    return blocks;
  }
  int64_t const full = rem == 0 ? waves : waves - rem + capacity;
  return static_cast<unsigned>(full / waves_per_block + 1);
}
#endif

void silu_and_mul_per_block_quant(torch::stable::Tensor& out,
                                  torch::stable::Tensor const& input,
                                  torch::stable::Tensor& scales,
                                  int64_t group_size,
                                  std::optional<torch::stable::Tensor> scale_ub,
                                  bool is_scale_transposed) {
  static torch::headeronly::ScalarType kFp8Type =
      is_fp8_ocp() ? torch::headeronly::ScalarType::Float8_e4m3fn
                   : torch::headeronly::ScalarType::Float8_e4m3fnuz;

  STD_TORCH_CHECK(out.scalar_type() == kFp8Type ||
                  out.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Input must be FP16 or BF16");
  STD_TORCH_CHECK(scales.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(group_size == 128 || group_size == 64,
                  "Unsupported group size: ", group_size);

  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(out.scalar_type() == kFp8Type);
  }

  int32_t hidden_size = out.size(-1);
  auto num_tokens = input.size(0);
  int32_t num_groups = hidden_size / group_size;

  STD_TORCH_CHECK(input.size(-1) == hidden_size * 2,
                  "input last dim must be 2x output hidden_size");
  STD_TORCH_CHECK(hidden_size % group_size == 0,
                  "hidden_size must be divisible by group_size");

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  dim3 grid(num_tokens, num_groups);
  dim3 block(group_size);
#ifdef USE_ROCM
  bool const on_rdna4 =
      std::strncmp(get_device_prop()->gcnArchName, "gfx120", 6) == 0;
#else
  constexpr bool on_rdna4 = false;
#endif
  // Below ~3k groups the one-block-per-group kernel has lower latency, except
  // on RDNA4.
  bool const use_vec =
      (on_rdna4 || static_cast<int64_t>(num_tokens) * num_groups >= 3072) &&
      input.element_size() == 2 &&
      reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(out.const_data_ptr()) % 8 == 0;
  constexpr int kVecBlock = 256;

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant", [&] {
        using scalar_in_t = scalar_t;

        VLLM_STABLE_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_block_quant", [&] {
              using scalar_out_t = scalar_t;

              VLLM_STABLE_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
                VLLM_STABLE_DISPATCH_BOOL(
                    is_scale_transposed, transpose_scale, [&] {
                      if constexpr (sizeof(scalar_in_t) == 2) {
                        if (use_vec) {
                          constexpr int kLanes = gs / 8;
                          int64_t const total =
                              static_cast<int64_t>(num_tokens) * num_groups *
                              kLanes;
                          dim3 vgrid(static_cast<unsigned>(
                              (total + kVecBlock - 1) / kVecBlock));
#ifdef USE_ROCM
                          if (on_rdna4) {
                            vgrid.x =
                                pad_grid_off_full_fill(vgrid.x, kVecBlock);
                          }
#endif
                          vllm::silu_and_mul_per_block_quant_vec_kernel<
                              scalar_in_t, scalar_out_t, transpose_scale, gs>
                              <<<vgrid, kVecBlock, 0, stream>>>(
                                  out.mutable_data_ptr<scalar_out_t>(),
                                  scales.mutable_data_ptr<float>(),
                                  input.const_data_ptr<scalar_in_t>(),
                                  scale_ub.has_value()
                                      ? scale_ub->const_data_ptr<float>()
                                      : nullptr,
                                  hidden_size, num_groups, num_tokens);
                          return;
                        }
                      }
                      vllm::silu_and_mul_per_block_quant_kernel<
                          scalar_in_t, scalar_out_t, transpose_scale, gs>
                          <<<grid, block, 0, stream>>>(
                              out.mutable_data_ptr<scalar_out_t>(),
                              scales.mutable_data_ptr<float>(),
                              input.const_data_ptr<scalar_in_t>(),
                              scale_ub.has_value()
                                  ? scale_ub->const_data_ptr<float>()
                                  : nullptr,
                              hidden_size);
                    });
              });
            });
      });
}
