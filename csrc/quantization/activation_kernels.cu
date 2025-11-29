#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include "core/math.hpp"
#include "../cuda_compat.h"
#include "dispatch_utils.h"

#include "quantization/w8a8/fp8/common.cuh"

#include <c10/util/Float8_e4m3fn.h>

#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <cuda_fp8.h>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
  #include <hip/hip_fp8.h>

typedef __hip_bfloat162 __nv_bfloat162;
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat16_raw __nv_bfloat16_raw;
  #if defined(HIP_FP8_TYPE_OCP)
typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
typedef __hip_fp8x4_e4m3 __nv_fp8x4_e4m3;
  #else
// ROCm 6.2 fallback: only *_fnuz types exist
typedef __hip_fp8_e4m3_fnuz __nv_fp8_e4m3;
typedef __hip_fp8x4_e4m3_fnuz __nv_fp8x4_e4m3;
  #endif
#endif

#include "core/registration.h"
namespace vllm {

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

// Activation and gating kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          typename fp8_type>
__global__ void act_and_mul_quant_kernel(
    fp8_type* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const float* scale, const int d) {
  const int32_t blocks_per_token = gridDim.y;

  const int32_t elems_per_128bit_load = (128 / 8) / sizeof(scalar_t);

  // We don't expect the hidden dimension to exceed 32 bits so int32 should
  // be safe here.
  const int32_t tgt_elems_per_block = div_ceil(d, blocks_per_token);
  const int32_t elems_per_block =
      round_to_next_multiple_of(tgt_elems_per_block, elems_per_128bit_load);
  const int32_t block_start = blockIdx.y * elems_per_block;
  int32_t block_end = block_start + elems_per_block;
  block_end = block_end > d ? d : block_end;

  // token_idx is 64 bit to prevent 32 bit overflow when the number of tokens
  // is very large
  const int64_t token_idx = blockIdx.x;
  const scalar_t* __restrict__ x_ptr = input + token_idx * 2 * d;
  const scalar_t* __restrict__ y_ptr = input + token_idx * 2 * d + d;
  fp8_type* __restrict__ out_ptr = out + token_idx * d;

  // 128-bit vectorized code
  const int32_t vec_loop_end =
      round_to_previous_multiple_of(elems_per_128bit_load, block_end);
  const int32_t vec_end_idx = vec_loop_end / elems_per_128bit_load;
  const int32_t vec_start_idx = block_start / elems_per_128bit_load;

  const int4* __restrict__ x_128bit_ptr = reinterpret_cast<const int4*>(x_ptr);
  const int4* __restrict__ y_128bit_ptr = reinterpret_cast<const int4*>(y_ptr);
  int2* __restrict__ out_128bit_ptr = reinterpret_cast<int2*>(out_ptr);

  float inverted_scale = 1 / *scale;
#pragma unroll
  for (int32_t vec_idx = vec_start_idx + threadIdx.x; vec_idx < vec_end_idx;
       vec_idx += blockDim.x) {
    const int4 x_128bit = VLLM_LDG(&x_128bit_ptr[vec_idx]);
    const int4 y_128bit = VLLM_LDG(&y_128bit_ptr[vec_idx]);
    using scalar_128bit_vec_t = std::array<scalar_t, elems_per_128bit_load>;
    using scalar_64bit_vec_t = std::array<fp8_type, elems_per_128bit_load>;

    scalar_64bit_vec_t out_vec;
    const auto x_vec = reinterpret_cast<scalar_128bit_vec_t const&>(x_128bit);
    const auto y_vec = reinterpret_cast<scalar_128bit_vec_t const&>(y_128bit);

#pragma unroll
    for (int i = 0; i < elems_per_128bit_load; i++) {
      out_vec[i] = scaled_fp8_conversion<true, fp8_type>(
          ACT_FN(x_vec[i]) * y_vec[i], inverted_scale);
    }

    out_128bit_ptr[vec_idx] = reinterpret_cast<const int2&>(out_vec);
  }

  // Scalar cleanup code
  if (block_end > vec_loop_end) {
    for (int64_t idx = vec_loop_end + threadIdx.x; idx < block_end;
         idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&x_ptr[idx]);
      const scalar_t y = VLLM_LDG(&y_ptr[idx]);
      out_ptr[idx] =
          scaled_fp8_conversion<true, fp8_type>(ACT_FN(x) * y, inverted_scale);
    }
  }
}

__device__ __forceinline__ float silu(float x) {
  return __fdividef(x, (1.f + expf(-x)));
}

__device__ __forceinline__ float2 silu2(float2 x) {
  return make_float2(silu(x.x), silu(x.y));
}

__device__ __forceinline__ __nv_bfloat162 silu2_v2(float2 x) {
#ifndef USE_ROCM
  return make_bfloat162(__float2bfloat16_rn(silu(x.x)),
                        __float2bfloat16_rn(silu(x.y)));
#else
  return __float22bfloat162_rn(make_float2(silu(x.x), silu(x.y)));
#endif
}

#ifndef USE_ROCM
__device__ __forceinline__ float warp_max(float v) {
  static constexpr unsigned FULL_MASK = 0xffffffffu;
  for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
    v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, offset));
  }
  return v;
}

__device__ __forceinline__ __nv_bfloat16 warp_max(__nv_bfloat16 v) {
  static constexpr unsigned FULL_MASK = 0xffffffffu;
  for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
    v = __hmax(v, __shfl_xor_sync(FULL_MASK, v, offset));
  }
  return v;
}
#endif

template <typename T, typename U>
__device__ __forceinline__ void cp_async4(T* _smem_ptr, const U* _glob_ptr) {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  auto smem_ptr = reinterpret_cast<void*>(_smem_ptr);
  auto glob_ptr = reinterpret_cast<const void*>(_glob_ptr);
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
#else
  _smem_ptr[0] = _glob_ptr[0];
#endif
}

__device__ __forceinline__ void cp_async_fence() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#else
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#else
#endif
}

template <>
__device__ __forceinline__ void cp_async_wait<0>() {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::);
#else
#endif
}

__device__ __forceinline__ float clip(float v, float mmin, float mmax) {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
  return fminf(mmax, fmaxf(v, mmin));
#else
#endif
}

__device__ __forceinline__ __nv_bfloat16 clip(__nv_bfloat16 v,
                                              __nv_bfloat16 mmin,
                                              __nv_bfloat16 mmax) {
  return __hmin(mmax, __hmax(v, mmin));
}

__device__ __forceinline__ __nv_bfloat162 clip(__nv_bfloat162 v,
                                               __nv_bfloat162 mmin,
                                               __nv_bfloat162 mmax) {
  return __hmin2(mmax, __hmax2(v, mmin));
}

// We use the following values for fp8 min/max:
//  __nv_fp8_e4m3 = (-448, +448)
//  __nv_fp8_e4m3uz = (-240.0, +240.0)
// It is currently assumed that only
template <class T>
constexpr __nv_bfloat16 get_fp8_max() {
  static_assert(std::is_same_v<T, c10::Float8_e4m3fn> ||
                std::is_same_v<T, c10::Float8_e4m3fnuz>);
  if constexpr (std::is_same_v<T, c10::Float8_e4m3fn>) {
    return __nv_bfloat16(__nv_bfloat16_raw{.x = 17376});
  } else {
    return __nv_bfloat16(__nv_bfloat16_raw{.x = 17264});
  }
}

template <class T>
constexpr __nv_bfloat16 get_fp8_min() {
  static_assert(std::is_same_v<T, c10::Float8_e4m3fn> ||
                std::is_same_v<T, c10::Float8_e4m3fnuz>);
  if constexpr (std::is_same_v<T, c10::Float8_e4m3fn>) {
    return __nv_bfloat16(__nv_bfloat16_raw{.x = 50144});
  } else {
    return __nv_bfloat16(__nv_bfloat16_raw{.x = 50032});
  }
}

template <typename Idx_t>
__device__ __forceinline__ int warp_expert_search(
    int idx, int n, const Idx_t* __restrict__ input, Idx_t val) {
  const Idx_t* input_ptr = input + idx;
  int base_offset = 0;

  for (;;) {
    bool move_on = (idx < n && *input_ptr <= val);

    unsigned mask = __ballot_sync(0xffffffff, move_on);

    if (mask != 0xffffffffu) {
      int last_lane = 31 - __clz(mask);
      return base_offset + last_lane;
    }

    input_ptr += 32;
    base_offset += 32;
    idx += 32;
  }
}

template <int num_parallel_tokens>
__device__ __forceinline__ void token_bounds(int32_t n_tokens,
                                             int32_t worker_id,
                                             int32_t& n_tokens_lower,
                                             int32_t& n_tokens_upper) {
  if (n_tokens < num_parallel_tokens && worker_id < n_tokens) {
    if (worker_id >= num_parallel_tokens) return;
    n_tokens_lower = worker_id;
    n_tokens_upper = worker_id + 1;
  } else {
    int32_t chunk_size = n_tokens / num_parallel_tokens;
    int32_t residual = n_tokens - chunk_size * num_parallel_tokens;
    auto calc_id = [&](int32_t id) {
      if (id < residual)
        return min(n_tokens, id * (chunk_size + 1));
      else
        return min(n_tokens, id * chunk_size + residual);
    };
    n_tokens_lower = calc_id(worker_id);
    n_tokens_upper = calc_id(worker_id + 1);
  }
}

template <int BLOCK_COUNT, int SMEM_SIZE_BYTES_Y, typename fp8_type,
          typename scale_t, int THREADS, typename Idx_t, bool CEIL_UE8M0,
          int GROUP_SIZE = 128, int NUM_STAGES = 3>
__global__ void silu_mul_fp8_quant_deep_gemm_kernel(
    const __nv_bfloat16* __restrict__ _input, fp8_type* __restrict__ _y_q,
    scale_t* __restrict__ _y_s, const int32_t* __restrict__ tokens_per_expert,
    // sizes
    Idx_t E, Idx_t T, Idx_t H,
    // strides (in elements)
    Idx_t stride_i_e, Idx_t stride_i_t, Idx_t stride_i_h, Idx_t stride_yq_e,
    Idx_t stride_yq_t, Idx_t stride_yq_h, Idx_t stride_ys_e, Idx_t stride_ys_t,
    Idx_t stride_ys_g, Idx_t stride_ys_p, Idx_t stride_counts_e) {
#ifndef USE_ROCM
  static constexpr int NUM_WARPS = THREADS / WARP_SIZE;

  static constexpr int LOAD_STAGE_SIZE = 2 * GROUP_SIZE / 8;
  static constexpr int LOAD_STAGE_MOD = NUM_STAGES * LOAD_STAGE_SIZE;

  static constexpr int COMPUTE_STAGE_SIZE = 2 * GROUP_SIZE / 4;
  static constexpr int COMPUTE_STAGE_MOD = COMPUTE_STAGE_SIZE * NUM_STAGES;

  extern __shared__ __align__(16) __int128_t smem_128[];

  int* s_expert_offsets =
      reinterpret_cast<int*>(smem_128 + (SMEM_SIZE_BYTES_Y / 16));

  static constexpr __nv_bfloat16 fp8_min = get_fp8_min<fp8_type>();
  static constexpr __nv_bfloat16 fp8_max = get_fp8_max<fp8_type>();
  // We assign EPS with it's 16-bit unsigned counterpart to allow constexpr.
  static constexpr __nv_bfloat16 EPS = (__nv_bfloat16_raw{.x = 11996});
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 0x1f;

  int running_sum{};
  if (!warp_id) {
    for (int i = 0; i < E; i += WARP_SIZE) {
      bool valid = (i + threadIdx.x) < E;
      int value =
          (valid ? tokens_per_expert[i + threadIdx.x * stride_counts_e] : 0) +
          (!lane_id ? running_sum : 0);

      for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(0xFFFFFFFFu, value, offset);
        if (lane_id >= offset) value += n;
      }

      if (valid) {
        s_expert_offsets[i + threadIdx.x + 1] = value;
      }

      running_sum = __shfl_sync(0xFFFFFFFFu, value, WARP_SIZE - 1);
    }

    if (!lane_id) {
      s_expert_offsets[0] = 0;
    }
  }

  __syncthreads();

  int32_t total_tokens = s_expert_offsets[E];

  const int warp_position_yq = warp_id * (H / NUM_WARPS);
  const int warp_position_scales = warp_id * (H / (GROUP_SIZE * NUM_WARPS));

  // A single block will handle tokens_per_block tokens.
  // Each block i iterates over tokens of a slice of n_tokens =
  // expert_counts[i], with the size of chunk being
  // (n_tokens / NUM_PARALLEL_TOKENS) + residual, instead of
  // updiv(n_tokens, NUM_PARALLEL_TOKENS) for better scheduling.

  // Each warp will get space to store its hidden dim for gate and up.
  __int128_t* s_hidden_load = smem_128 + warp_id * ((2 * 128 / 8) * NUM_STAGES);
  __int128_t* smem_load_ptr = s_hidden_load + lane_id;

  const __nv_bfloat16 fp8_inv = __hdiv(__float2bfloat16(1.f), fp8_max);

  int32_t compute_pipeline_offset_64 = 0;
  int32_t load_stage_offset{};
  const __nv_bfloat16 one_bf16 = __float2bfloat16_rn(1.f);

  __int64_t* smem_compute_ptr = reinterpret_cast<__int64_t*>(smem_128) +
                                warp_id * (2 * (GROUP_SIZE / 4) * NUM_STAGES) +
                                lane_id;
  __int64_t* s_gate64_ptr = smem_compute_ptr;
  __int64_t* s_up64_ptr = smem_compute_ptr + GROUP_SIZE / 4;

  int tokens_lower, tokens_upper;

  token_bounds<BLOCK_COUNT>(total_tokens, blockIdx.x, tokens_lower,
                            tokens_upper);

  Idx_t expert_id{}, expert_offset{}, next_expert_offset{};
  int token_id = tokens_lower;
  int32_t t_load{};

  if (token_id < tokens_upper) {
    expert_id = warp_expert_search<int>(lane_id, E, s_expert_offsets, token_id);
    expert_offset = s_expert_offsets[expert_id];
    next_expert_offset = s_expert_offsets[expert_id + 1];
  } else {
    // This thread block has no work to do.
    return;
  }

  int t_load_bound = H / (GROUP_SIZE * NUM_WARPS);

  Idx_t base_i = ((expert_id * stride_i_e) / 8) +
                 (token_id - expert_offset) * stride_i_t / 8;
  const Idx_t gate_warp_offset =
      warp_id * ((stride_i_h * H) / (8 * NUM_WARPS)) + (lane_id & 0b1111);

  const __int128_t* input_128_ptr =
      reinterpret_cast<const __int128_t*>(_input) + gate_warp_offset +
      ((lane_id < 16) ? 0 : ((H * stride_i_h) / 8));
  __int128_t* load_ptr = const_cast<__int128_t*>(input_128_ptr + base_i);

  auto token_offset = token_id - expert_offset;

  auto load_and_advance_y_pred = [&] {
    if (t_load < t_load_bound) {
      // Here we are simply continuing to load data
      // from the current token.
      auto smem_load_ptr_staged = smem_load_ptr + load_stage_offset;

      // It is very important that LOAD_STAGE_SIZE is constexpr to avoid
      // unnecessary ALU ops.
      load_stage_offset += LOAD_STAGE_SIZE;
      load_stage_offset %= LOAD_STAGE_MOD;

      cp_async4(smem_load_ptr_staged, load_ptr);
      load_ptr += GROUP_SIZE / 8;
      ++t_load;
    } else if (token_id + 1 < tokens_upper) {
      // We loaded everything from the current token, let's move on
      // to the next one, and we checked that we have more tokens to load.
      ++token_id;
      t_load = 0;
      if (token_id >= next_expert_offset) {
        // We need to find the next expert.
        do {
          // This is a loop because it's possible
          // that some experts are assigned 0 tokens.
          // NOTE: We are guaranteed that there's at least
          // one more token left so we don't have to check for
          // expert_id bounds.
          ++expert_id;
          // This skips 1 memory read.
          expert_offset = next_expert_offset;
          next_expert_offset = s_expert_offsets[expert_id + 1];
        } while (next_expert_offset == expert_offset);

        base_i = expert_id * (stride_i_e / 8);
        token_offset = 0;
        load_ptr = const_cast<__int128_t*>(input_128_ptr + base_i);
      } else {
        // We remain within the same expert, so just
        // move by H/4 __int128_t (2 * H/8).
        base_i += stride_yq_t / 4;
        token_offset++;
      }

      load_ptr = const_cast<__int128_t*>(input_128_ptr + base_i);

      auto smem_load_ptr_staged = smem_load_ptr + load_stage_offset;

      // It is very important that LOAD_STAGE_SIZE is constexpr to avoid
      // unnecessary ALU ops.
      load_stage_offset += LOAD_STAGE_SIZE;
      load_stage_offset %= LOAD_STAGE_MOD;

      cp_async4(smem_load_ptr_staged, load_ptr);
      load_ptr += GROUP_SIZE / 8;
      ++t_load;
    }
    // We fence even if there is nothing to load to simplify pipelining.
    cp_async_fence();
  };

  // We need to warm-up the pipeline.
  #pragma unroll
  for (int i = 0; i < NUM_STAGES - 1; i++) {
    load_and_advance_y_pred();
  }

  __nv_fp8x4_e4m3* y_q_base_ptr =
      reinterpret_cast<__nv_fp8x4_e4m3*>(_y_q) + lane_id;

  Idx_t scale_group_offset = 0;
  if constexpr (std::is_same<scale_t, uint8_t>::value) {
    // packed int32_t format
    int pack_id = warp_position_scales / 4;
    int scale_in_pack = warp_position_scales % 4;
    scale_group_offset = pack_id * stride_ys_p + scale_in_pack * stride_ys_g;
  } else {
    scale_group_offset = warp_position_scales * stride_ys_g;
  }

  scale_t* const y_scale_base_ptr = _y_s + scale_group_offset;

  for (auto j = tokens_lower; j < tokens_upper; j++) {
    int current_group_id = warp_position_scales;  // Running count of which
                                                  // group is being processed
    const Idx_t base_ys = expert_id * stride_ys_e;
    auto y_s_ptr = y_scale_base_ptr + base_ys + token_offset * stride_ys_t;
    __nv_fp8x4_e4m3* y_q_ptr =
        y_q_base_ptr + (expert_id * stride_yq_e + token_offset * stride_yq_t +
                        warp_position_yq * stride_yq_h) /
                           4;
    const int COMPUTE_LIMIT = H / (GROUP_SIZE * NUM_WARPS);

    for (int i = 0; i < COMPUTE_LIMIT; i++) {
      cp_async_wait<NUM_STAGES - 2>();
      __syncthreads();
      load_and_advance_y_pred();

      __int64_t* gate64_ptr = s_gate64_ptr + compute_pipeline_offset_64;
      __int64_t* up64_ptr = s_up64_ptr + compute_pipeline_offset_64;

      // COMPUTE_STAGE_SIZE/MOD must also be constexpr!
      compute_pipeline_offset_64 += COMPUTE_STAGE_SIZE;
      compute_pipeline_offset_64 %= COMPUTE_STAGE_MOD;

      __int64_t gate64 = *gate64_ptr;
      __int64_t up64 = *up64_ptr;

      // Compute
      __nv_bfloat162 res[2];
      __nv_bfloat162* s_up_comp = reinterpret_cast<__nv_bfloat162*>(&up64);
      __nv_bfloat162* s_gate_comp = reinterpret_cast<__nv_bfloat162*>(&gate64);

  #pragma unroll
      for (int32_t k = 0; k < 2; ++k) {
        __nv_bfloat162 gate = silu2_v2(__bfloat1622float2(s_gate_comp[k]));
        res[k] = __hmul2(gate, s_up_comp[k]);
      }

      auto _y_max2 = __hmax2(__habs2(res[0]), __habs2(res[1]));

      _y_max2.x = __hmax(__hmax(_y_max2.x, _y_max2.y), EPS);

      __nv_bfloat16 y_s = __hmul(warp_max(_y_max2.x), fp8_inv);

      if constexpr (CEIL_UE8M0) {
        y_s = hexp2(hceil(hlog2(y_s)));
      }

      __nv_bfloat16 inv_y = __hdiv(one_bf16, y_s);

      auto y_s2 = make_bfloat162(inv_y, inv_y);

  #pragma unroll
      for (int32_t k = 0; k < 2; ++k) {
        res[k] = clip(__hmul2(res[k], y_s2), __bfloat162bfloat162(fp8_min),
                      __bfloat162bfloat162(fp8_max));
      }

      *y_q_ptr = __nv_fp8x4_e4m3(res[0], res[1]);
      y_q_ptr += WARP_SIZE * stride_yq_h;

      if (!lane_id) {
        // Store scales.
        if constexpr (std::is_same<scale_t, uint8_t>::value) {
          // Packed UE8MO format. Remove Mantissa.
          *y_s_ptr = reinterpret_cast<int16_t&>(y_s) >> 7;

          bool const jump_pack = (current_group_id + 1) % 4 == 0;
          // Minus 3 because we need to get to the first group in the
          // next pack.
          y_s_ptr += jump_pack ? (stride_ys_p - 3) : stride_ys_g;

        } else {
          // float32 format
          static_assert(std::is_same<scale_t, float>::value);
          *y_s_ptr = y_s;
          y_s_ptr += stride_ys_g;
        }

        current_group_id += 1;
      }
    }
  }
#endif
}

}  // namespace vllm

// Launch activation, gating, and quantize kernel.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL)                               \
  int d = input.size(-1) / 2;                                               \
  int64_t num_tokens = input.numel() / input.size(-1);                      \
  dim3 grid(num_tokens, num_tokens > 16 ? num_tokens > 32 ? 1 : 2 : 4);     \
  dim3 block(std::min(d, 512));                                             \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));         \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();             \
  VLLM_DISPATCH_FLOATING_TYPES(                                             \
      input.scalar_type(), "act_and_mul_kernel", [&] {                      \
        VLLM_DISPATCH_FP8_TYPES(                                            \
            out.scalar_type(), "fused_add_rms_norm_kernel_fp8_type", [&] {  \
              vllm::act_and_mul_quant_kernel<scalar_t, KERNEL<scalar_t>,    \
                                             fp8_t>                         \
                  <<<grid, block, 0, stream>>>(out.data_ptr<fp8_t>(),       \
                                               input.data_ptr<scalar_t>(),  \
                                               scale.data_ptr<float>(), d); \
            });                                                             \
      });

void silu_and_mul_quant(torch::Tensor& out,    // [..., d]
                        torch::Tensor& input,  // [..., 2 * d]
                        torch::Tensor& scale) {
  TORCH_CHECK(out.dtype() == torch::kFloat8_e4m3fn ||
              out.dtype() == torch::kFloat8_e4m3fnuz);
  TORCH_CHECK(input.dtype() == torch::kFloat16 ||
              input.dtype() == torch::kBFloat16);
  TORCH_CHECK(input.size(-1) % 2 == 0);
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel);
}

void persistent_masked_m_silu_mul_quant(
    const at::Tensor& input,              // (E, T, 2*H)
    const at::Tensor& tokens_per_expert,  // (E)
    at::Tensor& y_q,                      // (E, T, H) [OUT]
    at::Tensor& y_s,                      // (E, T, H//group_size) [OUT]
    bool cast_scale_ue8m0) {
#ifndef USE_ROCM

  // This kernel currently only supports H % 128 == 0 and assumes a
  // fixed GROUP_SIZE of 128.
  static constexpr int GROUP_SIZE = 128;

  TORCH_CHECK(input.dtype() == torch::kBFloat16);
  TORCH_CHECK(y_q.dtype() == torch::kFloat8_e4m3fn ||
              y_q.dtype() == torch::kFloat8_e4m3fnuz);
  TORCH_CHECK(input.size(-1) % (GROUP_SIZE * 2) == 0);

  bool const is_packed_ue8m0 =
      (y_s.dtype() == torch::kInt32 && cast_scale_ue8m0);
  TORCH_CHECK(y_s.dtype() == torch::kFloat32 || is_packed_ue8m0);

  using Idx_t = int64_t;

  Idx_t E = input.size(0);
  Idx_t T = input.size(1);
  Idx_t H = input.size(2) / 2;
  Idx_t stride_i_e = input.stride(0);
  Idx_t stride_i_t = input.stride(1);
  Idx_t stride_i_h = input.stride(2);
  Idx_t stride_yq_e = y_q.stride(0);
  Idx_t stride_yq_t = y_q.stride(1);
  Idx_t stride_yq_h = y_q.stride(2);

  Idx_t stride_counts_e = tokens_per_expert.stride(0);

  int const NUM_GROUPS = H / GROUP_SIZE;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // TODO: Get this from cuda_arch ?
  static constexpr int SILU_V2_BLOCK_COUNT = 132 * 32;

  #define KERNEL(BLOCK_COUNT, scale_t, STRIDE_YS_E, STRIDE_YS_T, STRIDE_YS_G,  \
                 STRIDE_YS_P, CEIL_UE8M0, THREAD_COUNT, STAGES)                \
    static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;                 \
    int sms = SILU_V2_BLOCK_COUNT;                                             \
    static constexpr int max_shared_mem_bytes =                                \
        GROUP_SIZE * 2 * STAGES * NUM_WARPS * 2;                               \
    dim3 grid(sms), block(THREAD_COUNT);                                       \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));          \
    VLLM_DISPATCH_FP8_TYPES(                                                   \
        y_q.scalar_type(), "silu_mul_fp8_quant_deep_gemm_kernel", [&] {        \
          vllm::silu_mul_fp8_quant_deep_gemm_kernel<                           \
              BLOCK_COUNT, max_shared_mem_bytes, fp8_t, scale_t, THREAD_COUNT, \
              Idx_t, CEIL_UE8M0, GROUP_SIZE, STAGES>                           \
              <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(  \
                  reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),          \
                  (fp8_t*)y_q.data_ptr(),                                      \
                  reinterpret_cast<scale_t*>(y_s.data_ptr()),                  \
                  reinterpret_cast<int32_t*>(tokens_per_expert.data_ptr()), E, \
                  T, H, stride_i_e, stride_i_t, stride_i_h, stride_yq_e,       \
                  stride_yq_t, stride_yq_h, STRIDE_YS_E, STRIDE_YS_T,          \
                  STRIDE_YS_G, STRIDE_YS_P, stride_counts_e);                  \
        });

  #define LAUNCH_ON_H(scale_t, STRIDE_YS_E, STRIDE_YS_T, STRIDE_YS_G,         \
                      STRIDE_YS_P, CEIL_UE8M0)                                \
    if (H >= 4096 && (NUM_GROUPS % 8) == 0) {                                 \
      /* 8 warp config */                                                     \
      static constexpr int NUM_STAGES = 4;                                    \
      static constexpr int THREAD_COUNT = 256;                                \
      KERNEL(SILU_V2_BLOCK_COUNT, scale_t, STRIDE_YS_E, STRIDE_YS_T,          \
             STRIDE_YS_G, STRIDE_YS_P, CEIL_UE8M0, THREAD_COUNT, NUM_STAGES); \
    } else {                                                                  \
      /* 1 warp config */                                                     \
      static constexpr int THREAD_COUNT = 32;                                 \
      KERNEL(SILU_V2_BLOCK_COUNT, scale_t, STRIDE_YS_E, STRIDE_YS_T,          \
             STRIDE_YS_G, STRIDE_YS_P, CEIL_UE8M0, THREAD_COUNT, 2);          \
    }

  Idx_t stride_ys_e = y_s.stride(0);
  Idx_t stride_ys_t = y_s.stride(1);
  Idx_t stride_ys_g = y_s.stride(2);
  Idx_t stride_ys_p = 0;
  if (!cast_scale_ue8m0) {
    TORCH_CHECK(!is_packed_ue8m0);
    LAUNCH_ON_H(float, stride_ys_e, stride_ys_t, stride_ys_g, stride_ys_p,
                false);
    return;
  }

  if (!is_packed_ue8m0) {
    // UE8M0 but not packed
    LAUNCH_ON_H(float, stride_ys_e, stride_ys_t, stride_ys_g, stride_ys_p,
                true);
    return;
  }

  TORCH_CHECK(cast_scale_ue8m0 && is_packed_ue8m0);
  TORCH_CHECK(y_s.dtype() == torch::kInt32);

  // Int32 packed ue8m0 scales tensor.
  // Let E, T, G be the number to experts, number of tokens and number of groups
  // respectively. Let, E = 2, T = 4, G = 6, in this case the int32 scales
  // tensor are of shape [1, 4, 2] and stride [8, 1, 4]. The scales are expected
  // to be arranged as follows,
  // [[T0G0-T0G1-T0G2-T0G3, T0G4-T0G5-X-X,],
  //  [T1G0-T1G1-T1G2-T1G3, T1G4-T1G5-X-X,]
  //  [T2G0-T2G1-T2G2-T2G3, T2G4-T2G5-X-X,]
  //  [T3G0-T3G1-T3G2-T3G3, T3G4-T3G5-X-X,]]
  // where, TxGy is the scale ue8m0 scale value of Token x, Group y.
  //
  // In memory (in bytes) the scale values are arranged as,
  //  [T0G0, T0G1, T0G2, T0G3, T1G0, T1G2, T1G3, T1G4, T2G0, T2G1, T2G3, T2G4,
  //   T3G0, T3G1, T3G2, T3G3, T0G4, T0G5, X, X, T1G4, T1G5, X, X, T2G4, T2G5,
  //   X, X, T3G4, T3G5, X, X]
  //
  // An Int32 tensor of size [1, 4, 2] and stride [8, 1, 4] can be represented
  // as an uint8 tensor of shape [1, 2, 4, 4] and stride [32, 16, 4, 1]. In
  // english, ignoring the Experts dimension, the original int32 tensor is
  // simply treated as two packed [4, 4] uint8 tensor (or two [4, 1] int32
  // tensor). The following strides setting reflects this change. Caveat: This
  // means that the G dimension is no longer contiguous. i.e. Note that to move
  // from G3 to G4, we need to jump along the packing dimension. The kernel
  // handles this case.

  stride_ys_e *= sizeof(int32_t);
  stride_ys_p = T * sizeof(int32_t);  // Packing dimension
  stride_ys_t = sizeof(int32_t);
  stride_ys_g = 1;

  LAUNCH_ON_H(uint8_t, stride_ys_e, stride_ys_t, stride_ys_g, stride_ys_p,
              true);

#endif
}
