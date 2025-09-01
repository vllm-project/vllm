#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include "core/math.hpp"
#include "../cuda_compat.h"
#include "dispatch_utils.h"

#include "quantization/fp8/common.cuh"

#include <cuda_fp8.h>

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
  return x * (1.f / (1.f + expf(-x)));
}

__device__ __forceinline__ float2 silu2(float2 x) {
  return make_float2(silu(x.x), silu(x.y));
}

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

template <typename T, typename U>
__device__ __forceinline__ void cp_async4(T* _smem_ptr, const U* _glob_ptr) {
  auto smem_ptr = reinterpret_cast<void*>(_smem_ptr);
  auto glob_ptr = reinterpret_cast<const void*>(_glob_ptr);
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

__device__ __forceinline__ void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <>
__device__ __forceinline__ void cp_async_wait<0>() {
  asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ float clip(float v, float mmin, float mmax) {
  return fminf(mmax, fmaxf(v, mmin));
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

template <typename scalar_t, uint32_t NUM_WARPS, typename Idx_t,
          int NUM_PARALLEL_TOKENS, bool USE_UE8M0, int GROUP_SIZE = 128,
          int NUM_STAGES = 3>
__global__ void silu_mul_fp8_quant_deep_gemm_kernel(
    const scalar_t* __restrict__ _input, __nv_fp8_e4m3* __restrict__ _y_q,
    float* __restrict__ _y_s, const uint32_t* __restrict__ counts,

    // sizes
    Idx_t H, Idx_t G,

    // strides (in elements)
    Idx_t stride_i_e, Idx_t stride_i_t, Idx_t stride_i_h, Idx_t stride_yq_e,
    Idx_t stride_yq_t, Idx_t stride_yq_h, Idx_t stride_ys_e, Idx_t stride_ys_t,
    Idx_t stride_ys_g, Idx_t stride_counts_e,

    // quant params
    float fp8_min, float fp8_max) {
  static constexpr float EPS = 1e-10;
  static constexpr uint32_t S_NUM_128 =
      2 * (GROUP_SIZE / 8) * NUM_WARPS * NUM_STAGES;
  static constexpr auto THREAD_COUNT = NUM_WARPS * WARP_SIZE;
  static constexpr int HALF_THREAD_COUNT = THREAD_COUNT / 2;
  static constexpr uint32_t S_NUM_64 = S_NUM_128 * 2;
  static constexpr uint32_t S_NUM_32 = S_NUM_64 * 2;
  __shared__ __int128_t __align__(16) s_buff_128[S_NUM_128];
  __shared__ Idx_t s_counts[1];

  const Idx_t tid = threadIdx.x;
  const Idx_t warp_id = tid / WARP_SIZE;
  const Idx_t lane_id = tid % WARP_SIZE;

  auto s_buff_compute_32 = reinterpret_cast<__nv_bfloat162*>(s_buff_128);

  // block handles one (expert e, group g)
  Idx_t pid = blockIdx.x;
  Idx_t e = pid / G;
  Idx_t g = pid % G;

  if (!tid) {
    s_counts[0] = counts[e * stride_counts_e];
  }

  const Idx_t stride_i_t_128 = stride_i_t / 8u;

  __syncthreads();
  const Idx_t n_tokens = s_counts[0];

  auto par_id = blockIdx.y;
  auto chunk_size = (n_tokens + NUM_PARALLEL_TOKENS - 1) / NUM_PARALLEL_TOKENS;
  auto n_tokens_lower = par_id * chunk_size;
  auto n_tokens_upper = min(n_tokens, (par_id + 1) * chunk_size);

  // base offsets (element-based)
  const Idx_t base_i = e * stride_i_e + NUM_WARPS * g * GROUP_SIZE * stride_i_h;
  const Idx_t base_ys = e * stride_ys_e + NUM_WARPS * g * stride_ys_g;
  const Idx_t base_yq =
      e * stride_yq_e + NUM_WARPS * g * GROUP_SIZE * stride_yq_h;

  Idx_t gate_off_128 = (base_i / 8u);
  auto input_128_ptr = reinterpret_cast<const __int128_t*>(_input);
  auto gate_128_ptr = input_128_ptr + gate_off_128 + (tid % 64) +
                      stride_i_t_128 * n_tokens_lower;
  auto up_128_ptr = gate_128_ptr + (H * stride_i_h) / 8u;

  auto y_s_ptr = _y_s + base_ys + warp_id + stride_ys_t * n_tokens_lower;

  auto y_q_ptr = _y_q + base_yq + warp_id * GROUP_SIZE + 2 * lane_id +
                 stride_yq_t * n_tokens_lower;

  Idx_t t_load = n_tokens_lower, load_stage_id = 0;
  auto s_buff_gate_load_128 = s_buff_128 + (tid % HALF_THREAD_COUNT);
  auto s_buff_up_load_128 = s_buff_gate_load_128 + S_NUM_128 / 2u;
  auto load_and_advance_y_pred = [&] {
    if (t_load < n_tokens_upper) {
      auto stage_offset =
          (load_stage_id % NUM_STAGES) * (NUM_WARPS * WARP_SIZE / 2);

      auto s_gate_stage_128_staged_ptr = s_buff_gate_load_128 + stage_offset;
      auto s_up_stage_128_staged_ptr = s_buff_up_load_128 + stage_offset;

      if (tid < HALF_THREAD_COUNT) {
        cp_async4(s_gate_stage_128_staged_ptr, gate_128_ptr);
        gate_128_ptr += stride_i_t_128;
      } else {
        cp_async4(s_up_stage_128_staged_ptr, up_128_ptr);
        up_128_ptr += stride_i_t_128;
      }
      ++t_load;
      ++load_stage_id;
    }
    cp_async_fence();
  };

#pragma unroll
  for (int i = 0; i < NUM_STAGES - 1; i++) {
    load_and_advance_y_pred();
  }

  auto s_gate_ptr = s_buff_compute_32 + warp_id * (GROUP_SIZE / 2) + lane_id;
  auto s_up_ptr = s_gate_ptr + S_NUM_32 / 2;

  Idx_t stage_id{};
  for (Idx_t t = n_tokens_lower; t < n_tokens_upper; ++t) {
    float y_max = EPS;
    float results[4];

    cp_async_wait<NUM_STAGES - 2>();
    __syncthreads();

    load_and_advance_y_pred();

    const auto compute_pipeline_offset =
        ((stage_id++) % NUM_STAGES) * (GROUP_SIZE / 2u) * NUM_WARPS;

    auto s_gate_compute = s_gate_ptr + compute_pipeline_offset;
    auto s_up_compute = s_up_ptr + compute_pipeline_offset;

#pragma unroll
    for (int i = 0; i < 2; i++) {
      float2 gate = silu2(__bfloat1622float2(*s_gate_compute));
      float2 upv = __bfloat1622float2(*s_up_compute);

      results[2 * i] = gate.x * upv.x;
      results[2 * i + 1] = gate.y * upv.y;

      y_max =
          fmaxf(y_max, fmaxf(fabsf(results[2 * i]), fabsf(results[2 * i + 1])));
      s_gate_compute += WARP_SIZE;
      s_up_compute += WARP_SIZE;
    }

    float y_s = warp_max(y_max) / fp8_max;

    if constexpr (USE_UE8M0) {
      y_s = exp2f(ceilf(log2f(y_s)));
    }

#pragma unroll
    for (Idx_t i = 0; i < 4; ++i) {
      results[i] = clip(results[i] / y_s, fp8_min, fp8_max);
    }

    auto local_y_q_ptr = reinterpret_cast<unsigned char*>(y_q_ptr);
    const auto r4 = reinterpret_cast<const float4*>(results);
    auto fp8x4 = __nv_fp8x4_e4m3(*r4);
    auto resultfp8x2 = reinterpret_cast<__nv_fp8x2_e4m3*>(&fp8x4.__x);

#pragma unroll
    for (Idx_t i = 0; i < 2; ++i) {
      auto res_u8 = reinterpret_cast<unsigned char*>(&resultfp8x2[i].__x);
      local_y_q_ptr[0] = res_u8[0];
      local_y_q_ptr[1] = res_u8[1];
      local_y_q_ptr += 2 * WARP_SIZE;
    }

    y_q_ptr += stride_yq_t;

    if (lane_id == 0) {
      *y_s_ptr = y_s;
      y_s_ptr += stride_ys_t;
    }
  }
}

template <typename scalar_t, uint32_t NUM_WARPS, typename Idx_t,
          int NUM_PARALLEL_TOKENS, bool USE_UE8M0, int GROUP_SIZE = 128,
          int NUM_STAGES = 3>
__global__ void __silu_mul_fp8_quant_deep_gemm_kernel(
    const scalar_t* __restrict__ _input, __nv_fp8_e4m3* __restrict__ _y_q,
    float* __restrict__ _y_s, const uint32_t* __restrict__ counts,

    // sizes
    Idx_t H, Idx_t G,

    // strides (in elements)
    Idx_t stride_i_e, Idx_t stride_i_t, Idx_t stride_i_h, Idx_t stride_yq_e,
    Idx_t stride_yq_t, Idx_t stride_yq_h, Idx_t stride_ys_e, Idx_t stride_ys_t,
    Idx_t stride_ys_g, Idx_t stride_counts_e,

    // quant params
    float fp8_min, float fp8_max) {
  static constexpr float EPS = 1e-10;
  static constexpr uint32_t S_NUM_128 =
      2 * (GROUP_SIZE / 8) * NUM_WARPS * NUM_STAGES;
  static constexpr auto THREAD_COUNT = NUM_WARPS * WARP_SIZE;
  static constexpr int HALF_THREAD_COUNT = THREAD_COUNT / 2;
  static constexpr uint32_t S_NUM_64 = S_NUM_128 * 2;
  static constexpr uint32_t S_NUM_32 = S_NUM_64 * 2;
  __shared__ __int128_t __align__(16) s_buff_128[S_NUM_128];
  __shared__ Idx_t s_counts[1];

  const Idx_t tid = threadIdx.x;
  const Idx_t warp_id = tid / WARP_SIZE;
  const Idx_t lane_id = tid % WARP_SIZE;

  auto s_buff_compute_32 = reinterpret_cast<__nv_bfloat162*>(s_buff_128);

  // block handles one (expert e, group g)
  Idx_t pid = blockIdx.x;
  Idx_t e = pid / G;
  Idx_t g = pid % G;

  if (!tid) {
    s_counts[0] = counts[e * stride_counts_e];
  }

  const Idx_t stride_i_t_128 = stride_i_t / 8u;

  __syncthreads();
  const Idx_t n_tokens = s_counts[0];

  auto par_id = blockIdx.y;
  auto chunk_size = (n_tokens + NUM_PARALLEL_TOKENS - 1) / NUM_PARALLEL_TOKENS;
  auto n_tokens_lower = par_id * chunk_size;
  auto n_tokens_upper = min(n_tokens, (par_id + 1) * chunk_size);

  // base offsets (element-based)
  const Idx_t base_i = e * stride_i_e + NUM_WARPS * g * GROUP_SIZE * stride_i_h;
  const Idx_t base_ys = e * stride_ys_e + NUM_WARPS * g * stride_ys_g;
  const Idx_t base_yq =
      e * stride_yq_e + NUM_WARPS * g * GROUP_SIZE * stride_yq_h;

  Idx_t gate_off_128 = (base_i / 8u);
  auto input_128_ptr = reinterpret_cast<const __int128_t*>(_input);
  auto gate_128_ptr = input_128_ptr + gate_off_128 + (tid % 64) +
                      stride_i_t_128 * n_tokens_lower;
  auto up_128_ptr = gate_128_ptr + (H * stride_i_h) / 8u;

  auto y_s_ptr = _y_s + base_ys + warp_id + stride_ys_t * n_tokens_lower;

  auto y_q_ptr = _y_q + base_yq + warp_id * GROUP_SIZE + 2 * lane_id +
                 stride_yq_t * n_tokens_lower;

  Idx_t t_load = n_tokens_lower, load_stage_id = 0;
  auto s_buff_gate_load_128 = s_buff_128 + (tid % HALF_THREAD_COUNT);
  auto s_buff_up_load_128 = s_buff_gate_load_128 + S_NUM_128 / 2u;
  auto load_and_advance_y_pred = [&] {
    if (t_load < n_tokens_upper) {
      auto stage_offset =
          (load_stage_id % NUM_STAGES) * (NUM_WARPS * WARP_SIZE / 2);

      auto s_gate_stage_128_staged_ptr = s_buff_gate_load_128 + stage_offset;
      auto s_up_stage_128_staged_ptr = s_buff_up_load_128 + stage_offset;

      if (tid < HALF_THREAD_COUNT) {
        cp_async4(s_gate_stage_128_staged_ptr, gate_128_ptr);
        gate_128_ptr += stride_i_t_128;
      } else {
        cp_async4(s_up_stage_128_staged_ptr, up_128_ptr);
        up_128_ptr += stride_i_t_128;
      }
      ++t_load;
      ++load_stage_id;
    }
    cp_async_fence();
  };

#pragma unroll
  for (int i = 0; i < NUM_STAGES - 1; i++) {
    load_and_advance_y_pred();
  }

  auto s_gate_ptr = s_buff_compute_32 + warp_id * (GROUP_SIZE / 2) + lane_id;
  auto s_up_ptr = s_gate_ptr + S_NUM_32 / 2;
  using bf16x2 = __nv_bfloat162;
  using bf16 = __nv_bfloat16;

  Idx_t stage_id{};
  for (Idx_t t = n_tokens_lower; t < n_tokens_upper; ++t) {
    bf16 y_max = EPS;
    bf16x2 results[2];

    cp_async_wait<NUM_STAGES - 2>();
    __syncthreads();

    load_and_advance_y_pred();

    const auto compute_pipeline_offset =
        ((stage_id++) % NUM_STAGES) * (GROUP_SIZE / 2u) * NUM_WARPS;

    auto s_gate_compute = s_gate_ptr + compute_pipeline_offset;
    auto s_up_compute = s_up_ptr + compute_pipeline_offset;

#pragma unroll
    for (int i = 0; i < 2; i++) {
      bf16x2 gate = __float22bfloat162_rn(__bfloat1622float2(*s_gate_compute));
      bf16x2 upv = *s_up_compute;

      results[i] = __hmul2(gate, upv);

      s_gate_compute += WARP_SIZE;
      s_up_compute += WARP_SIZE;
    }

    auto hmx2 = __hmax2(results[0], results[1]);
    y_max = __hmax(hmx2.x, hmx2.y);

    auto y_s = warp_max(y_max) / bf16(fp8_max);

    const auto fp8min2 = make_bfloat162(fp8_min, fp8_min);
    const auto fp8max2 = make_bfloat162(fp8_max, fp8_max);

    auto y_s2 = make_bfloat162(y_s, y_s);
#pragma unroll
    for (Idx_t i = 0; i < 2; ++i) {
      results[i] /= y_s2;
      results[i] = clip(results[i], fp8min2, fp8max2);
    }

    auto local_y_q_ptr = reinterpret_cast<unsigned char*>(y_q_ptr);
#pragma unroll
    for (Idx_t i = 0; i < 2; ++i) {
      auto resultfp8x2 = __nv_fp8x2_e4m3(results[i]);
      auto res_u8 = reinterpret_cast<unsigned char*>(&resultfp8x2.__x);
      local_y_q_ptr[0] = res_u8[0];
      local_y_q_ptr[1] = res_u8[1];
      local_y_q_ptr += 2 * WARP_SIZE;
    }

    y_q_ptr += stride_yq_t;

    if (lane_id == 0) {
      *y_s_ptr = y_s;
      y_s_ptr += stride_ys_t;
    }
  }
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

void silu_mul_fp8_quant_deep_gemm_cuda(
    const at::Tensor& input,   // (E, T, 2*H)
    const at::Tensor& counts,  // (E)
    at::Tensor& y_q,           // (E, T, H) [OUT]
    at::Tensor& y_s,           // (E, T, H//group_size) [OUT]
    int64_t group_size, double eps, double fp8_min, double fp8_max,
    bool use_ue8m0) {
  static constexpr int NUM_WARPS = 4;

  using Idx_t = uint32_t;

  Idx_t E = input.size(0);
  Idx_t T = input.size(1);
  Idx_t H = input.size(2) / 2;
  Idx_t G = H / (group_size * NUM_WARPS);
  Idx_t stride_i_e = input.stride(0);
  Idx_t stride_i_t = input.stride(1);
  Idx_t stride_i_h = input.stride(2);
  Idx_t stride_yq_e = y_q.stride(0);
  Idx_t stride_yq_t = y_q.stride(1);
  Idx_t stride_yq_h = y_q.stride(2);
  Idx_t stride_ys_e = y_s.stride(0);
  Idx_t stride_ys_t = y_s.stride(1);
  Idx_t stride_ys_g = y_s.stride(2);

  int stride_counts_e = counts.stride(0);

  static constexpr int NUM_PARALLEL_TOKENS = 16;
  dim3 grid(E * G, NUM_PARALLEL_TOKENS);
  dim3 block(NUM_WARPS * 32);

  if (use_ue8m0) {
    vllm::silu_mul_fp8_quant_deep_gemm_kernel<__nv_bfloat16, NUM_WARPS, Idx_t,
                                              NUM_PARALLEL_TOKENS, true>
        <<<grid, block>>>(
            reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(y_q.data_ptr<at::Float8_e4m3fn>()),
            y_s.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(counts.data_ptr<int>()), H, G,
            stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t,
            stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g, stride_counts_e,
            static_cast<float>(fp8_min), static_cast<float>(fp8_max));
  } else {
    vllm::silu_mul_fp8_quant_deep_gemm_kernel<__nv_bfloat16, NUM_WARPS, Idx_t,
                                              NUM_PARALLEL_TOKENS, false>
        <<<grid, block>>>(
            reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(y_q.data_ptr<at::Float8_e4m3fn>()),
            y_s.data_ptr<float>(),
            reinterpret_cast<uint32_t*>(counts.data_ptr<int>()), H, G,
            stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t,
            stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g, stride_counts_e,
            static_cast<float>(fp8_min), static_cast<float>(fp8_max));
  }
}
