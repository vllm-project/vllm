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
  return x / (1.f + __expf(-x));
}

// warp-wide max reduction
__device__ __forceinline__ float warp_max(float v) {
  unsigned mask = 0xffffffffu;
  // shuffle-down tree
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(mask, v, offset);
    v = fmaxf(v, other);
  }
  return v;
}

__device__ inline void cp_async1(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 4;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.ca.shared.global [%0], [%1], %2;\n"
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

template <typename scalar_t, uint32_t NUM_WARPS, typename Idx_t,
          int GROUP_SIZE = 128, int NUM_STAGES = 1>
__global__ void silu_mul_fp8_quant_deep_gemm_kernel(
    const scalar_t* __restrict__ _input,  // (E, T, 2*H), element strides
    __nv_fp8_e4m3* __restrict__ _y_q,     // (E, T, H)
    float* __restrict__ _y_s,             // (E, T, H//group_size)
    const uint32_t* __restrict__ counts,  // (E)

    // sizes
    Idx_t E, Idx_t T, Idx_t H, Idx_t G,

    // strides (in elements)
    Idx_t stride_i_e, Idx_t stride_i_t, Idx_t stride_i_h, Idx_t stride_yq_e,
    Idx_t stride_yq_t, Idx_t stride_yq_h, Idx_t stride_ys_e, Idx_t stride_ys_t,
    Idx_t stride_ys_g, Idx_t stride_counts_e,

    // quant params
    float eps, float fp8_min, float fp8_max, bool use_ue8m0) {
  __shared__ __nv_bfloat162 s_buff[GROUP_SIZE * NUM_STAGES];
  __shared__ float s_warp_max[NUM_WARPS];  // size = numWarpsPerBlock
  // block handles one (expert e, group g)
  Idx_t pid = blockIdx.x;
  Idx_t e = pid / G;
  Idx_t g = pid % G;
  if (e >= E) return;

  const Idx_t tid = threadIdx.x;
  const Idx_t warp_id = tid / WARP_SIZE;
  const Idx_t lane_id = tid % WARP_SIZE;
  const Idx_t num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  // how many tokens for this expert

  __shared__ Idx_t s_count;

  if (!tid) {
    s_count = counts[e * stride_counts_e];
  }

  const Idx_t stride_i_t_half = stride_i_t >> 1u;

  // base offsets (element-based)
  const Idx_t base_i = e * stride_i_e + g * GROUP_SIZE * stride_i_h;
  const Idx_t base_yq = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h;
  const Idx_t base_ys = e * stride_ys_e + g * stride_ys_g;

  __syncthreads();

  Idx_t gate_off = ((base_i) >> 1u) + tid;
  Idx_t up_off = ((base_i + H * stride_i_h) >> 1u) + (tid - 64u);
  Idx_t yq_off = base_yq + tid * stride_yq_h;

  auto gate_ptr =
      reinterpret_cast<const __nv_bfloat162* __restrict__>(_input) + gate_off;
  auto up_ptr =
      reinterpret_cast<const __nv_bfloat162* __restrict__>(_input) + up_off;
  auto y_q_ptr = _y_q + yq_off;
  auto y_s_ptr = _y_s + base_ys;

  auto s_ptr = reinterpret_cast<void*>(s_buff + tid);

  const Idx_t n_tokens = s_count;

  Idx_t t_load = 0;
  auto load_and_advance_y_pred = [&]() {
    if (t_load < n_tokens) {
      if (tid < 64) {
        cp_async1(s_ptr, reinterpret_cast<const void*>(gate_ptr));
        gate_ptr += stride_i_t_half;
      } else {
        cp_async1(s_ptr, reinterpret_cast<const void*>(up_ptr));
        up_ptr += stride_i_t_half;
      }
      t_load++;
    }
    cp_async_fence();
  };

  load_and_advance_y_pred();

  auto s_gate_ptr = reinterpret_cast<__nv_bfloat16*>(s_buff) + tid;
  auto s_up_ptr = reinterpret_cast<__nv_bfloat16*>(s_buff) + GROUP_SIZE + tid;

  for (Idx_t t = 0; t < n_tokens; ++t) {
    float gate, upv, y = -std::numeric_limits<float>::max();

    cp_async_wait<0>();
    __syncthreads();

    gate = __bfloat162float(*s_gate_ptr);
    gate = silu(gate);
    upv = __bfloat162float(*s_up_ptr);
    y = gate * upv;

    __syncthreads();

    load_and_advance_y_pred();

    float v = fabsf(y);
    v = warp_max(v);

    if (lane_id == 0) s_warp_max[warp_id] = v;

    __syncthreads();

    float absmax = 0.f;
    if (warp_id == 0) {
      float m = (lane_id < num_warps) ? s_warp_max[lane_id] : 0.f;
      m = warp_max(m);
      if (lane_id == 0) s_warp_max[0] = m;
    }

    __syncthreads();
    absmax = s_warp_max[0];
    absmax = fmaxf(absmax, eps);

    float scale = absmax / fp8_max;

    if (use_ue8m0) {
      scale = exp2f(ceilf(log2f(scale)));
    }

    float q = fminf(fmaxf(y / scale, fp8_min), fp8_max);

    __syncthreads();

    // Keep writes here - we don't need to wait for them to resolve.
    *y_q_ptr = __nv_fp8_e4m3(q);
    y_q_ptr += stride_yq_t;
    if (tid == 0) {
      *y_s_ptr = scale;
    }
    y_s_ptr += stride_ys_t;
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
  Idx_t G = H / group_size;
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

  dim3 grid(E * G);
  dim3 block(NUM_WARPS * 32);
  vllm::silu_mul_fp8_quant_deep_gemm_kernel<__nv_bfloat16, NUM_WARPS, Idx_t>
      <<<grid, block>>>(
          reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
          reinterpret_cast<__nv_fp8_e4m3*>(y_q.data_ptr<at::Float8_e4m3fn>()),
          y_s.data_ptr<float>(),
          reinterpret_cast<uint32_t*>(counts.data_ptr<int>()), E, T, H, G,
          stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t,
          stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g, stride_counts_e,
          eps, fp8_min, fp8_max, use_ue8m0);
}
