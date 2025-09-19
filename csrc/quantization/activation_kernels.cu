#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include "core/math.hpp"
#include "../cuda_compat.h"
#include "dispatch_utils.h"

#include "quantization/fp8/common.cuh"

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

typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
typedef __hip_fp8x4_e4m3 __nv_fp8x4_e4m3;
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
  return (__fdividef(x, (1.f + expf(-x))));
}

__device__ __forceinline__ float2 silu2(float2 x) {
  return make_float2(silu(x.x), silu(x.y));
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
#ifndef USE_ROCM
template <typename fp8_type, int32_t NUM_WARPS, typename Idx_t,
          int NUM_PARALLEL_TOKENS, bool USE_UE8M0, int GROUP_SIZE = 128,
          int NUM_STAGES = 3>
__global__ void silu_mul_fp8_quant_deep_gemm_kernel(
    const __nv_bfloat16* __restrict__ _input, fp8_type* __restrict__ _y_q,
    float* __restrict__ _y_s, const int32_t* __restrict__ counts,

    // sizes
    int H, int G,

    // strides (in elements)
    Idx_t stride_i_e, Idx_t stride_i_t, Idx_t stride_i_h, Idx_t stride_yq_e,
    Idx_t stride_yq_t, Idx_t stride_yq_h, Idx_t stride_ys_e, Idx_t stride_ys_t,
    Idx_t stride_ys_g, Idx_t stride_counts_e) {
  static constexpr __nv_bfloat16 fp8_min = get_fp8_min<fp8_type>();
  static constexpr __nv_bfloat16 fp8_max = get_fp8_max<fp8_type>();
  // We assign EPS with its 16-bit unsigned counterpart to allow constexpr.
  static constexpr __nv_bfloat16 EPS = (__nv_bfloat16_raw{.x = 11996});

  // We pack 8 16-bit bfloat16 values into a 128-bit __int128_t.
  static constexpr int32_t BFLOAT16_PER_GROUP = 8;

  // We split the shared memory in half, corresponding to gate and up matrices:
  // [...gate_i, ...up_i]  where 0 <= i < stages.
  static constexpr int32_t S_NUM_128 =
      2u * (GROUP_SIZE / BFLOAT16_PER_GROUP) * NUM_WARPS * NUM_STAGES;
  static constexpr auto THREAD_COUNT = NUM_WARPS * WARP_SIZE;
  static constexpr int HALF_THREAD_COUNT = THREAD_COUNT / 2;
  static constexpr int32_t S_NUM_64 = S_NUM_128 * 2;
  __shared__ __int128_t __align__(16) s_buff_128[S_NUM_128];

  const int32_t tid = threadIdx.x;
  const int32_t warp_id = tid / WARP_SIZE;
  const int32_t lane_id = tid % WARP_SIZE;

  auto s_buff_compute_32 = reinterpret_cast<__nv_bfloat162*>(s_buff_128);

  // block handles one (expert e, group g)
  int32_t pid = blockIdx.x;
  int32_t e = pid / G;
  int32_t g = pid % G;

  const int32_t n_tokens = counts[e * stride_counts_e];

  if (!n_tokens) {
    return;  // Exit ASAP.
  }

  const Idx_t stride_i_t_128 = stride_i_t / 8u;

  int32_t n_tokens_lower, n_tokens_upper;

  // Each block i iterates over tokens of a slice of n_tokens =
  // expert_counts[i], with the size of chunk being
  // (n_tokens / NUM_PARALLEL_TOKENS) + residual, instead of
  // updiv(n_tokens, NUM_PARALLEL_TOKENS) for better scheduling.
  if (n_tokens < NUM_PARALLEL_TOKENS && blockIdx.y < n_tokens) {
    // Specialize this, but can be likely fused.
    if (blockIdx.y >= NUM_PARALLEL_TOKENS) {
      return;
    }
    n_tokens_lower = blockIdx.y;
    n_tokens_upper = blockIdx.y + 1;
  } else {
    auto chunk_size = n_tokens / NUM_PARALLEL_TOKENS;
    auto residual = n_tokens - chunk_size * NUM_PARALLEL_TOKENS;
    auto calc_id = [&](int32_t id) {
      if (id < residual) {
        return min(n_tokens, id * (chunk_size + 1));
      } else {
        return min(n_tokens, id * chunk_size + residual);
      }
    };
    n_tokens_lower = calc_id(blockIdx.y);
    n_tokens_upper = calc_id(blockIdx.y + 1);
  }

  if (n_tokens_lower >= n_tokens_upper) {
    return;
  }

  // We do calculations here, using constexpr wherever possible.
  const Idx_t base_i = e * stride_i_e + NUM_WARPS * g * GROUP_SIZE * stride_i_h;
  const Idx_t base_ys = e * stride_ys_e + NUM_WARPS * g * stride_ys_g;
  const Idx_t base_yq =
      e * stride_yq_e + NUM_WARPS * g * GROUP_SIZE * stride_yq_h;
  Idx_t gate_off_128 = (base_i / static_cast<Idx_t>(8u));
  auto input_128_ptr = reinterpret_cast<const __int128_t*>(_input);
  auto gate_128_ptr = input_128_ptr + gate_off_128 + (tid % HALF_THREAD_COUNT) +
                      stride_i_t_128 * n_tokens_lower;
  auto up_128_ptr = gate_128_ptr + (H * stride_i_h) / 8u;
  auto y_s_ptr =
      _y_s + base_ys + warp_id * stride_ys_g + n_tokens_lower * stride_ys_t;
  auto y_q_ptr = _y_q + base_yq + warp_id * GROUP_SIZE +
                 stride_yq_t * n_tokens_lower + 4 * lane_id;
  int32_t t_load = n_tokens_lower, load_stage_id = 0;
  auto s_buff_gate_load_128 = s_buff_128 + (tid % HALF_THREAD_COUNT);
  auto s_buff_up_load_128 = s_buff_gate_load_128 + S_NUM_128 / 2u;
  int32_t stage_offset{};

  static constexpr int32_t LOAD_STAGE_SIZE = (NUM_WARPS * WARP_SIZE / 2);
  static constexpr int32_t LOAD_STAGE_MOD =
      NUM_STAGES * (NUM_WARPS * WARP_SIZE / 2);

  // Two halves of all threads in a block conduct global loads for gate and up,
  // repsectively.
  auto load_and_advance_y_pred = [&] {
    if (t_load < n_tokens_upper) {
      auto s_gate_stage_128_staged_ptr = s_buff_gate_load_128 + stage_offset;
      auto s_up_stage_128_staged_ptr = s_buff_up_load_128 + stage_offset;

      // It is very important that LOAD_STAGE_SIZE is constexpr to avoid
      // unnecessary ALU ops.
      stage_offset += LOAD_STAGE_SIZE;
      stage_offset %= LOAD_STAGE_MOD;

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
    // We fence even if there is nothing to load to simplify pipelining.
    cp_async_fence();
  };

  #pragma unroll
  for (int i = 0; i < NUM_STAGES - 1; i++) {
    load_and_advance_y_pred();
  }

  __int64_t* s_gate_ptr = reinterpret_cast<__int64_t*>(
                              s_buff_compute_32 + warp_id * (GROUP_SIZE / 2)) +
                          lane_id;
  __int64_t* s_up_ptr = s_gate_ptr + S_NUM_64 / 2;

  static constexpr int32_t STAGE_SIZE = (GROUP_SIZE * NUM_WARPS) / 4u;
  static constexpr int32_t STAGE_MOD = STAGE_SIZE * NUM_STAGES;

  int32_t compute_pipeline_offset_64 = 0;

  for (int32_t t = n_tokens_lower; t < n_tokens_upper; ++t) {
    __nv_bfloat162 results_bf162[2];

    cp_async_wait<NUM_STAGES - 2>();
    __syncthreads();

    // We double-buffer pipelined loads so that the next load will
    // concurrently run with compute without overwrites.
    load_and_advance_y_pred();

    auto s_gate_compute_64 = s_gate_ptr + compute_pipeline_offset_64;
    auto s_up_compute_64 = s_up_ptr + compute_pipeline_offset_64;

    // STAGE_SIZE must also be constexpr!
    compute_pipeline_offset_64 += STAGE_SIZE;
    compute_pipeline_offset_64 %= STAGE_MOD;

    // Each thread loads (gate/up) 2X 4X bfloat16 values into registers.
    __int64_t gate64 = *s_gate_compute_64;
    __nv_bfloat162* s_gate_compute_32 =
        reinterpret_cast<__nv_bfloat162*>(&gate64);

    __int64_t up64 = *s_up_compute_64;
    __nv_bfloat162* s_up_compute_32 = reinterpret_cast<__nv_bfloat162*>(&up64);

  #pragma unroll
    for (int i = 0; i < 2; i++) {
      // For silu, we make sure that div is emitted.
      float2 gate = silu2(__bfloat1622float2(s_gate_compute_32[i]));
      results_bf162[i] = __float22bfloat162_rn(gate);
    }

  #pragma unroll
    for (int i = 0; i < 2; i++) {
      results_bf162[i] = __hmul2(results_bf162[i], s_up_compute_32[i]);
    }

    auto _y_max2 =
        __hmax2(__habs2(results_bf162[0]), __habs2(results_bf162[1]));

    __nv_bfloat16 y_max_bf16 = __hmax(EPS, __hmax(_y_max2.x, _y_max2.y));

    // An entire group is assigned to a single warp, so a simple warp reduce
    // is used.
    __nv_bfloat16 y_s = warp_max(y_max_bf16) / fp8_max;

    if constexpr (USE_UE8M0) {
      y_s = hexp2(hceil(hlog2(y_s)));
    }

    auto inv_y = __float2bfloat16_rn(1.f) / y_s;

    auto y_s2 = make_bfloat162(inv_y, inv_y);

  #pragma unroll
    for (int32_t i = 0; i < 2; ++i) {
      results_bf162[i] =
          clip(__hmul2(results_bf162[i], y_s2), __bfloat162bfloat162(fp8_min),
               __bfloat162bfloat162(fp8_max));
    }

    auto fp8x4 = __nv_fp8x4_e4m3(results_bf162[0], results_bf162[1]);
    *reinterpret_cast<__nv_fp8x4_e4m3*>(y_q_ptr) = fp8x4;
    y_q_ptr += stride_yq_t;

    if (lane_id == 0) {
      *y_s_ptr = y_s;
      y_s_ptr += stride_ys_t;
    }
  }
}
#endif

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
    int64_t group_size, bool use_ue8m0, int64_t num_parallel_tokens) {
#ifndef USE_ROCM
  // This kernel relies heavily on cp.async and fp8 support.
  // This kernel currently only supports H % 128 == 0 and assumes a
  // fixed GROUP_SIZE of 128.
  TORCH_CHECK(input.dtype() == torch::kBFloat16);
  TORCH_CHECK(y_q.dtype() == torch::kFloat8_e4m3fn ||
              y_q.dtype() == torch::kFloat8_e4m3fnuz);
  TORCH_CHECK(y_s.dtype() == torch::kFloat32);
  TORCH_CHECK(input.size(-1) % 256 == 0);

  // Check that num_parallel_tokens is of power of 2 and between 1 and 64.
  TORCH_CHECK(1 <= num_parallel_tokens && num_parallel_tokens <= 64);
  TORCH_CHECK(!(num_parallel_tokens & (num_parallel_tokens - 1)));

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
  Idx_t stride_ys_e = y_s.stride(0);
  Idx_t stride_ys_t = y_s.stride(1);
  Idx_t stride_ys_g = y_s.stride(2);

  Idx_t stride_counts_e = counts.stride(0);

  static constexpr int GROUP_SIZE = 128;

  #define KERNEL_FN                                                         \
    if (use_ue8m0) {                                                        \
      vllm::silu_mul_fp8_quant_deep_gemm_kernel<fp8_t, NUM_WARPS, Idx_t,    \
                                                NUM_PARALLEL_TOKENS, true>  \
          <<<grid, block, 0, stream>>>(                                     \
              reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),           \
              (fp8_t*)y_q.data_ptr(), y_s.data_ptr<float>(),                \
              reinterpret_cast<int32_t*>(counts.data_ptr<int>()), H, G,     \
              stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t, \
              stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g,           \
              stride_counts_e);                                             \
    } else {                                                                \
      vllm::silu_mul_fp8_quant_deep_gemm_kernel<fp8_t, NUM_WARPS, Idx_t,    \
                                                NUM_PARALLEL_TOKENS, false> \
          <<<grid, block, 0, stream>>>(                                     \
              reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),           \
              (fp8_t*)y_q.data_ptr(), y_s.data_ptr<float>(),                \
              reinterpret_cast<int32_t*>(counts.data_ptr<int>()), H, G,     \
              stride_i_e, stride_i_t, stride_i_h, stride_yq_e, stride_yq_t, \
              stride_yq_h, stride_ys_e, stride_ys_t, stride_ys_g,           \
              stride_counts_e);                                             \
    }

  #define KERNEL_CALL_H                                       \
    if (H % (4 * GROUP_SIZE) == 0) {                          \
      static constexpr int NUM_WARPS = 4;                     \
      populate_launch_params(NUM_WARPS, NUM_PARALLEL_TOKENS); \
      KERNEL_FN                                               \
    } else {                                                  \
      static constexpr int NUM_WARPS = 1;                     \
      populate_launch_params(NUM_WARPS, NUM_PARALLEL_TOKENS); \
      KERNEL_FN                                               \
    }

  #define KERNEL_CALL_TOP_LEVEL                      \
    if (num_parallel_tokens == 1) {                  \
      static constexpr int NUM_PARALLEL_TOKENS = 1;  \
      KERNEL_CALL_H                                  \
    } else if (num_parallel_tokens == 2) {           \
      static constexpr int NUM_PARALLEL_TOKENS = 2;  \
      KERNEL_CALL_H                                  \
    } else if (num_parallel_tokens == 4) {           \
      static constexpr int NUM_PARALLEL_TOKENS = 4;  \
      KERNEL_CALL_H                                  \
    } else if (num_parallel_tokens == 8) {           \
      static constexpr int NUM_PARALLEL_TOKENS = 8;  \
      KERNEL_CALL_H                                  \
    } else if (num_parallel_tokens == 16) {          \
      static constexpr int NUM_PARALLEL_TOKENS = 16; \
      KERNEL_CALL_H                                  \
    } else if (num_parallel_tokens == 32) {          \
      static constexpr int NUM_PARALLEL_TOKENS = 32; \
      KERNEL_CALL_H                                  \
    } else if (num_parallel_tokens == 64) {          \
      static constexpr int NUM_PARALLEL_TOKENS = 64; \
      KERNEL_CALL_H                                  \
    }

  Idx_t G;
  dim3 block, grid;
  auto populate_launch_params = [&](int num_warps, int _num_parallel_tokens) {
    G = H / Idx_t(group_size * num_warps);
    grid = dim3(E * G, _num_parallel_tokens);
    block = dim3(num_warps * WARP_SIZE);
  };

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  VLLM_DISPATCH_FP8_TYPES(y_q.scalar_type(),
                          "silu_mul_fp8_quant_deep_gemm_kernel",
                          [&] { KERNEL_CALL_TOP_LEVEL });

#endif
}
