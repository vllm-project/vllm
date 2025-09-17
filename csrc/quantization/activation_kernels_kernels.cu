#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace vllm {

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

__device__ __forceinline__ float silu(float x) {
  return x * (1.f / (1.f + expf(-x)));
}

__device__ __forceinline__ float2 silu2(float2 x) {
  return make_float2(silu(x.x), silu(x.y));
}

__device__ __forceinline__ float warp_max(float v) {
  static constexpr unsigned FULL_MASK = 0xffffffffu;
  for (int offset = 1; offset < 32; offset *= 2) {
    v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, offset));
  }
  return v;
}

__device__ __forceinline__ __nv_bfloat16 warp_max(__nv_bfloat16 v) {
  static constexpr unsigned FULL_MASK = 0xffffffffu;
  for (int offset = 1; offset < 32; offset *= 2) {
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

// We use the following values for fp8 min/max:
//  __nv_fp8_e4m3 = (-448, +448)
//  __nv_fp8_e5m2 = (-57344.0, 57344.0)
//  __nv_fp8_e8m0 = (5.877471754111438e-39, 1.7014118346046923e+38)
template <class T>
constexpr __nv_bfloat16 get_fp8_max() {
  return __nv_bfloat16(__nv_bfloat16_raw{.x = 17376});
}
template <class T>
constexpr __nv_bfloat16 get_fp8_min() {
  return __nv_bfloat16(__nv_bfloat16_raw{.x = 50144});
}

template <typename fp8_type, uint32_t NUM_WARPS, typename Idx_t,
          int NUM_PARALLEL_TOKENS, bool USE_UE8M0, int GROUP_SIZE = 128,
          int NUM_STAGES = 3>
__global__ void silu_mul_fp8_quant_deep_gemm_kernel(
    const __nv_bfloat16* __restrict__ _input, fp8_type* __restrict__ _y_q,
    float* __restrict__ _y_s, const uint32_t* __restrict__ counts,

    // sizes
    Idx_t H, Idx_t G,

    // strides (in elements)
    Idx_t stride_i_e, Idx_t stride_i_t, Idx_t stride_i_h, Idx_t stride_yq_e,
    Idx_t stride_yq_t, Idx_t stride_yq_h, Idx_t stride_ys_e, Idx_t stride_ys_t,
    Idx_t stride_ys_g, Idx_t stride_counts_e) {
  static constexpr __nv_bfloat16 fp8_min = get_fp8_min<fp8_type>();
  static constexpr __nv_bfloat16 fp8_max = get_fp8_max<fp8_type>();
  static constexpr int WARP_SIZE = 32;

  // Same for EPS = 1e-10
  static constexpr __nv_bfloat16 EPS = (__nv_bfloat16_raw{.x = 11996});
  static constexpr uint32_t S_NUM_128 =
      2u * (GROUP_SIZE / 8u) * NUM_WARPS * NUM_STAGES;
  static constexpr auto THREAD_COUNT = NUM_WARPS * WARP_SIZE;
  static constexpr int HALF_THREAD_COUNT = THREAD_COUNT / 2;
  static constexpr uint32_t S_NUM_64 = S_NUM_128 * 2;
  __shared__ __int128_t __align__(16) s_buff_128[S_NUM_128];

  const Idx_t tid = threadIdx.x;
  const Idx_t warp_id = tid / WARP_SIZE;
  const Idx_t lane_id = tid % WARP_SIZE;

  auto s_buff_compute_32 = reinterpret_cast<__nv_bfloat162*>(s_buff_128);

  // block handles one (expert e, group g)
  Idx_t pid = blockIdx.x;
  Idx_t e = pid / G;
  Idx_t g = pid % G;

  const Idx_t n_tokens = counts[e * stride_counts_e];

  if (!n_tokens) {
    return;  // Exit ASAP.
  }

  const Idx_t stride_i_t_128 = stride_i_t / 8u;

  Idx_t n_tokens_lower, n_tokens_upper;

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
    auto calc_id = [&](Idx_t id) {
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

  // base offsets (element-based)
  const Idx_t base_i = e * stride_i_e + NUM_WARPS * g * GROUP_SIZE * stride_i_h;
  const Idx_t base_ys = e * stride_ys_e + NUM_WARPS * g * stride_ys_g;
  const Idx_t base_yq =
      e * stride_yq_e + NUM_WARPS * g * GROUP_SIZE * stride_yq_h;

  Idx_t gate_off_128 = (base_i / 8u);
  auto input_128_ptr = reinterpret_cast<const __int128_t*>(_input);
  auto gate_128_ptr = input_128_ptr + gate_off_128 + (tid % HALF_THREAD_COUNT) +
                      stride_i_t_128 * n_tokens_lower;
  auto up_128_ptr = gate_128_ptr + (H * stride_i_h) / 8u;

  auto y_s_ptr =
      _y_s + base_ys + warp_id * stride_ys_g + n_tokens_lower * stride_ys_t;

  auto y_q_ptr = _y_q + base_yq + warp_id * GROUP_SIZE +
                 stride_yq_t * n_tokens_lower + 4 * lane_id;

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

  __int64_t* s_gate_ptr = reinterpret_cast<__int64_t*>(
                              s_buff_compute_32 + warp_id * (GROUP_SIZE / 2)) +
                          lane_id;
  __int64_t* s_up_ptr = s_gate_ptr + S_NUM_64 / 2;

  Idx_t stage_id{};
  for (Idx_t t = n_tokens_lower; t < n_tokens_upper; ++t) {
    __nv_bfloat16 y_max_bf16 = EPS;
    __nv_bfloat162 results_bf162[2];

    cp_async_wait<NUM_STAGES - 2>();
    __syncthreads();

    load_and_advance_y_pred();

    const auto compute_pipeline_offset_64 =
        (((stage_id++) % NUM_STAGES) * (GROUP_SIZE / 2u) * NUM_WARPS) / 2u;
    auto s_gate_compute_64 = s_gate_ptr + compute_pipeline_offset_64;
    auto s_up_compute_64 = s_up_ptr + compute_pipeline_offset_64;
    __int64_t gate64 = *s_gate_compute_64;
    __nv_bfloat162* s_gate_compute_32 =
        reinterpret_cast<__nv_bfloat162*>(&gate64);

    __int64_t up64 = *s_up_compute_64;
    __nv_bfloat162* s_up_compute_32 = reinterpret_cast<__nv_bfloat162*>(&up64);

#pragma unroll
    for (int i = 0; i < 2; i++) {
      float2 gate = silu2(__bfloat1622float2(s_gate_compute_32[i]));
      results_bf162[i] = __float22bfloat162_rn(gate);
    }

#pragma unroll
    for (int i = 0; i < 2; i++) {
      results_bf162[i] = __hmul2(results_bf162[i], s_up_compute_32[i]);
    }

    auto _y_max2 =
        __hmax2(__habs2(results_bf162[0]), __habs2(results_bf162[1]));

    y_max_bf16 = __hmax(_y_max2.x, _y_max2.y);

    __nv_bfloat16 y_s = warp_max(y_max_bf16) / fp8_max;

    if constexpr (USE_UE8M0) {
      y_s = hexp2(hceil(hlog2(y_s)));
    }

    auto inv_y = __float2bfloat16_rn(1.f) / y_s;

    auto y_s2 = make_bfloat162(inv_y, inv_y);

#pragma unroll
    for (Idx_t i = 0; i < 2; ++i) {
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

}  // namespace vllm
