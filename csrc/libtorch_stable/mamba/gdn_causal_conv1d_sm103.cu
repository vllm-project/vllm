#include "../torch_utils.h"
#include "../ops.h"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

__device__ __forceinline__ float silu_approx(float x) {
  float t;
  asm("tanh.approx.f32 %0,%1;" : "=f"(t) : "f"(0.5f * x));
  return 0.5f * x * (1.f + t);
}

template <int V, int BLOCK, int CHUNK, int DEPTH, int MIN_BLOCKS, bool HAS_BIAS>
__global__ void __launch_bounds__(BLOCK, MIN_BLOCKS)
    conv_fwd(const __nv_bfloat16* __restrict__ x,
             const __nv_bfloat16* __restrict__ weight,
             const __nv_bfloat16* __restrict__ bias,
             __nv_bfloat16* __restrict__ states,
             const int* __restrict__ state_indices,
             const bool* __restrict__ has_initial_state,
             __nv_bfloat16* __restrict__ output, int dim, int tokens,
             long stride_x_token, long stride_output_token,
             long stride_state_seq, long stride_state_dim,
             long stride_state_token) {
  using Vec = float2;
  constexpr int LANES = V / 2;
  const int channel_groups = dim / V;
  const int token_chunks = (tokens + CHUNK - 1) / CHUNK;
  const long linear = static_cast<long>(blockIdx.x) * BLOCK + threadIdx.x;
  const int channel_group = linear % channel_groups;
  const int chunk = linear / channel_groups;
  if (chunk >= token_chunks) return;

  const int channel = channel_group * V;
  const int token_start = chunk * CHUNK;
  const int token_end = min(token_start + CHUNK, tokens);
  const int state_index = state_indices[0];
  const bool use_initial_state = has_initial_state[0];
  if (state_index <= 0) return;

  extern __shared__ __nv_bfloat16 ring[];
  __nv_bfloat16* thread_ring =
      ring + static_cast<long>(threadIdx.x) * DEPTH * V;
  __nv_bfloat162 weights[4][LANES];
  __nv_bfloat162 biases[LANES];
#pragma unroll
  for (int lane = 0; lane < LANES; ++lane) {
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
      weights[tap][lane] =
          __halves2bfloat162(weight[(channel + 2 * lane) * 4 + tap],
                             weight[(channel + 2 * lane + 1) * 4 + tap]);
    }
    if constexpr (HAS_BIAS) {
      biases[lane] = __halves2bfloat162(bias[channel + 2 * lane],
                                        bias[channel + 2 * lane + 1]);
    } else {
      biases[lane] = __float2bfloat162_rn(0.f);
    }
  }

  __nv_bfloat162 tap0[LANES], tap1[LANES], tap2[LANES];
  auto load_x = [&](int token, __nv_bfloat162* destination) {
    Vec packed = *reinterpret_cast<const Vec*>(
        x + static_cast<long>(token) * stride_x_token + channel);
    const auto* halves = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
    for (int lane = 0; lane < LANES; ++lane) destination[lane] = halves[lane];
  };
  auto load_state = [&](int token, __nv_bfloat162* destination) {
#pragma unroll
    for (int lane = 0; lane < LANES; ++lane) {
      long base = static_cast<long>(state_index) * stride_state_seq +
                  static_cast<long>(channel + 2 * lane) * stride_state_dim +
                  static_cast<long>(token) * stride_state_token;
      destination[lane] =
          __halves2bfloat162(states[base], states[base + stride_state_dim]);
    }
  };
  auto zero = [&](__nv_bfloat162* destination) {
#pragma unroll
    for (int lane = 0; lane < LANES; ++lane) {
      destination[lane] = __float2bfloat162_rn(0.f);
    }
  };

  if (token_start == 0) {
    if (use_initial_state) {
      load_state(0, tap0);
      load_state(1, tap1);
      load_state(2, tap2);
    } else {
      zero(tap0);
      zero(tap1);
      zero(tap2);
    }
  } else {
    load_x(token_start - 3, tap0);
    load_x(token_start - 2, tap1);
    load_x(token_start - 1, tap2);
  }

#pragma unroll
  for (int stage = 0; stage < DEPTH; ++stage) {
    int token = token_start + stage;
    if (token < token_end) {
      __pipeline_memcpy_async(
          thread_ring + stage * V,
          x + static_cast<long>(token) * stride_x_token + channel,
          V * sizeof(__nv_bfloat16));
    }
    __pipeline_commit();
  }

  for (int token = token_start; token < token_end; ++token) {
    int slot = (token - token_start) % DEPTH;
    __pipeline_wait_prior(DEPTH - 1);
    __nv_bfloat162 current[LANES];
    Vec packed = *reinterpret_cast<const Vec*>(thread_ring + slot * V);
    const auto* halves = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
    for (int lane = 0; lane < LANES; ++lane) current[lane] = halves[lane];

    int next_token = token + DEPTH;
    if (next_token < token_end) {
      __pipeline_memcpy_async(
          thread_ring + slot * V,
          x + static_cast<long>(next_token) * stride_x_token + channel,
          V * sizeof(__nv_bfloat16));
    }
    __pipeline_commit();

    Vec packed_output;
    auto* output_halves = reinterpret_cast<__nv_bfloat162*>(&packed_output);
#pragma unroll
    for (int lane = 0; lane < LANES; ++lane) {
      __nv_bfloat162 acc =
          __hfma2(weights[0][lane], tap0[lane],
                  __hfma2(weights[1][lane], tap1[lane],
                          __hfma2(weights[2][lane], tap2[lane],
                                  __hmul2(weights[3][lane], current[lane]))));
      acc = __hadd2(acc, biases[lane]);
      output_halves[lane] =
          __halves2bfloat162(__float2bfloat16(silu_approx(__low2float(acc))),
                             __float2bfloat16(silu_approx(__high2float(acc))));
      tap0[lane] = tap1[lane];
      tap1[lane] = tap2[lane];
      tap2[lane] = current[lane];
    }
    *reinterpret_cast<Vec*>(
        output + static_cast<long>(token) * stride_output_token + channel) =
        packed_output;
  }

  if (chunk == token_chunks - 1) {
#pragma unroll
    for (int state_token = 0; state_token < 3; ++state_token) {
      int source_token = tokens - 3 + state_token;
#pragma unroll
      for (int lane = 0; lane < V; ++lane) {
        long state_offset =
            static_cast<long>(state_index) * stride_state_seq +
            static_cast<long>(channel + lane) * stride_state_dim +
            static_cast<long>(state_token) * stride_state_token;
        states[state_offset] =
            x[static_cast<long>(source_token) * stride_x_token + channel +
              lane];
      }
    }
  }
}

torch::stable::Tensor gdn_causal_conv1d_sm103(
    const torch::stable::Tensor& x, const torch::stable::Tensor& weight,
    const torch::stable::Tensor& bias, torch::stable::Tensor& states,
    const torch::stable::Tensor& state_indices,
    const torch::stable::Tensor& has_initial_state, bool has_bias) {
  const int dim = static_cast<int>(x.size(0));
  const int tokens = static_cast<int>(x.size(1));
  auto output = torch::stable::empty_like(x);
  constexpr int V = 4;
  constexpr int BLOCK = 128;
  constexpr int CHUNK = 64;
  constexpr int DEPTH = 6;
  const int64_t threads =
      static_cast<int64_t>(dim / V) * ((tokens + CHUNK - 1) / CHUNK);
  const int64_t grid = (threads + BLOCK - 1) / BLOCK;
  const size_t shared =
      static_cast<size_t>(BLOCK) * DEPTH * V * sizeof(__nv_bfloat16);
  const torch::stable::accelerator::DeviceGuard device_guard(
      x.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  if (has_bias) {
    conv_fwd<V, BLOCK, CHUNK, DEPTH, 8, true><<<grid, BLOCK, shared, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(bias.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(states.mutable_data_ptr()),
        reinterpret_cast<const int*>(state_indices.const_data_ptr()),
        reinterpret_cast<const bool*>(has_initial_state.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()), dim,
        tokens, x.stride(1), output.stride(1), states.stride(0),
        states.stride(1), states.stride(2));
  } else {
    conv_fwd<V, BLOCK, CHUNK, DEPTH, 8, false><<<grid, BLOCK, shared, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(bias.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(states.mutable_data_ptr()),
        reinterpret_cast<const int*>(state_indices.const_data_ptr()),
        reinterpret_cast<const bool*>(has_initial_state.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()), dim,
        tokens, x.stride(1), output.stride(1), states.stride(0),
        states.stride(1), states.stride(2));
  }
  return output;
}
