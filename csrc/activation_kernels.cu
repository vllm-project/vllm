#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}
// Activation and gating kernel template.

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

}  // namespace vllm

// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                 \
  int d = input.size(-1) / 2;                                            \
  int64_t num_tokens = input.numel() / input.size(-1);                   \
  dim3 grid(num_tokens);                                                 \
  dim3 block(std::min(d, 1024));                                         \
  if (num_tokens == 0) {                                                 \
    return;                                                              \
  }                                                                      \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));      \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();          \
  VLLM_DISPATCH_FLOATING_TYPES(                                          \
      input.scalar_type(), "act_and_mul_kernel", [&] {                   \
        vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>  \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),       \
                                         input.data_ptr<scalar_t>(), d); \
      });

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, true);
}

void mul_and_silu(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  // The difference between mul_and_silu and silu_and_mul is that mul_and_silu
  // applies the silu to the latter half of the input.
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, false);
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, true);
}

void gelu_tanh_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel, true);
}

namespace vllm {

template <typename T>
__device__ __forceinline__ T fatrelu_kernel(const T& x, const float threshold) {
  const float f = (float)x;
  return (T)(f > threshold ? f : 0.0f);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&, const float)>
__global__ void act_and_mul_kernel_with_param(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const int d,
    const float param) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = ACT_FN(x, param) * y;
  }
}

__device__ __forceinline__ float _swigluoai_and_mul(const float& gate,
                                                    const float& up,
                                                    float alpha, float limit) {
  // clamp gate: min=None, max=limit
  const float gate_f = (float)gate;
  const float clamped_gate = gate_f > limit ? limit : gate_f;

  // clamp up: min=-limit, max=limit
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  // glu = gate * sigmoid(gate * alpha)
  const float sigmoid_val = 1.0f / (1.0f + expf(-clamped_gate * alpha));
  const float glu = clamped_gate * sigmoid_val;

  // (up + 1) * glu
  return (float)((clamped_up + 1.0f) * glu);
}

template <typename T>
__device__ __forceinline__ T swigluoai_and_mul(const T& gate, const T& up,
                                               float alpha, float limit) {
  // clamp gate: min=None, max=limit
  const float gate_f = (float)gate;
  const float clamped_gate = gate_f > limit ? limit : gate_f;

  // clamp up: min=-limit, max=limit
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  // glu = gate * sigmoid(gate * alpha)
  const float sigmoid_val = 1.0f / (1.0f + expf(-clamped_gate * alpha));
  const float glu = clamped_gate * sigmoid_val;

  // (up + 1) * glu
  return (T)((clamped_up + 1.0f) * glu);
}

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float)>
__global__ void swigluoai_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d, 2]
    const int d, const float alpha, const float limit) {
  const int64_t token_idx = blockIdx.x;
  // TODO: Vectorize loads and stores.
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    // gate = x[..., ::2]  (even indices)
    const scalar_t gate = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx]);
    // up = x[..., 1::2]   (odd indices)
    const scalar_t up = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx + 1]);

    out[token_idx * d + idx] = ACT_FN(gate, up, alpha, limit);
  }
}

// CUDA Utils
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

template <float (*ACT_FN)(const float&, const float&, const float, const float),
          int BLOCK_COUNT, int SMEM_SIZE_BYTES_Y, int THREADS, typename Idx_t,
          int NUM_STAGES = 3>
__global__ void act_and_mul_masked_kernel(
    const __nv_bfloat16* __restrict__ _input,
    __nv_bfloat16* __restrict__ _output,
    const int32_t* __restrict__ tokens_per_expert,
    // sizes
    Idx_t E, Idx_t T, Idx_t H, float ALPHA, float LIMIT) {
#ifndef USE_ROCM
  static constexpr int NUM_WARPS = THREADS / WARP_SIZE;

  static constexpr int GROUP_SIZE = 128;

  static constexpr int LOAD_STAGE_SIZE = 2 * GROUP_SIZE / 8;
  static constexpr int LOAD_STAGE_MOD = NUM_STAGES * LOAD_STAGE_SIZE;

  static constexpr int COMPUTE_STAGE_SIZE = 2 * GROUP_SIZE / 4;
  static constexpr int COMPUTE_STAGE_MOD = COMPUTE_STAGE_SIZE * NUM_STAGES;

  extern __shared__ __align__(16) __int128_t smem_128[];

  int* s_expert_offsets =
      reinterpret_cast<int*>(smem_128 + (SMEM_SIZE_BYTES_Y / 16));

  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 0x1f;

  int running_sum{};
  if (!warp_id) {
    for (int i = 0; i < E; i += WARP_SIZE) {
      bool valid = (i + threadIdx.x) < E;
      int value = (valid ? tokens_per_expert[i + threadIdx.x] : 0) +
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

  // A single block will handle tokens_per_block tokens.
  // Each block i iterates over tokens of a slice of n_tokens =
  // expert_counts[i], with the size of chunk being
  // (n_tokens / NUM_PARALLEL_TOKENS) + residual, instead of
  // updiv(n_tokens, NUM_PARALLEL_TOKENS) for better scheduling.

  // Each warp will get space to store its hidden dim for gate and up.
  __int128_t* s_hidden_load = smem_128 + warp_id * ((2 * 128 / 8) * NUM_STAGES);
  __int128_t* smem_load_ptr = s_hidden_load + lane_id;

  int32_t compute_pipeline_offset_64 = 0;
  int32_t load_stage_offset{};

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

  Idx_t base_i = ((expert_id * T * H) / 8) + (token_id - expert_offset) * H / 8;
  const Idx_t gate_warp_offset =
      // TODO(elvircrn):
      //              v check this
      warp_id * (H / (8 * NUM_WARPS)) + (lane_id & 0b1111);

  const __int128_t* input_128_ptr =
      reinterpret_cast<const __int128_t*>(_input) + gate_warp_offset +
      ((lane_id < 16) ? 0 : (H / 8));
  __int64_t* out_64_ptr = reinterpret_cast<__int64_t*>(_output) +
                          2 * gate_warp_offset + ((lane_id < 16) ? 0 : (H / 4));
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

        base_i = expert_id * ((T * H) / 8);
        token_offset = 0;
        load_ptr = const_cast<__int128_t*>(input_128_ptr + base_i);
      } else {
        // We remain within the same expert, so just
        // move by H/4 __int128_t (2 * H/8).
        base_i += H / 4;
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

  for (auto j = tokens_lower; j < tokens_upper; j++) {
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
        auto in2 = __bfloat1622float2(s_gate_comp[k]);
        auto _in2 = __bfloat1622float2(s_up_comp[k]);
        __nv_bfloat162 gate = make_bfloat162(
            __float2bfloat16_rd(ACT_FN(in2.x, _in2.x, ALPHA, LIMIT)),
            __float2bfloat16_rd(ACT_FN(in2.y, _in2.y, ALPHA, LIMIT)));
        res[k] = gate;
      }

      *out_64_ptr = *reinterpret_cast<__int64_t*>(res);
      out_64_ptr += THREADS;
    }
  }
#endif
}

}  // namespace vllm

#define LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(KERNEL, PARAM)         \
  int d = input.size(-1) / 2;                                           \
  int64_t num_tokens = input.numel() / input.size(-1);                  \
  dim3 grid(num_tokens);                                                \
  dim3 block(std::min(d, 1024));                                        \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));     \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();         \
  VLLM_DISPATCH_FLOATING_TYPES(                                         \
      input.scalar_type(), "act_and_mul_kernel_with_param", [&] {       \
        vllm::act_and_mul_kernel_with_param<scalar_t, KERNEL<scalar_t>> \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),      \
                                         input.data_ptr<scalar_t>(), d, \
                                         PARAM);                        \
      });

#define LAUNCH_SIGLUOAI_AND_MUL(KERNEL, ALPHA, LIMIT)                          \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "clamp_swiglu_kernel_with_params", [&] {            \
        vllm::swigluoai_and_mul_kernel<scalar_t, KERNEL<scalar_t>>             \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),             \
                                         input.data_ptr<scalar_t>(), d, ALPHA, \
                                         LIMIT);                               \
      });

#define LAUNCH_SIGLUOAI_AND_MUL_MASKED(KERNEL, ALPHA, LIMIT, BLOCK_COUNT,    \
                                       THREAD_COUNT, NUM_STAGES)             \
  int64_t num_tokens = (input.numel() / input.size(-1)) / E;                 \
  dim3 grid(BLOCK_COUNT);                                                    \
  dim3 block(THREAD_COUNT);                                                  \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));          \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();              \
  using cuda_type = __nv_bfloat16;                                           \
  using cuda_type2 = __nv_bfloat16;                                          \
  static constexpr int NUM_WARPS = THREAD_COUNT / WARP_SIZE;                 \
  static constexpr int max_shared_mem_bytes =                                \
      128 * 2 * NUM_STAGES * NUM_WARPS * 2;                                  \
  vllm::act_and_mul_masked_kernel<KERNEL, BLOCK_COUNT, max_shared_mem_bytes, \
                                  THREAD_COUNT, __int64_t, NUM_STAGES>       \
      <<<grid, block, max_shared_mem_bytes + (E + 1) * 16, stream>>>(        \
          (const __nv_bfloat16*)input.data_ptr<at::BFloat16>(),              \
          (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),                      \
          tokens_per_expert.data_ptr<int32_t>(), E, num_tokens, d, ALPHA,    \
          LIMIT);

void fatrelu_and_mul(torch::Tensor& out,    // [..., d],
                     torch::Tensor& input,  // [..., 2 * d]
                     double threshold) {
  LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(vllm::fatrelu_kernel, threshold);
}
void swigluoai_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input,  // [..., 2 * d]
                       double alpha, double limit) {
  LAUNCH_SIGLUOAI_AND_MUL(vllm::swigluoai_and_mul, alpha, limit);
}

void swigluoai_and_mul_masked(torch::Tensor& out,    // [..., d]
                              torch::Tensor& input,  // [..., 2 * d]
                              int64_t E, torch::Tensor& tokens_per_expert,
                              double alpha = 1.702, double limit = 7.0) {
  int d = input.size(-1) / 2;
  static constexpr int BLOCK_COUNT = 132 * 8;
  if (d >= 4096) {
    static constexpr int NUM_STAGES = 4;
    static constexpr int THREAD_COUNT = 256;
    LAUNCH_SIGLUOAI_AND_MUL_MASKED(vllm::_swigluoai_and_mul, alpha, limit,
                                   BLOCK_COUNT, THREAD_COUNT, NUM_STAGES)
  } else {
    static constexpr int NUM_STAGES = 1;
    static constexpr int THREAD_COUNT = 32;
    LAUNCH_SIGLUOAI_AND_MUL_MASKED(vllm::_swigluoai_and_mul, alpha, limit,
                                   BLOCK_COUNT, THREAD_COUNT, NUM_STAGES)
  }
}

namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = ACT_FN(x);
  }
}

}  // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                       \
  int d = input.size(-1);                                                      \
  int64_t num_tokens = input.numel() / d;                                      \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activation_kernel", [&] { \
    vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>                        \
        <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),                 \
                                     input.data_ptr<scalar_t>(), d);           \
  });

namespace vllm {

template <typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + expf(-1.702f * (float)x)));
}

}  // namespace vllm

void gelu_new(torch::Tensor& out,    // [..., d]
              torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(torch::Tensor& out,    // [..., d]
               torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}

void gelu_quick(torch::Tensor& out,    // [..., d]
                torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
}