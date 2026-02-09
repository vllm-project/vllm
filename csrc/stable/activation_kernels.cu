#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cmath>
#include <algorithm>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "torch_utils.h"

namespace vllm {

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

// Check if all pointers are 16-byte aligned for int4 vectorized access
__device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

// Activation and gating kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
  const int64_t token_idx = blockIdx.x;
  const scalar_t* x_ptr = input + token_idx * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + token_idx * d;

  // Check alignment for 128-bit vectorized access.
  // All three pointers must be 16-byte aligned for safe int4 operations.
  const bool aligned = is_16byte_aligned(x_ptr) && is_16byte_aligned(y_ptr) &&
                       is_16byte_aligned(out_ptr);

  if (aligned && d >= VEC_SIZE) {
    // Fast path: 128-bit vectorized loop
    const int4* x_vec = reinterpret_cast<const int4*>(x_ptr);
    const int4* y_vec = reinterpret_cast<const int4*>(y_ptr);
    int4* out_vec = reinterpret_cast<int4*>(out_ptr);
    const int num_vecs = d / VEC_SIZE;
    const int vec_end = num_vecs * VEC_SIZE;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      int4 x = VLLM_LDG(&x_vec[i]), y = VLLM_LDG(&y_vec[i]), r;
      auto* xp = reinterpret_cast<scalar_t*>(&x);
      auto* yp = reinterpret_cast<scalar_t*>(&y);
      auto* rp = reinterpret_cast<scalar_t*>(&r);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        rp[j] = compute<scalar_t, ACT_FN, act_first>(xp[j], yp[j]);
      }
      out_vec[i] = r;
    }
    // Scalar cleanup for remaining elements
    for (int i = vec_end + threadIdx.x; i < d; i += blockDim.x) {
      out_ptr[i] = compute<scalar_t, ACT_FN, act_first>(VLLM_LDG(&x_ptr[i]),
                                                        VLLM_LDG(&y_ptr[i]));
    }
  } else {
    // Scalar fallback for unaligned data or small d
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&x_ptr[idx]);
      const scalar_t y = VLLM_LDG(&y_ptr[idx]);
      out_ptr[idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
    }
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

// Launch activation and gating kernel using stable APIs.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                       \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  if (num_tokens == 0) {                                                       \
    return;                                                                    \
  }                                                                            \
  torch::stable::accelerator::DeviceGuard device_guard(                        \
      input.get_device_index());                                               \
  cudaStream_t stream = get_current_cuda_stream(input.get_device_index());     \
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                         \
      input.scalar_type(), "act_and_mul_kernel", [&] {                         \
        vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>        \
            <<<grid, block, 0, stream>>>(out.mutable_data_ptr<scalar_t>(),     \
                                         input.const_data_ptr<scalar_t>(), d); \
      });

void silu_and_mul(torch::stable::Tensor& out,    // [..., d]
                  torch::stable::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, true);
}

void mul_and_silu(torch::stable::Tensor& out,    // [..., d]
                  torch::stable::Tensor& input)  // [..., 2 * d]
{
  // The difference between mul_and_silu and silu_and_mul is that mul_and_silu
  // applies the silu to the latter half of the input.
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, false);
}

void gelu_and_mul(torch::stable::Tensor& out,    // [..., d]
                  torch::stable::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, true);
}

void gelu_tanh_and_mul(torch::stable::Tensor& out,    // [..., d]
                       torch::stable::Tensor& input)  // [..., 2 * d]
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
  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
  const int64_t token_idx = blockIdx.x;
  const scalar_t* x_ptr = input + token_idx * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + token_idx * d;

  // Check alignment for 128-bit vectorized access
  const bool aligned = is_16byte_aligned(x_ptr) && is_16byte_aligned(y_ptr) &&
                       is_16byte_aligned(out_ptr);

  if (aligned && d >= VEC_SIZE) {
    // Fast path: 128-bit vectorized loop
    const int4* x_vec = reinterpret_cast<const int4*>(x_ptr);
    const int4* y_vec = reinterpret_cast<const int4*>(y_ptr);
    int4* out_vec = reinterpret_cast<int4*>(out_ptr);
    const int num_vecs = d / VEC_SIZE;
    const int vec_end = num_vecs * VEC_SIZE;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      int4 x = VLLM_LDG(&x_vec[i]), y = VLLM_LDG(&y_vec[i]), r;
      auto* xp = reinterpret_cast<scalar_t*>(&x);
      auto* yp = reinterpret_cast<scalar_t*>(&y);
      auto* rp = reinterpret_cast<scalar_t*>(&r);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        rp[j] = ACT_FN(xp[j], param) * yp[j];
      }
      out_vec[i] = r;
    }
    // Scalar cleanup for remaining elements
    for (int i = vec_end + threadIdx.x; i < d; i += blockDim.x) {
      out_ptr[i] = ACT_FN(VLLM_LDG(&x_ptr[i]), param) * VLLM_LDG(&y_ptr[i]);
    }
  } else {
    // Scalar fallback for unaligned data or small d
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&x_ptr[idx]);
      const scalar_t y = VLLM_LDG(&y_ptr[idx]);
      out_ptr[idx] = ACT_FN(x, param) * y;
    }
  }
}

template <typename T>
__device__ __forceinline__ T swigluoai_and_mul(const T& gate, const T& up,
                                               float alpha, float limit) {
  // Clamp gate to (-inf, limit] and up to [-limit, limit]
  const float g = fminf((float)gate, limit);
  const float u = fmaxf(fminf((float)up, limit), -limit);
  // glu = gate * sigmoid(gate * alpha), then return (up + 1) * glu
  return (T)((u + 1.0f) * g / (1.0f + expf(-g * alpha)));
}

// Interleaved gate/up: input has [gate0, up0, gate1, up1, ...].
template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float)>
__global__ void swigluoai_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2 * d] (interleaved)
    const int d, const float alpha, const float limit) {
  // For interleaved data: input has 2*d elements per token (gate/up pairs)
  // output has d elements per token
  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
  constexpr int PAIRS = VEC_SIZE / 2;  // Number of gate/up pairs per int4 load
  const int64_t token_idx = blockIdx.x;
  const scalar_t* in_ptr = input + token_idx * 2 * d;
  scalar_t* out_ptr = out + token_idx * d;

  // Check alignment for 128-bit vectorized access on input.
  // For output we use int2 (64-bit) which has 8-byte alignment requirement.
  const bool in_aligned = is_16byte_aligned(in_ptr);
  const bool out_aligned =
      (reinterpret_cast<uintptr_t>(out_ptr) & 7) == 0;  // 8-byte for int2

  if (in_aligned && out_aligned && d >= PAIRS) {
    // Fast path: vectorized loop
    // Each int4 load gives VEC_SIZE elements = PAIRS gate/up pairs
    // Each int2 store writes PAIRS output elements
    const int4* in_vec = reinterpret_cast<const int4*>(in_ptr);
    int2* out_vec = reinterpret_cast<int2*>(out_ptr);
    const int num_vecs = d / PAIRS;
    const int vec_end = num_vecs * PAIRS;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      int4 v = VLLM_LDG(&in_vec[i]);
      int2 r;
      auto* vp = reinterpret_cast<scalar_t*>(&v);
      auto* rp = reinterpret_cast<scalar_t*>(&r);
#pragma unroll
      for (int j = 0; j < PAIRS; j++) {
        rp[j] = ACT_FN(vp[2 * j], vp[2 * j + 1], alpha, limit);
      }
      out_vec[i] = r;
    }
    // Scalar cleanup for remaining elements
    for (int i = vec_end + threadIdx.x; i < d; i += blockDim.x) {
      out_ptr[i] = ACT_FN(VLLM_LDG(&in_ptr[2 * i]),
                          VLLM_LDG(&in_ptr[2 * i + 1]), alpha, limit);
    }
  } else {
    // Scalar fallback for unaligned data or small d
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      // gate = x[..., ::2]  (even indices)
      const scalar_t gate = VLLM_LDG(&in_ptr[2 * idx]);
      // up = x[..., 1::2]   (odd indices)
      const scalar_t up = VLLM_LDG(&in_ptr[2 * idx + 1]);
      out_ptr[idx] = ACT_FN(gate, up, alpha, limit);
    }
  }
}

}  // namespace vllm

#define LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(KERNEL, PARAM)               \
  int d = input.size(-1) / 2;                                                 \
  int64_t num_tokens = input.numel() / input.size(-1);                        \
  dim3 grid(num_tokens);                                                      \
  dim3 block(std::min(d, 1024));                                              \
  torch::stable::accelerator::DeviceGuard device_guard(                       \
      input.get_device_index());                                              \
  cudaStream_t stream = get_current_cuda_stream(input.get_device_index());    \
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                        \
      input.scalar_type(), "act_and_mul_kernel_with_param", [&] {             \
        vllm::act_and_mul_kernel_with_param<scalar_t, KERNEL<scalar_t>>       \
            <<<grid, block, 0, stream>>>(out.mutable_data_ptr<scalar_t>(),    \
                                         input.const_data_ptr<scalar_t>(), d, \
                                         PARAM);                              \
      });

#define LAUNCH_SIGLUOAI_AND_MUL(KERNEL, ALPHA, LIMIT)                         \
  int d = input.size(-1) / 2;                                                 \
  int64_t num_tokens = input.numel() / input.size(-1);                        \
  dim3 grid(num_tokens);                                                      \
  dim3 block(std::min(d, 1024));                                              \
  torch::stable::accelerator::DeviceGuard device_guard(                       \
      input.get_device_index());                                              \
  cudaStream_t stream = get_current_cuda_stream(input.get_device_index());    \
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                        \
      input.scalar_type(), "clamp_swiglu_kernel_with_params", [&] {           \
        vllm::swigluoai_and_mul_kernel<scalar_t, KERNEL<scalar_t>>            \
            <<<grid, block, 0, stream>>>(out.mutable_data_ptr<scalar_t>(),    \
                                         input.const_data_ptr<scalar_t>(), d, \
                                         ALPHA, LIMIT);                       \
      });

void fatrelu_and_mul(torch::stable::Tensor& out,    // [..., d],
                     torch::stable::Tensor& input,  // [..., 2 * d]
                     double threshold) {
  LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(vllm::fatrelu_kernel, threshold);
}

void swigluoai_and_mul(torch::stable::Tensor& out,    // [..., d]
                       torch::stable::Tensor& input,  // [..., 2 * d]
                       double alpha, double limit) {
  LAUNCH_SIGLUOAI_AND_MUL(vllm::swigluoai_and_mul, alpha, limit);
}

namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {
  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
  const int64_t token_idx = blockIdx.x;
  const scalar_t* in_ptr = input + token_idx * d;
  scalar_t* out_ptr = out + token_idx * d;

  // Check alignment for 128-bit vectorized access
  const bool aligned = is_16byte_aligned(in_ptr) && is_16byte_aligned(out_ptr);

  if (aligned && d >= VEC_SIZE) {
    // Fast path: 128-bit vectorized loop
    const int4* in_vec = reinterpret_cast<const int4*>(in_ptr);
    int4* out_vec = reinterpret_cast<int4*>(out_ptr);
    const int num_vecs = d / VEC_SIZE;
    const int vec_end = num_vecs * VEC_SIZE;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      int4 v = VLLM_LDG(&in_vec[i]), r;
      auto* vp = reinterpret_cast<scalar_t*>(&v);
      auto* rp = reinterpret_cast<scalar_t*>(&r);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        rp[j] = ACT_FN(vp[j]);
      }
      out_vec[i] = r;
    }
    // Scalar cleanup for remaining elements
    for (int i = vec_end + threadIdx.x; i < d; i += blockDim.x) {
      out_ptr[i] = ACT_FN(VLLM_LDG(&in_ptr[i]));
    }
  } else {
    // Scalar fallback for unaligned data or small d
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&in_ptr[idx]);
      out_ptr[idx] = ACT_FN(x);
    }
  }
}

}  // namespace vllm

// Launch element-wise activation kernel using stable APIs.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                       \
  int d = input.size(-1);                                                      \
  int64_t num_tokens = input.numel() / d;                                      \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  torch::stable::accelerator::DeviceGuard device_guard(                        \
      input.get_device_index());                                               \
  cudaStream_t stream = get_current_cuda_stream(input.get_device_index());     \
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                         \
      input.scalar_type(), "activation_kernel", [&] {                          \
        vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>                    \
            <<<grid, block, 0, stream>>>(out.mutable_data_ptr<scalar_t>(),     \
                                         input.const_data_ptr<scalar_t>(), d); \
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

void gelu_new(torch::stable::Tensor& out,    // [..., d]
              torch::stable::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(torch::stable::Tensor& out,    // [..., d]
               torch::stable::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}

void gelu_quick(torch::stable::Tensor& out,    // [..., d]
                torch::stable::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
}
