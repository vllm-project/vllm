#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>

#ifndef USE_ROCM
  #include <cuda_fp8.h>
#endif

#include <cmath>
#include <type_traits>

#include "../cuda_compat.h"
#include "async_util.cuh"
#include "cuda_vec_utils.cuh"
#include "dispatch_utils.h"
#include "torch_utils.h"

namespace vllm {

// `alpha` and `beta` are applied to opposite operands:
//   - alpha lives INSIDE the activation (the activated half): the gated
//     activation computes act_half * sigmoid(alpha * act_half).
//   - beta is added to the OTHER (non-activated) half before the multiply.
// So the result is always ACT(act_half, alpha) * (other_half + beta).
// Which half is which depends on `act_first` (see below). Defaults
// alpha=1.0, beta=0.0 reproduce the plain SwiGLU/GeGLU behavior.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&, const float),
          bool act_first, bool HAS_CLAMP>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y,
                                            const float limit,
                                            const float alpha,
                                            const float beta) {
  if constexpr (act_first) {
    scalar_t gate = x;
    scalar_t up = y;
    if constexpr (HAS_CLAMP) {
      gate = (scalar_t)fminf((float)gate, limit);
      up = (scalar_t)fmaxf(fminf((float)up, limit), -limit);
    }
    // act_first: gate is the activated half -> alpha applies to gate;
    // beta is added to up (the non-activated half).
    return (scalar_t)(ACT_FN(gate, alpha) * ((float)up + beta));
  } else {
    scalar_t gate = x;
    scalar_t up = y;
    if constexpr (HAS_CLAMP) {
      gate = (scalar_t)fmaxf(fminf((float)gate, limit), -limit);
      up = (scalar_t)fminf((float)up, limit);
    }
    // !act_first: up is the activated half -> alpha applies to up;
    // beta is added to gate (the non-activated half).
    return (scalar_t)(((float)gate + beta) * ACT_FN(up, alpha));
  }
}

template <typename packed_t,
          packed_t (*PACKED_ACT_FN)(const packed_t&, const float),
          bool act_first, bool HAS_CLAMP>
__device__ __forceinline__ packed_t packed_compute(const packed_t& x,
                                                   const packed_t& y,
                                                   const float limit,
                                                   const float alpha,
                                                   const float beta) {
  if constexpr (act_first) {
    packed_t gate = x;
    packed_t up = y;
    float2 u = cast_to_float2(up);
    if constexpr (HAS_CLAMP) {
      float2 g = cast_to_float2(gate);
      g.x = fminf(g.x, limit);
      g.y = fminf(g.y, limit);
      u.x = fmaxf(fminf(u.x, limit), -limit);
      u.y = fmaxf(fminf(u.y, limit), -limit);
      gate = cast_to_packed<packed_t>(g);
    }
    // act_first: gate is the activated half -> alpha applies to gate;
    // beta is added to up (the non-activated half).
    float2 activated = cast_to_float2(PACKED_ACT_FN(gate, alpha));
    activated.x *= u.x + beta;
    activated.y *= u.y + beta;
    return cast_to_packed<packed_t>(activated);
  } else {
    packed_t gate = x;
    packed_t up = y;
    float2 g = cast_to_float2(gate);
    if constexpr (HAS_CLAMP) {
      float2 u = cast_to_float2(up);
      g.x = fmaxf(fminf(g.x, limit), -limit);
      g.y = fmaxf(fminf(g.y, limit), -limit);
      u.x = fminf(u.x, limit);
      u.y = fminf(u.y, limit);
      up = cast_to_packed<packed_t>(u);
    }
    // !act_first: up is the activated half -> alpha applies to up;
    // beta is added to gate (the non-activated half).
    float2 activated = cast_to_float2(PACKED_ACT_FN(up, alpha));
    activated.x *= g.x + beta;
    activated.y *= g.y + beta;
    return cast_to_packed<packed_t>(activated);
  }
}

// Activation and gating kernel template.
template <typename scalar_t, typename packed_t,
          scalar_t (*ACT_FN)(const scalar_t&, const float),
          packed_t (*PACKED_ACT_FN)(const packed_t&, const float),
          bool act_first, bool use_vec, bool HAS_CLAMP, bool use_256b = false>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d, const float limit, const float alpha, const float beta) {
  const int64_t token_idx = blockIdx.x;
  const scalar_t* x_ptr = input + token_idx * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + token_idx * d;

  if constexpr (use_vec) {
    using cuda_t = typename CUDATypeConverter<scalar_t>::Type;
    using pvec_t = PackedVec<cuda_t, use_256b>;

    const pvec_t* x_vec = reinterpret_cast<const pvec_t*>(x_ptr);
    const pvec_t* y_vec = reinterpret_cast<const pvec_t*>(y_ptr);
    pvec_t* out_vec = reinterpret_cast<pvec_t*>(out_ptr);
    const int num_vecs = d / 2 / pvec_t::NUM_ELTS;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      pvec_t x, y;
      if constexpr (use_256b) {
        ld256(x, &x_vec[i]);
        ld256(y, &y_vec[i]);
      } else {
        ld128(x, &x_vec[i]);
        ld128(y, &y_vec[i]);
      }
#pragma unroll
      for (int j = 0; j < pvec_t::NUM_ELTS; j++) {
        x.elts[j] =
            packed_compute<packed_t, PACKED_ACT_FN, act_first, HAS_CLAMP>(
                x.elts[j], y.elts[j], limit, alpha, beta);
      }
      if constexpr (use_256b) {
        st256(x, &out_vec[i]);
      } else {
        st128(x, &out_vec[i]);
      }
    }
  } else {
    // Scalar fallback for unaligned data or small d
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&x_ptr[idx]);
      const scalar_t y = VLLM_LDG(&y_ptr[idx]);
      out_ptr[idx] = compute<scalar_t, ACT_FN, act_first, HAS_CLAMP>(
          x, y, limit, alpha, beta);
    }
  }
}

// Gated activations take an `alpha` argument that scales the sigmoid input
// (`x * sigmoid(alpha * x)`). alpha defaults to 1.0 at all call sites, which
// is exactly SiLU; only the clamp path (silu_and_mul_with_clamp) passes a
// non-default alpha. Activations that do not use alpha simply ignore it.
template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x, const float alpha) {
  // x * sigmoid(alpha * x)
  return (T)(((float)x) / (1.0f + expf((float)-x * alpha)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_silu_kernel(const packed_t& val,
                                                       const float alpha) {
  // x * sigmoid(alpha * x)
  float2 fval = cast_to_float2(val);
  fval.x = fval.x / (1.0f + expf(-fval.x * alpha));
  fval.y = fval.y / (1.0f + expf(-fval.y * alpha));
  return cast_to_packed<packed_t>(fval);
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x, const float /*alpha*/) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_gelu_kernel(const packed_t& val,
                                                       const float /*alpha*/) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  constexpr float ALPHA = M_SQRT1_2;
  float2 fval = cast_to_float2(val);
  fval.x = fval.x * 0.5f * (1.0f + ::erf(fval.x * ALPHA));
  fval.y = fval.y * 0.5f * (1.0f + ::erf(fval.y * ALPHA));
  return cast_to_packed<packed_t>(fval);
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x,
                                              const float /*alpha*/) {
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

template <typename packed_t>
__device__ __forceinline__ packed_t
packed_gelu_tanh_kernel(const packed_t& val, const float /*alpha*/) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  float2 fval = cast_to_float2(val);
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;

  float x_cube = fval.x * fval.x * fval.x;
  float inner = BETA * (fval.x + KAPPA * x_cube);
  fval.x = 0.5f * fval.x * (1.0f + ::tanhf(inner));

  x_cube = fval.y * fval.y * fval.y;
  inner = BETA * (fval.y + KAPPA * x_cube);
  fval.y = 0.5f * fval.y * (1.0f + ::tanhf(inner));
  return cast_to_packed<packed_t>(fval);
}

}  // namespace vllm

// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first. HAS_CLAMP (bool) enables pre-activation clamping: gate input is
// clamped (max only) and up input is clamped (both sides) before the
// activation function is applied.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, PACKED_KERNEL, ACT_FIRST,        \
                                      HAS_CLAMP, LIMIT, ALPHA, BETA)           \
  auto dtype = input.scalar_type();                                            \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  if (num_tokens == 0) {                                                       \
    return;                                                                    \
  }                                                                            \
  dim3 grid(num_tokens);                                                       \
  int cc_major = get_device_prop()->major;                                     \
  int support_vec =                                                            \
      (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128)            \
          ? vllm::VecTraits<true>::ARCH_MAX_VEC_SIZE                           \
          : vllm::VecTraits<false>::ARCH_MAX_VEC_SIZE;                         \
  int vec_size = support_vec / input.element_size();                           \
  const bool use_vec = (d % vec_size == 0);                                    \
  const torch::stable::accelerator::DeviceGuard device_guard(                  \
      input.get_device_index());                                               \
  const cudaStream_t stream = get_current_cuda_stream();                       \
  if (use_vec) {                                                               \
    dim3 block(std::min(d / vec_size, 1024));                                  \
    if (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128) {         \
      VLLM_STABLE_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel", [&] {   \
        vllm::act_and_mul_kernel<                                              \
            scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,      \
            KERNEL<scalar_t>,                                                  \
            PACKED_KERNEL<typename vllm::PackedTypeConverter<scalar_t>::Type>, \
            ACT_FIRST, true, HAS_CLAMP, true><<<grid, block, 0, stream>>>(     \
            out.mutable_data_ptr<scalar_t>(),                                  \
            input.const_data_ptr<scalar_t>(), d, LIMIT, ALPHA, BETA);          \
      });                                                                      \
    } else {                                                                   \
      VLLM_STABLE_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel", [&] {   \
        vllm::act_and_mul_kernel<                                              \
            scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,      \
            KERNEL<scalar_t>,                                                  \
            PACKED_KERNEL<typename vllm::PackedTypeConverter<scalar_t>::Type>, \
            ACT_FIRST, true, HAS_CLAMP, false><<<grid, block, 0, stream>>>(    \
            out.mutable_data_ptr<scalar_t>(),                                  \
            input.const_data_ptr<scalar_t>(), d, LIMIT, ALPHA, BETA);          \
      });                                                                      \
    }                                                                          \
  } else {                                                                     \
    dim3 block(std::min(d, 1024));                                             \
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel", [&] {     \
      vllm::act_and_mul_kernel<                                                \
          scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,        \
          KERNEL<scalar_t>,                                                    \
          PACKED_KERNEL<typename vllm::PackedTypeConverter<scalar_t>::Type>,   \
          ACT_FIRST, false, HAS_CLAMP><<<grid, block, 0, stream>>>(            \
          out.mutable_data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(),  \
          d, LIMIT, ALPHA, BETA);                                              \
    });                                                                        \
  }

void silu_and_mul(torch::stable::Tensor& out,    // [..., d]
                  torch::stable::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, vllm::packed_silu_kernel,
                                true, false, 0.0f, 1.0f, 0.0f);
}

void silu_and_mul_clamp(torch::stable::Tensor& out,    // [..., d]
                        torch::stable::Tensor& input,  // [..., 2 * d]
                        double limit, double alpha, double beta) {
  // out = (gate.clamp(max=limit) * sigmoid(alpha * gate.clamp(max=limit)))
  //       * (up.clamp(+-limit) + beta)
  // alpha=1.0, beta=0.0 reduce this to silu(gate) * up.
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, vllm::packed_silu_kernel,
                                true, true, (float)limit, (float)alpha,
                                (float)beta);
}

void mul_and_silu(torch::stable::Tensor& out,    // [..., d]
                  torch::stable::Tensor& input)  // [..., 2 * d]
{
  // The difference between mul_and_silu and silu_and_mul is that mul_and_silu
  // applies the silu to the latter half of the input.
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, vllm::packed_silu_kernel,
                                false, false, 0.0f, 1.0f, 0.0f);
}

void gelu_and_mul(torch::stable::Tensor& out,    // [..., d]
                  torch::stable::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, vllm::packed_gelu_kernel,
                                true, false, 0.0f, 1.0f, 0.0f);
}

void gelu_tanh_and_mul(torch::stable::Tensor& out,    // [..., d]
                       torch::stable::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel,
                                vllm::packed_gelu_tanh_kernel, true, false,
                                0.0f, 1.0f, 0.0f);
}

namespace vllm {

template <typename T>
__device__ __forceinline__ T fatrelu_kernel(const T& x, const float threshold) {
  const float f = (float)x;
  return (T)(f > threshold ? f : 0.0f);
}

template <typename packed_t>
__device__ __forceinline__ packed_t
packed_fatrelu_kernel(const packed_t& val, const float threshold) {
  float2 fval = cast_to_float2(val);
  fval.x = fval.x > threshold ? fval.x : 0.0f;
  fval.y = fval.y > threshold ? fval.y : 0.0f;
  return cast_to_packed<packed_t>(fval);
}

template <typename scalar_t, typename packed_t,
          scalar_t (*ACT_FN)(const scalar_t&, const float),
          packed_t (*PACKED_ACT_FN)(const packed_t&, const float), bool use_vec,
          bool use_256b = false>
__global__ void act_and_mul_kernel_with_param(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const int d,
    const float param) {
  const int64_t token_idx = blockIdx.x;
  const scalar_t* x_ptr = input + token_idx * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + token_idx * d;

  if constexpr (use_vec) {
    using cuda_t = typename CUDATypeConverter<scalar_t>::Type;
    using pvec_t = PackedVec<cuda_t, use_256b>;

    const pvec_t* x_vec = reinterpret_cast<const pvec_t*>(x_ptr);
    const pvec_t* y_vec = reinterpret_cast<const pvec_t*>(y_ptr);
    pvec_t* out_vec = reinterpret_cast<pvec_t*>(out_ptr);
    const int num_vecs = d / 2 / pvec_t::NUM_ELTS;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      pvec_t x, y;
      if constexpr (use_256b) {
        ld256(x, &x_vec[i]);
        ld256(y, &y_vec[i]);
      } else {
        ld128(x, &x_vec[i]);
        ld128(y, &y_vec[i]);
      }
#pragma unroll
      for (int j = 0; j < pvec_t::NUM_ELTS; j++) {
        x.elts[j] = packed_mul(PACKED_ACT_FN(x.elts[j], param), y.elts[j]);
      }
      if constexpr (use_256b) {
        st256(x, &out_vec[i]);
      } else {
        st128(x, &out_vec[i]);
      }
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

// SITU (Kimi SituGLU) gated activation. Non-interleaved layout:
// input = [gate(d), up(d)] per token.
//   gate_out = beta * tanh(gate / beta) * sigmoid(gate)
//   up_out   = (linear_beta > 0) ? linear_beta * tanh(up / linear_beta) : up
//   out      = gate_out * up_out
// Compute is done in fp32 and written straight to `out` -- no intermediate
// tensors and no full-tensor fp32 upcast (the pure-torch forward_native
// allocated ~8 fp32 temporaries per call, which blows up MoE profiling).
// Single shared implementation of the SITU math, called from every kernel
// below. __forceinline__ since it runs inside #pragma-unrolled per-element
// loops, where an out-of-line call would spike register pressure for no gain.
//
// tanh via the sm_75+ hardware tanh.approx.f32 (single MUFU). Both tanhf and
// libdevice __tanhf compile to an out-of-line range-reduction CALL, which drags
// in a stack frame and ABI register spills; the PTX intrinsic has no slow path.
__device__ __forceinline__ float tanh_approx(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// Kimi-K3 SITU activation params (config.json text_config). The pipelined
// kernel bakes these so its reciprocals and up-clamp fold at compile time; the
// host launch falls back to the runtime scalar kernel if a model differs.
static constexpr float SITU_BETA = 4.0f;
static constexpr float SITU_LINEAR_BETA = 25.0f;
__device__ __forceinline__ float situ_activation(float g, float u, float beta,
                                                 float linear_beta,
                                                 bool clamp_up, float inv_beta,
                                                 float inv_linear_beta) {
  // sigmoid(g) == (1 + tanh(g/2)) / 2, an exact identity, so the whole gate
  // runs on two tanh.approx (2 MUFU) instead of tanh + __expf + reciprocal (3
  // MUFU). The two tanh are independent (better ILP) and there is no divide;
  // error stays the tanh.approx class, negligible under the FP8 quant that
  // follows.
  const float gate_out = (0.5f * beta) * tanh_approx(g * inv_beta) *
                         (1.0f + tanh_approx(g * 0.5f));
  const float up_out =
      clamp_up ? linear_beta * tanh_approx(u * inv_linear_beta) : u;
  return gate_out * up_out;
}

template <typename scalar_t>
__global__ void situ_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d, const float beta, const float linear_beta,
    const int64_t* __restrict__ valid_rows_ptr) {
  const int64_t row = blockIdx.x;
  if (valid_rows_ptr != nullptr && row >= *valid_rows_ptr) return;
  const scalar_t* gate_ptr = input + row * 2 * d;
  const scalar_t* up_ptr = gate_ptr + d;
  scalar_t* out_ptr = out + row * d;
  const bool clamp_up = linear_beta > 0.0f;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const float g = (float)VLLM_LDG(&gate_ptr[idx]);
    const float u = (float)VLLM_LDG(&up_ptr[idx]);
    out_ptr[idx] = (scalar_t)situ_activation(g, u, beta, linear_beta, clamp_up,
                                             inv_beta, inv_linear_beta);
  }
}

// Saturating float -> fp8 conversion. `inv_scale = 1 / scale`, so this computes
// clamp(val / scale) and casts (round-to-nearest-even) to fp8. `fp8_max` is the
// representable max for the target fp8 type (448 for e4m3fn), passed from the
// host to avoid device-side numeric_limits.
// c10::Float8_e4m3fn's `static_cast` operator is a software/bit-manipulation
// implementation of float->fp8 rounding; it does NOT necessarily round ties
// the same way as the hardware cvt instruction (`__nv_cvt_float_to_fp8`,
// PTX `cvt.rn.satfinite.e4m3.f32`) that scaled_fp8_conversion (see
// quantization/w8a8/fp8/common.cuh + nvidia/quant_utils.cuh) uses on SM80+,
// and that Triton's `.to(tl.float8e4nv)` also lowers to -- humming's
// quant_input is a Triton kernel. Using `static_cast` here instead of the
// hardware instruction was the root cause of a ~0.1-0.2%-of-elements
// quantization mismatch against humming's output: on exact or near-exact
// ties between two representable fp8 values, the two implementations
// occasionally round to different buckets.
__device__ __forceinline__ c10::Float8_e4m3fn cvt_fp8_hw(float x) {
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return c10::Float8_e4m3fn(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3),
                            c10::Float8_e4m3fn::from_bits());
#else
  return static_cast<c10::Float8_e4m3fn>(x);
#endif
}

template <typename fp8_type>
__device__ __forceinline__ fp8_type quant_to_fp8(float val, float inv_scale,
                                                 float fp8_max) {
  float x = val * inv_scale;
  x = fmaxf(-fp8_max, fminf(x, fp8_max));
  if constexpr (std::is_same_v<fp8_type, c10::Float8_e4m3fn>) {
    return cvt_fp8_hw(x);
  } else {
    return static_cast<fp8_type>(x);
  }
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
  }
  return v;
}

// Fused SITU + block-FP8 (per-128-group) quant, one warp per group. Persistent
// grid: each block strides over valid rows only, so padding rows past
// *valid_rows are never touched. cp.async stages gate+up into smem per group.
template <typename scalar_t, typename fp8_type, int THREADS, int NUM_STAGES,
          int GROUP_SIZE, int GRID_DIM, int D>
__global__ void situ_and_mul_quant_group_pipelined_kernel(
    fp8_type* __restrict__ out,          // [num_tokens, D]
    float* __restrict__ scale_out,       // [num_tokens, NUM_GROUPS]
    const scalar_t* __restrict__ input,  // [num_tokens, 2, D]
    const int64_t num_rows, const int64_t* __restrict__ valid_rows_ptr) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  static constexpr int NUM_WARPS = THREADS / WARP_SIZE;
  static constexpr int NUM_GROUPS = D / GROUP_SIZE;
  const int64_t row_bound =
      valid_rows_ptr != nullptr ? *valid_rows_ptr : num_rows;

  // Padding rows past *valid_rows are never streamed below; fill their scales
  // with 1.0 so masked-out rows don't feed NaN/Inf into the w2 GEMM. NUM_GROUPS
  // is a multiple of 4, so the run is float4-aligned and exact (scale_out is a
  // tensor base, >=16B aligned).
  static_assert(NUM_GROUPS % 4 == 0,
                "float4 scale fill needs NUM_GROUPS % 4 == 0");
  static constexpr int NG4 = NUM_GROUPS / 4;
  const int64_t pad_start4 = row_bound * NG4;
  const int64_t pad_end4 = num_rows * NG4;
  float4* scale4 = reinterpret_cast<float4*>(scale_out);
  constexpr float4 ones{1.0f, 1.0f, 1.0f, 1.0f};
  for (int64_t i = pad_start4 + (int64_t)blockIdx.x * THREADS + tid;
       i < pad_end4; i += (int64_t)GRID_DIM * THREADS) {
    scale4[i] = ones;
  }

  // Vectorized loads/stores assume a 2-byte scalar_t; the fp32 dispatch branch
  // is compiled out here and served by the scalar group kernel instead.
  if constexpr (sizeof(scalar_t) == 2) {
    static constexpr int LD_ELTS = 16 / sizeof(scalar_t);         // 8 (load)
    static constexpr int ELTS_PER_LANE = GROUP_SIZE / WARP_SIZE;  // 4 (compute)
    static constexpr int STAGE_ELTS = 2 * GROUP_SIZE;  // gate+up per stage/warp

    extern __shared__ __align__(16) unsigned char smem_raw[];
    scalar_t* warp_smem = reinterpret_cast<scalar_t*>(smem_raw) +
                          (size_t)warp_id * NUM_STAGES * STAGE_ELTS;

    // beta/linear_beta are baked (Kimi-K3): every reciprocal and the up-clamp
    // fold at compile time. The host gates this launch on the runtime values
    // matching, else falls back to the scalar kernel.
    static constexpr float beta = SITU_BETA;
    static constexpr float linear_beta = SITU_LINEAR_BETA;
    static constexpr bool clamp_up = linear_beta > 0.0f;
    static constexpr float inv_beta = 1.0f / beta;
    static constexpr float inv_linear_beta =
        clamp_up ? 1.0f / linear_beta : 0.0f;
    // fp8_max is fixed by the output type -> compile-time reciprocal, no RCP.
    static constexpr float FP8_MAX =
        std::is_same_v<fp8_type, c10::Float8_e4m3fn> ? 448.0f : 224.0f;
    static constexpr float inv_fp8_max = 1.0f / FP8_MAX;

    // Groups per warp. When NUM_GROUPS splits evenly across warps every warp
    // does the same count, so this folds to a constant (no per-warp register,
    // compile-time loop bound). D=SITU_D satisfies this.
    static_assert(
        NUM_GROUPS % NUM_WARPS == 0,
        "constexpr num_iters requires groups evenly split across warps");
    static constexpr int num_iters = NUM_GROUPS / NUM_WARPS;

    // Per-lane load offsets, hoisted out of the loops: lanes 0..15 take the
    // gate half, 16..31 the up half (+D), each a 16B cp.async. Only the base
    // bumps.
    const bool up_half = lane_id >= WARP_SIZE / 2;
    const int lane_l = up_half ? lane_id - WARP_SIZE / 2 : lane_id;
    const int lane_src_off = (up_half ? D : 0) + lane_l * LD_ELTS;
    const int lane_dst_off = (up_half ? GROUP_SIZE : 0) + lane_l * LD_ELTS;
    static constexpr int warp_stride = NUM_WARPS * GROUP_SIZE;

    // Invariant per-lane source base; the row term folds into the running
    // pointer so the loop bumps by a compile-time stride instead of
    // recomputing row * 2 * D each iteration. All offsets fit int32 (stride
    // GRID_DIM*2*D ~6.5M); the pointer stays 64-bit, so num_tokens is
    // unbounded.
    const scalar_t* src_ptr = input + warp_id * GROUP_SIZE + lane_src_off;
    static constexpr int src_row_stride = GRID_DIM * 2 * D;
    const scalar_t* row_src = src_ptr + blockIdx.x * 2 * D;

    // Same reduction for the store bases (out uint32 view = 4 fp8/lane, and
    // scale). Per-row strides fit int32 (out ~0.8M, scale ~25K); ptrs stay 64b.
    uint32_t* out_ptr =
        reinterpret_cast<uint32_t*>(out + warp_id * GROUP_SIZE) + lane_id;
    static constexpr int out_row_stride = GRID_DIM * D / 4;
    uint32_t* row_out = out_ptr + blockIdx.x * (D / 4);
    float* scale_ptr = scale_out + warp_id;
    static constexpr int scale_row_stride = GRID_DIM * NUM_GROUPS;
    float* row_scale = scale_ptr + blockIdx.x * NUM_GROUPS;

    // Persistent outer loop kept sequential (nounroll): it wraps the whole
    // pipeline, and GRID_DIM is compile-time so the stride folds to a constant.
#pragma unroll 1
    for (int64_t row = blockIdx.x; row < row_bound; row += GRID_DIM,
                 row_src += src_row_stride, row_out += out_row_stride,
                 row_scale += scale_row_stride) {
      // Per-lane source for this row; issue_load bumps `src` by warp_stride.
      const scalar_t* src = row_src;

      // Stage the next group into `slot` (rotating 0..NUM_STAGES-1). Called
      // once per group in increasing order, so the bumping load pointer stays
      // in step.
      auto issue_load = [&](int it, int slot) {
        if (it < num_iters) {
          cuda_async::cp_async_shared_global_16_cg(
              warp_smem + (size_t)slot * STAGE_ELTS + lane_dst_off, src);
          src += warp_stride;
        }
        cuda_async::cp_async_commit_group();
      };

      int load_slot = 0;
      auto bump = [](int s) { return s + 1 == NUM_STAGES ? 0 : s + 1; };
#pragma unroll
      for (int s = 0; s < NUM_STAGES - 1; s++) {
        issue_load(s, load_slot);
        load_slot = bump(load_slot);
      }

      // Compute/store bases for this row (bumped per group below).
      uint32_t* out_st = row_out;
      float* scale_st = row_scale;

      int comp_slot = 0;
      for (int it = 0; it < num_iters; it++) {
        issue_load(it + NUM_STAGES - 1, load_slot);
        load_slot = bump(load_slot);
        cuda_async::cp_async_wait_group<NUM_STAGES - 1>();

        const scalar_t* stage = warp_smem + (size_t)comp_slot * STAGE_ELTS;
        comp_slot = bump(comp_slot);
        // Lane L owns 4 contiguous group elements {4L..4L+3}: one 64-bit smem
        // load (float2) each for gate and up, vs 4x 32-bit before. lane L ->
        // word L is the conflict-free 64-bit pattern, and float2 is 8B-aligned
        // (stage is 512B-aligned, up half at +256B).
        static_assert(ELTS_PER_LANE == 4,
                      "expects GROUP_SIZE == 4 * WARP_SIZE");
        union V {
          float2 f2;
          scalar_t s[ELTS_PER_LANE];
        };
        const float2* gate2 = reinterpret_cast<const float2*>(stage);
        const float2* up2 = reinterpret_cast<const float2*>(stage + GROUP_SIZE);
        const V gv{gate2[lane_id]};
        const V uv{up2[lane_id]};
        const scalar_t* gs = gv.s;
        const scalar_t* us = uv.s;

        float acts[ELTS_PER_LANE];
        float thread_max = 0.0f;
#pragma unroll
        for (int e = 0; e < ELTS_PER_LANE; e++) {
          acts[e] = (float)(scalar_t)situ_activation(
              (float)gs[e], (float)us[e], beta, linear_beta, clamp_up, inv_beta,
              inv_linear_beta);
          thread_max = fmaxf(thread_max, fabsf(acts[e]));
        }
        const float absmax = fmaxf(warp_reduce_max(thread_max), 1e-30f);
        const float scale = absmax * inv_fp8_max;
        if (lane_id == 0) *scale_st = scale;
        scale_st += NUM_WARPS;
        const float inv_scale = __fdividef(1.0f, scale);

        // 4 contiguous fp8 outputs -> one 32-bit coalesced store: lanes 0..31
        // write out[0..127] of the group as 128 contiguous bytes.
        union O {
          uint32_t u;
          fp8_type f[ELTS_PER_LANE];
        };
        O o;
#pragma unroll
        for (int e = 0; e < ELTS_PER_LANE; e++) {
          o.f[e] = quant_to_fp8<fp8_type>(acts[e], inv_scale, FP8_MAX);
        }
        out_st[0] = o.u;
        out_st += NUM_WARPS * GROUP_SIZE / 4;
      }
      // Drain before reusing smem slots for the next row.
      cuda_async::cp_async_wait_group<0>();
    }
  }
}

// Scalar fallback for the group path (odd d or the otherwise-unreachable fp32
// dispatch branch). Persistent grid; each block strides over valid rows, and
// within a row warp `w` owns groups w, w+num_warps, ... (GROUP_SIZE % 32 == 0).
template <typename scalar_t, typename fp8_type, int GROUP_SIZE>
__global__ void situ_and_mul_quant_group_scalar_kernel(
    fp8_type* __restrict__ out,          // [num_tokens, d]
    float* __restrict__ scale_out,       // [num_tokens, num_groups]
    const scalar_t* __restrict__ input,  // [num_tokens, 2, d]
    const int d, const int num_groups, const float beta,
    const float linear_beta, const int64_t num_rows,
    const int64_t* __restrict__ valid_rows_ptr) {
  static constexpr int ELTS_PER_LANE = GROUP_SIZE / WARP_SIZE;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;
  const int64_t row_bound =
      valid_rows_ptr != nullptr ? *valid_rows_ptr : num_rows;

  const bool clamp_up = linear_beta > 0.0f;
  // __fdividef reciprocal: single MUFU.RCP, no IEEE-div FCHK/CALL slow path
  // (~2 ulp, folds into the pending accuracy eval).
  const float inv_beta = __fdividef(1.0f, beta);
  const float inv_linear_beta = clamp_up ? __fdividef(1.0f, linear_beta) : 0.0f;
  // fp8_max is fixed by the output type -> compile-time reciprocal, no RCP.
  static constexpr float FP8_MAX =
      std::is_same_v<fp8_type, c10::Float8_e4m3fn> ? 448.0f : 224.0f;
  static constexpr float inv_fp8_max = 1.0f / FP8_MAX;

  for (int64_t row = blockIdx.x; row < row_bound; row += gridDim.x) {
    const scalar_t* gate_ptr = input + row * 2 * (int64_t)d;
    const scalar_t* up_ptr = gate_ptr + d;
    fp8_type* out_ptr = out + row * (int64_t)d;
    float* scale_row = scale_out + row * (int64_t)num_groups;

    for (int g = warp_id; g < num_groups; g += num_warps) {
      const int base = g * GROUP_SIZE;
      float acts[ELTS_PER_LANE];
      float thread_max = 0.0f;
#pragma unroll
      for (int e = 0; e < ELTS_PER_LANE; e++) {
        const int idx = base + e * WARP_SIZE + lane_id;
        const float gv = (float)VLLM_LDG(&gate_ptr[idx]);
        const float uv = (float)VLLM_LDG(&up_ptr[idx]);
        acts[e] = (float)(scalar_t)situ_activation(
            gv, uv, beta, linear_beta, clamp_up, inv_beta, inv_linear_beta);
        thread_max = fmaxf(thread_max, fabsf(acts[e]));
      }
      const float absmax = fmaxf(warp_reduce_max(thread_max), 1e-30f);
      const float scale = absmax * inv_fp8_max;
      if (lane_id == 0) scale_row[g] = scale;
      const float inv_scale = __fdividef(1.0f, scale);
#pragma unroll
      for (int e = 0; e < ELTS_PER_LANE; e++) {
        const int idx = base + e * WARP_SIZE + lane_id;
        out_ptr[idx] = quant_to_fp8<fp8_type>(acts[e], inv_scale, FP8_MAX);
      }
    }
  }

  // Padding rows past *valid_rows are never streamed above; fill their scales
  // with a finite value so masked-out rows don't feed NaN/Inf into the w2 GEMM.
  const int64_t pad_start = row_bound * (int64_t)num_groups;
  const int64_t pad_end = num_rows * (int64_t)num_groups;
  for (int64_t i = pad_start + (int64_t)blockIdx.x * blockDim.x + tid;
       i < pad_end; i += (int64_t)gridDim.x * blockDim.x) {
    scale_out[i] = 1.0f;
  }
}

template <typename scalar_t>
__global__ void masked_situ_and_mul_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const int* __restrict__ expert_num_tokens, const int max_num_tokens,
    const int d, const float beta, const float linear_beta) {
  const int expert = blockIdx.y;
  const int num_tokens = expert_num_tokens[expert];
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= d || num_tokens == 0) {
    return;
  }

  const bool clamp_up = linear_beta > 0.0f;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
  const int64_t expert_row = static_cast<int64_t>(expert) * max_num_tokens;
  for (int token = 0; token < num_tokens; ++token) {
    const int64_t row = expert_row + token;
    const scalar_t* gate_ptr = input + row * 2 * d;
    const scalar_t* up_ptr = gate_ptr + d;
    scalar_t* out_ptr = out + row * d;
    const float g = (float)VLLM_LDG(&gate_ptr[idx]);
    const float u = (float)VLLM_LDG(&up_ptr[idx]);
    out_ptr[idx] = (scalar_t)situ_activation(g, u, beta, linear_beta, clamp_up,
                                             inv_beta, inv_linear_beta);
  }
}

}  // namespace vllm

#define LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(KERNEL, PACKED_KERNEL, PARAM) \
  auto dtype = input.scalar_type();                                            \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  if (num_tokens == 0) {                                                       \
    return;                                                                    \
  }                                                                            \
  dim3 grid(num_tokens);                                                       \
  int cc_major = get_device_prop()->major;                                     \
  int support_vec =                                                            \
      (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128)            \
          ? vllm::VecTraits<true>::ARCH_MAX_VEC_SIZE                           \
          : vllm::VecTraits<false>::ARCH_MAX_VEC_SIZE;                         \
  int vec_size = support_vec / input.element_size();                           \
  const bool use_vec = (d % vec_size == 0);                                    \
  const torch::stable::accelerator::DeviceGuard device_guard(                  \
      input.get_device_index());                                               \
  const cudaStream_t stream = get_current_cuda_stream();                       \
  if (use_vec) {                                                               \
    dim3 block(std::min(d / vec_size, 1024));                                  \
    if (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128) {         \
      VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                     \
          dtype, "act_and_mul_kernel_with_param", [&] {                        \
            vllm::act_and_mul_kernel_with_param<                               \
                scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,  \
                KERNEL<scalar_t>,                                              \
                PACKED_KERNEL<                                                 \
                    typename vllm::PackedTypeConverter<scalar_t>::Type>,       \
                true, true><<<grid, block, 0, stream>>>(                       \
                out.mutable_data_ptr<scalar_t>(),                              \
                input.const_data_ptr<scalar_t>(), d, PARAM);                   \
          });                                                                  \
    } else {                                                                   \
      VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                     \
          dtype, "act_and_mul_kernel_with_param", [&] {                        \
            vllm::act_and_mul_kernel_with_param<                               \
                scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,  \
                KERNEL<scalar_t>,                                              \
                PACKED_KERNEL<                                                 \
                    typename vllm::PackedTypeConverter<scalar_t>::Type>,       \
                true, false><<<grid, block, 0, stream>>>(                      \
                out.mutable_data_ptr<scalar_t>(),                              \
                input.const_data_ptr<scalar_t>(), d, PARAM);                   \
          });                                                                  \
    }                                                                          \
  } else {                                                                     \
    dim3 block(std::min(d, 1024));                                             \
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                       \
        dtype, "act_and_mul_kernel_with_param", [&] {                          \
          vllm::act_and_mul_kernel_with_param<                                 \
              scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,    \
              KERNEL<scalar_t>,                                                \
              PACKED_KERNEL<                                                   \
                  typename vllm::PackedTypeConverter<scalar_t>::Type>,         \
              false><<<grid, block, 0, stream>>>(                              \
              out.mutable_data_ptr<scalar_t>(),                                \
              input.const_data_ptr<scalar_t>(), d, PARAM);                     \
        });                                                                    \
  }

#define LAUNCH_SIGLUOAI_AND_MUL(KERNEL, ALPHA, LIMIT)                         \
  int d = input.size(-1) / 2;                                                 \
  int64_t num_tokens = input.numel() / input.size(-1);                        \
  dim3 grid(num_tokens);                                                      \
  dim3 block(std::min(d, 1024));                                              \
  const torch::stable::accelerator::DeviceGuard device_guard(                 \
      input.get_device_index());                                              \
  const cudaStream_t stream = get_current_cuda_stream();                      \
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
  LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(
      vllm::fatrelu_kernel, vllm::packed_fatrelu_kernel, threshold);
}
void swigluoai_and_mul(torch::stable::Tensor& out,    // [..., d]
                       torch::stable::Tensor& input,  // [..., 2 * d]
                       double alpha, double limit) {
  LAUNCH_SIGLUOAI_AND_MUL(vllm::swigluoai_and_mul, alpha, limit);
}

// Kimi SITU gated activation. `linear_beta <= 0` means "unset" (up passed
// through), matching SituAndMul(linear_beta=None) on the Python side.
void situ_and_mul(torch::stable::Tensor& out,    // [..., d]
                  torch::stable::Tensor& input,  // [..., 2 * d]
                  double beta, double linear_beta,
                  std::optional<torch::stable::Tensor> valid_rows) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  if (num_tokens == 0) {
    return;
  }
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const int64_t* valid_rows_ptr = nullptr;
  if (valid_rows.has_value()) {
    valid_rows_ptr = valid_rows->const_data_ptr<int64_t>();
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "situ_and_mul_kernel", [&] {
        vllm::situ_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.mutable_data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(),
            d, (float)beta, (float)linear_beta, valid_rows_ptr);
      });
}

void masked_situ_and_mul(torch::stable::Tensor& out,    // [E, T, d]
                         torch::stable::Tensor& input,  // [E, T, 2 * d]
                         const torch::stable::Tensor& expert_num_tokens,
                         double beta, double linear_beta) {
  int num_experts = input.size(0);
  int max_num_tokens = input.size(1);
  int d = input.size(2) / 2;
  if (num_experts == 0 || max_num_tokens == 0) {
    return;
  }
  constexpr int block_size = 256;
  dim3 grid((d + block_size - 1) / block_size, num_experts);
  dim3 block(block_size);
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "masked_situ_and_mul_kernel", [&] {
        vllm::masked_situ_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.mutable_data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(),
            expert_num_tokens.const_data_ptr<int>(), max_num_tokens, d,
            (float)beta, (float)linear_beta);
      });
}

// Fused Kimi SITU activation + dynamic FP8 quantization. Produces the fp8
// down-projection input (`out`) and its scale (`scale`, dequant = q * scale) in
// one pass, replacing the separate situ_and_mul + quant_input kernels on the
// Humming w2 path. `group_size == 128` selects k-major block-FP8 group scales
// (scale [.., d / 128], matching humming quant_input(group_size=128, float32)).
// `linear_beta <= 0` means "unset" (up passed through), matching
// SituAndMul(linear_beta=None). `valid_rows` (int64 scalar tensor) is the
// DeepEP v2 contiguous-layout valid row count; padding rows are skipped and
// receive a benign scale.
void situ_and_mul_quant(torch::stable::Tensor& out,    // [..., d]  (fp8)
                        torch::stable::Tensor& scale,  // [..., 1 or d/128]
                        torch::stable::Tensor& input,  // [..., 2 * d]
                        double beta, double linear_beta, int64_t group_size,
                        std::optional<torch::stable::Tensor> valid_rows) {
  STD_TORCH_CHECK(
      out.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn ||
          out.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fnuz,
      "situ_and_mul_quant output must be FP8 e4m3");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "situ_and_mul_quant input must be FP16 or BF16");
  STD_TORCH_CHECK(scale.scalar_type() == torch::headeronly::ScalarType::Float,
                  "situ_and_mul_quant scale must be float32");
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  if (num_tokens == 0) {
    return;
  }
  const int64_t* valid_rows_ptr = nullptr;
  if (valid_rows.has_value()) {
    valid_rows_ptr = valid_rows->const_data_ptr<int64_t>();
  }
  // fp8_max (448 e4m3fn / 224 fnuz) is derived from fp8_type inside the kernel.
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  static constexpr int THREADS = 256;
  static constexpr int GROUP_STAGES = 4;   // warp-per-group cp.async depth
  static constexpr int BLOCKS_PER_SM = 8;  // matches kernel __launch_bounds__
  static constexpr int SM_COUNT = 132;     // H200 (GH100, 132 SMs)
  // Fixed persistent grid so the kernel's grid stride is a compile-time
  // constant.
  static constexpr int GRID_DIM = SM_COUNT * BLOCKS_PER_SM;
  static constexpr int SITU_D = 3072;  // fixed Kimi-K3 hidden dim (fused w2 in)

  // Block-FP8 group path only: k-major float32 group scales [num_tokens,
  // d / group_size], matching humming quant_input(group_size, float32).
  STD_TORCH_CHECK(group_size == 128,
                  "situ_and_mul_quant: only group_size 128 (block-FP8) "
                  "supported, got ",
                  group_size);
  STD_TORCH_CHECK(d % group_size == 0, "situ_and_mul_quant: d (", d,
                  ") must be divisible by group_size ", group_size);
  {
    const int num_groups = d / (int)group_size;
    // Persistent grid: always launch the fixed GRID_DIM block pool so the
    // kernel's grid stride is a constant; blocks past num_tokens just no-op.
    dim3 grid(GRID_DIM);
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "situ_and_mul_quant_group_kernel", [&] {
          VLLM_STABLE_DISPATCH_FP8_TYPES(
              out.scalar_type(), "situ_and_mul_quant_group_kernel_fp8", [&] {
                // Warp-per-group pipelined kernel: one warp owns a whole
                // 128-group, so the full-warp abs-max reduction is always safe
                // (no partial-warp mask hazard) and any d % 128 == 0 works.
                // Only 2-byte scalar_t is vectorized; the fp32 dispatch branch
                // falls back to the scalar group kernel. Pipelined path only
                // for the fixed hidden dim; other d (and the fp32 branch) fall
                // through to the runtime scalar kernel.
                if constexpr (sizeof(scalar_t) == 2) {
                  if (d == SITU_D && (float)beta == vllm::SITU_BETA &&
                      (float)linear_beta == vllm::SITU_LINEAR_BETA) {
                    constexpr int D = SITU_D;
                    constexpr int NUM_WARPS = THREADS / 32;
                    constexpr int STAGE_ELTS = 2 * 128;  // gate + up per group
                    auto kernel =
                        &vllm::situ_and_mul_quant_group_pipelined_kernel<
                            scalar_t, fp8_t, THREADS, GROUP_STAGES, 128,
                            GRID_DIM, D>;
                    size_t smem_bytes = (size_t)NUM_WARPS * GROUP_STAGES *
                                        STAGE_ELTS * sizeof(scalar_t);
                    dim3 block(THREADS);
                    kernel<<<grid, block, smem_bytes, stream>>>(
                        out.mutable_data_ptr<fp8_t>(),
                        scale.mutable_data_ptr<float>(),
                        input.const_data_ptr<scalar_t>(), num_tokens,
                        valid_rows_ptr);
                    return;
                  }
                }
                const int num_warps = std::min(num_groups, 32);
                dim3 block(num_warps * 32);
                vllm::situ_and_mul_quant_group_scalar_kernel<scalar_t, fp8_t,
                                                             128>
                    <<<grid, block, 0, stream>>>(
                        out.mutable_data_ptr<fp8_t>(),
                        scale.mutable_data_ptr<float>(),
                        input.const_data_ptr<scalar_t>(), d, num_groups,
                        (float)beta, (float)linear_beta, num_tokens,
                        valid_rows_ptr);
              });
        });
  }
}
namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), bool use_vec,
          bool use_256b = false>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  const scalar_t* in_ptr = input + token_idx * d;
  scalar_t* out_ptr = out + token_idx * d;

  if constexpr (use_vec) {
    // Fast path: 128-bit/256-bit vectorized loop
    using vec_t = typename VecTraits<use_256b>::vec_t;
    constexpr int ARCH_MAX_VEC_SIZE = VecTraits<use_256b>::ARCH_MAX_VEC_SIZE;
    constexpr int VEC_SIZE = ARCH_MAX_VEC_SIZE / sizeof(scalar_t);
    const vec_t* in_vec = reinterpret_cast<const vec_t*>(in_ptr);
    vec_t* out_vec = reinterpret_cast<vec_t*>(out_ptr);
    const int num_vecs = d / VEC_SIZE;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      vec_t v;
      if constexpr (use_256b) {
        ld256(v, &in_vec[i]);
      } else {
        v = VLLM_LDG(&in_vec[i]);
      }
      auto* vp = reinterpret_cast<scalar_t*>(&v);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        vp[j] = ACT_FN(vp[j]);
      }
      if constexpr (use_256b) {
        st256(v, &out_vec[i]);
      } else {
        out_vec[i] = v;
      }
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

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                       \
  auto dtype = input.scalar_type();                                            \
  int d = input.size(-1);                                                      \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  if (num_tokens == 0) {                                                       \
    return;                                                                    \
  }                                                                            \
  dim3 grid(num_tokens);                                                       \
  int cc_major = get_device_prop()->major;                                     \
  int support_vec =                                                            \
      (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128)            \
          ? vllm::VecTraits<true>::ARCH_MAX_VEC_SIZE                           \
          : vllm::VecTraits<false>::ARCH_MAX_VEC_SIZE;                         \
  int vec_size = support_vec / input.element_size();                           \
  const bool use_vec = (d % vec_size == 0);                                    \
  const torch::stable::accelerator::DeviceGuard device_guard(                  \
      input.get_device_index());                                               \
  const cudaStream_t stream = get_current_cuda_stream();                       \
  if (use_vec) {                                                               \
    dim3 block(std::min(d / vec_size, 1024));                                  \
    if (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128) {         \
      VLLM_STABLE_DISPATCH_FLOATING_TYPES(dtype, "activation_kernel", [&] {    \
        vllm::activation_kernel<scalar_t, KERNEL<scalar_t>, true, true>        \
            <<<grid, block, 0, stream>>>(out.mutable_data_ptr<scalar_t>(),     \
                                         input.const_data_ptr<scalar_t>(), d); \
      });                                                                      \
    } else {                                                                   \
      VLLM_STABLE_DISPATCH_FLOATING_TYPES(dtype, "activation_kernel", [&] {    \
        vllm::activation_kernel<scalar_t, KERNEL<scalar_t>, true, false>       \
            <<<grid, block, 0, stream>>>(out.mutable_data_ptr<scalar_t>(),     \
                                         input.const_data_ptr<scalar_t>(), d); \
      });                                                                      \
    }                                                                          \
  } else {                                                                     \
    dim3 block(std::min(d, 1024));                                             \
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(dtype, "activation_kernel", [&] {      \
      vllm::activation_kernel<scalar_t, KERNEL<scalar_t>, false>               \
          <<<grid, block, 0, stream>>>(out.mutable_data_ptr<scalar_t>(),       \
                                       input.const_data_ptr<scalar_t>(), d);   \
    });                                                                        \
  }

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

template <typename T>
__device__ __forceinline__ T relu_squared_kernel(const T& x) {
  // relu(x)^2 — introduced in https://arxiv.org/abs/2109.08668v2
  const float f = (float)x;
  const float val = f > 0.0f ? f : 0.0f;
  return (T)(val * val);
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

void relu_squared(torch::stable::Tensor& out,    // [..., d]
                  torch::stable::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::relu_squared_kernel);
}
