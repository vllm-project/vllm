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
// below (situ_and_mul_kernel, situ_and_mul_quant_scalar_kernel,
// situ_and_mul_quant_pipelined_kernel). Deliberately __noinline__: CUDA does
// NOT guarantee bit-identical codegen for the "same" formula written out (or
// even __forceinline__'d) in different call contexts -- FMA contraction can
// fuse a caller's surrounding multiply/add into an inlined callee differently
// depending on what's around it. A version of this that duplicated the
// formula inline diverged from situ_and_mul_kernel's output by 1 ULP on a
// small fraction of elements (~0.1-0.2%), enough to occasionally flip which
// fp8 bucket a value quantizes to; switching the shared helper from
// __forceinline__ to __noinline__ (below) is what actually made it bit-exact
// -- __forceinline__ alone only changed which elements diverged, not whether
// any did, confirming the divergence came from FMA contraction across the
// inline boundary rather than from the formula itself.
__device__ __noinline__ float situ_activation(float g, float u, float beta,
                                               float linear_beta,
                                               bool clamp_up, float inv_beta,
                                               float inv_linear_beta) {
  const float gate_out = beta * tanhf(g * inv_beta) / (1.0f + expf(-g));
  const float up_out =
      clamp_up ? linear_beta * tanhf(u * inv_linear_beta) : u;
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

// Fused SITU (Kimi SituGLU) activation + per-token dynamic FP8 quantization.
// Scalar fallback: one block per row, plain strided element-at-a-time loads,
// full-block shared-memory tree reduction for the per-row abs-max. Used only
// when `d` isn't a multiple of VEC_ELTS (see situ_and_mul_quant_pipelined_kernel
// below, which every real MoE shape hits in practice). Padding rows
// (row >= *valid_rows_ptr) from the DeepEP v2 contiguous layout are never read
// by the down GEMM: leave `out` untouched and store a benign scale.
template <typename scalar_t, typename fp8_type>
__global__ void situ_and_mul_quant_scalar_kernel(
    fp8_type* __restrict__ out,          // [num_tokens, d]
    float* __restrict__ scale_out,       // [num_tokens] (per-token)
    const scalar_t* __restrict__ input,  // [num_tokens, 2, d]
    const int d, const float beta, const float linear_beta, const float fp8_max,
    const int64_t* __restrict__ valid_rows_ptr) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  if (valid_rows_ptr != nullptr && row >= *valid_rows_ptr) {
    if (tid == 0) scale_out[row] = 1.0f;
    return;
  }

  const scalar_t* gate_ptr = input + row * 2 * d;
  const scalar_t* up_ptr = gate_ptr + d;
  fp8_type* out_ptr = out + row * d;
  const bool clamp_up = linear_beta > 0.0f;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;

  // Pass 1: SITU activation (rounded to scalar_t) + per-row abs-max reduction.
  float thread_max = 0.0f;
  for (int idx = tid; idx < d; idx += nthreads) {
    const float g = (float)VLLM_LDG(&gate_ptr[idx]);
    const float u = (float)VLLM_LDG(&up_ptr[idx]);
    const float act = (float)(scalar_t)situ_activation(
        g, u, beta, linear_beta, clamp_up, inv_beta, inv_linear_beta);
    thread_max = fmaxf(thread_max, fabsf(act));
  }

  __shared__ float s_reduce[1024];
  s_reduce[tid] = thread_max;
  __syncthreads();
  for (int n = nthreads; n > 1;) {
    const int half = (n + 1) >> 1;
    if (tid < n - half) {
      s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + half]);
    }
    __syncthreads();
    n = half;
  }

  const float absmax = fmaxf(s_reduce[0], 1e-30f);
  // humming's calc_scale computes `absmax / 448` as a Triton kernel, where
  // 448 is a compile-time constant; Triton (like most LLVM-based compilers)
  // constant-folds division by a literal into a multiply by the precomputed
  // reciprocal, which is NOT bit-identical to a true division for divisors
  // that aren't exact powers of 2. A true `absmax / fp8_max` here disagreed
  // with humming's output by 1 ULP on ~5% of rows -- rare, but occasionally
  // enough to push a value across a quantization bucket boundary. Matching
  // the reciprocal-multiply form exactly reproduces Triton's rounding.
  const float scale = absmax * (1.0f / fp8_max);
  if (tid == 0) scale_out[row] = scale;
  const float inv_scale = 1.0f / scale;

  // Pass 2: recompute SITU and emit the quantized fp8 down-projection input.
  for (int idx = tid; idx < d; idx += nthreads) {
    const float g = (float)VLLM_LDG(&gate_ptr[idx]);
    const float u = (float)VLLM_LDG(&up_ptr[idx]);
    const float act = (float)(scalar_t)situ_activation(
        g, u, beta, linear_beta, clamp_up, inv_beta, inv_linear_beta);
    out_ptr[idx] = quant_to_fp8<fp8_type>(act, inv_scale, fp8_max);
  }
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
  }
  return v;
}

// Fused SITU + per-token dynamic FP8 quantization: vectorized, cp.async
// double-buffered version used whenever `d` is a multiple of VEC_ELTS (every
// real MoE intermediate size). One block per row.
//
// Because the per-token scale depends on the abs-max of the WHOLE row, this
// can't be collapsed into a single pass the way a per-128-group block-FP8
// kernel can (c.f. silu_mul_fp8_quant_deep_gemm_kernel in
// quantization/activation_kernels.cu, which finalizes each 128-group as soon
// as it lands and never needs to look at the rest of the row). Instead:
// each thread double-buffers its own private stripe of the row via cp.async
// (no cross-thread sync needed on the load side -- a thread only ever
// consumes bytes it issued itself), computes SITU into a per-row shared-
// memory cache `s_act` (so pass 2 never re-reads `input` from global memory
// or recomputes tanh/exp), reduces the abs-max with warp shuffles plus one
// small cross-warp step (a single __syncthreads() for the whole kernel), then
// re-reads `s_act` to emit 128-bit vectorized fp8 stores.
//
// Padding rows (row >= *valid_rows_ptr) are skipped, matching
// situ_and_mul_quant_scalar_kernel above.
template <typename scalar_t, typename fp8_type, int THREADS, int NUM_STAGES>
__global__ void situ_and_mul_quant_pipelined_kernel(
    fp8_type* __restrict__ out,          // [num_tokens, d]
    float* __restrict__ scale_out,       // [num_tokens]
    const scalar_t* __restrict__ input,  // [num_tokens, 2, d]
    const int d, const float beta, const float linear_beta, const float fp8_max,
    const int64_t* __restrict__ valid_rows_ptr) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;

  if (valid_rows_ptr != nullptr && row >= *valid_rows_ptr) {
    if (tid == 0) scale_out[row] = 1.0f;
    return;
  }

  // This kernel's vectorized loads/stores (int4 gate/up loads, int2 fp8
  // stores) assume a 2-byte scalar_t (VEC_ELTS == 8, matching fp8_type's
  // 1-byte width against int2's 8 bytes). VLLM_STABLE_DISPATCH_FLOATING_TYPES
  // also instantiates a `float` (4-byte) branch; the host launcher never
  // actually calls it (situ_and_mul_quant requires Half/BFloat16 input), but
  // it still has to compile, so it's compiled out here via `if constexpr`
  // rather than left to silently miscompute VEC_ELTS-vs-int2 sizing.
  if constexpr (sizeof(scalar_t) == 2) {
    static constexpr int VEC_ELTS = 16 / sizeof(scalar_t);
    static constexpr int NUM_WARPS = THREADS / WARP_SIZE;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    scalar_t* s_gate = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* s_up = s_gate + (size_t)NUM_STAGES * THREADS * VEC_ELTS;
    float* s_act = reinterpret_cast<float*>(
        s_up + (size_t)NUM_STAGES * THREADS * VEC_ELTS);
    __shared__ float s_warp_max[NUM_WARPS];

    const scalar_t* gate_ptr = input + row * 2 * (int64_t)d;
    const scalar_t* up_ptr = gate_ptr + d;
    fp8_type* out_ptr = out + row * (int64_t)d;

    const int vec_count = d / VEC_ELTS;
    const int tail_start = vec_count * VEC_ELTS;
    const int iters = (vec_count + THREADS - 1) / THREADS;

    const bool clamp_up = linear_beta > 0.0f;
    const float inv_beta = 1.0f / beta;
    const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;

    auto situ = [&](float g, float u) -> float {
      return (float)(scalar_t)situ_activation(g, u, beta, linear_beta,
                                              clamp_up, inv_beta,
                                              inv_linear_beta);
    };

    // Always commits exactly one group, even when this thread has nothing to
    // load (vidx out of range) -- keeps the commit-order ==
    // logical-load-order invariant that makes a constant
    // wait_group<NUM_STAGES - 1>() correct.
    auto issue_load = [&](int iter) {
      const int slot = iter % NUM_STAGES;
      const int vidx = iter * THREADS + tid;
      if (vidx < vec_count) {
        scalar_t* g_dst =
            s_gate + (size_t)slot * THREADS * VEC_ELTS + tid * VEC_ELTS;
        scalar_t* u_dst =
            s_up + (size_t)slot * THREADS * VEC_ELTS + tid * VEC_ELTS;
        cuda_async::cp_async_shared_global_16_cg(
            g_dst, &gate_ptr[(size_t)vidx * VEC_ELTS]);
        cuda_async::cp_async_shared_global_16_cg(
            u_dst, &up_ptr[(size_t)vidx * VEC_ELTS]);
      }
      cuda_async::cp_async_commit_group();
    };

#pragma unroll
    for (int s = 0; s < NUM_STAGES - 1; s++) issue_load(s);

    float thread_max = 0.0f;
    for (int iter = 0; iter < iters; iter++) {
      issue_load(iter + NUM_STAGES - 1);
      cuda_async::cp_async_wait_group<NUM_STAGES - 1>();
      const int vidx = iter * THREADS + tid;
      if (vidx < vec_count) {
        const int slot = iter % NUM_STAGES;
        const scalar_t* g_src =
            s_gate + (size_t)slot * THREADS * VEC_ELTS + tid * VEC_ELTS;
        const scalar_t* u_src =
            s_up + (size_t)slot * THREADS * VEC_ELTS + tid * VEC_ELTS;
#pragma unroll
        for (int e = 0; e < VEC_ELTS; e++) {
          const float act = situ((float)g_src[e], (float)u_src[e]);
          s_act[(size_t)vidx * VEC_ELTS + e] = act;
          thread_max = fmaxf(thread_max, fabsf(act));
        }
      }
    }

    // Scalar tail (d % VEC_ELTS leftover elements). Dead code on the
    // dispatch path today (the host only launches this kernel when
    // d % VEC_ELTS == 0, required for gate/up 16-byte alignment), kept for
    // robustness.
    for (int idx = tail_start + tid; idx < d; idx += THREADS) {
      const float act =
          situ((float)VLLM_LDG(&gate_ptr[idx]), (float)VLLM_LDG(&up_ptr[idx]));
      s_act[idx] = act;
      thread_max = fmaxf(thread_max, fabsf(act));
    }

    // Reduce: warp shuffle, then one small cross-warp step (one syncthreads
    // total for the whole kernel), replacing the old O(log2 nthreads) full
    // block-wide shared-memory tree.
    thread_max = warp_reduce_max(thread_max);
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) s_warp_max[warp_id] = thread_max;
    __syncthreads();
    float absmax = 0.0f;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; w++) absmax = fmaxf(absmax, s_warp_max[w]);
    absmax = fmaxf(absmax, 1e-30f);
    // humming's calc_scale computes `absmax / 448` as a Triton kernel, where
  // 448 is a compile-time constant; Triton (like most LLVM-based compilers)
  // constant-folds division by a literal into a multiply by the precomputed
  // reciprocal, which is NOT bit-identical to a true division for divisors
  // that aren't exact powers of 2. A true `absmax / fp8_max` here disagreed
  // with humming's output by 1 ULP on ~5% of rows -- rare, but occasionally
  // enough to push a value across a quantization bucket boundary. Matching
  // the reciprocal-multiply form exactly reproduces Triton's rounding.
  const float scale = absmax * (1.0f / fp8_max);
    const float inv_scale = 1.0f / scale;
    if (tid == 0) scale_out[row] = scale;

    // Pass 2: re-read the row cache (no second global read of `input`),
    // vectorized fp8 store.
    for (int iter = 0; iter < iters; iter++) {
      const int vidx = iter * THREADS + tid;
      if (vidx >= vec_count) continue;
      fp8_type vec_out[VEC_ELTS];
#pragma unroll
      for (int e = 0; e < VEC_ELTS; e++) {
        vec_out[e] = quant_to_fp8<fp8_type>(s_act[(size_t)vidx * VEC_ELTS + e],
                                            inv_scale, fp8_max);
      }
      *reinterpret_cast<int2*>(&out_ptr[(size_t)vidx * VEC_ELTS]) =
          *reinterpret_cast<int2*>(vec_out);
    }
    for (int idx = tail_start + tid; idx < d; idx += THREADS) {
      out_ptr[idx] = quant_to_fp8<fp8_type>(s_act[idx], inv_scale, fp8_max);
    }
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

// Fused Kimi SITU activation + per-token dynamic FP8 quantization. Produces the
// fp8 down-projection input (`out`) and its per-token scale (`scale`, dequant =
// q * scale) in one pass, replacing the separate situ_and_mul + quant_input
// kernels on the Humming w2 path. `linear_beta <= 0` means "unset" (up passed
// through), matching SituAndMul(linear_beta=None). `valid_rows` (int64 scalar
// tensor) is the DeepEP v2 contiguous-layout valid row count; padding rows are
// skipped and receive a benign scale.
void situ_and_mul_quant(torch::stable::Tensor& out,    // [..., d]  (fp8)
                        torch::stable::Tensor& scale,  // [..., 1]  (float32)
                        torch::stable::Tensor& input,  // [..., 2 * d]
                        double beta, double linear_beta,
                        std::optional<torch::stable::Tensor> valid_rows) {
  STD_TORCH_CHECK(
      out.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn ||
          out.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fnuz,
      "situ_and_mul_quant output must be FP8 e4m3");
  STD_TORCH_CHECK(input.scalar_type() == torch::headeronly::ScalarType::Half ||
                      input.scalar_type() ==
                          torch::headeronly::ScalarType::BFloat16,
                  "situ_and_mul_quant input must be FP16 or BF16");
  STD_TORCH_CHECK(scale.scalar_type() == torch::headeronly::ScalarType::Float,
                  "situ_and_mul_quant scale must be float32");
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  if (num_tokens == 0) {
    return;
  }
  dim3 grid(num_tokens);
  const int64_t* valid_rows_ptr = nullptr;
  if (valid_rows.has_value()) {
    valid_rows_ptr = valid_rows->const_data_ptr<int64_t>();
  }
  float fp8_max = 448.0f;  // e4m3fn; matches humming calc_scale (absmax / 448)
  if (out.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fnuz) {
    fp8_max = 224.0f;  // ROCm fnuz convention (vllm quant_type_max)
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  static constexpr int THREADS = 256;
  static constexpr int NUM_STAGES = 2;
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "situ_and_mul_quant_kernel", [&] {
        VLLM_STABLE_DISPATCH_FP8_TYPES(
            out.scalar_type(), "situ_and_mul_quant_kernel_fp8", [&] {
              // The pipelined kernel needs gate/up 16-byte aligned per row
              // (d % VEC_ELTS == 0) and only supports 2-byte scalar_t (see
              // the if constexpr guard inside the kernel); every real MoE
              // shape hits this path. Anything else (odd d, or the
              // otherwise-unreachable fp32 dispatch branch) falls back to
              // the scalar kernel.
              constexpr int VEC_ELTS = 16 / sizeof(scalar_t);
              if constexpr (sizeof(scalar_t) == 2) {
                if (d % VEC_ELTS == 0) {
                  auto kernel = &vllm::situ_and_mul_quant_pipelined_kernel<
                      scalar_t, fp8_t, THREADS, NUM_STAGES>;
                  size_t stage_bytes = 2 * (size_t)NUM_STAGES * THREADS *
                                      VEC_ELTS * sizeof(scalar_t);
                  size_t smem_bytes = stage_bytes + (size_t)d * sizeof(float);
                  cudaError_t err = cudaFuncSetAttribute(
                      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                      smem_bytes);
                  STD_TORCH_CHECK(
                      err == cudaSuccess,
                      "situ_and_mul_quant: failed to opt into ", smem_bytes,
                      " bytes of dynamic shared memory: ",
                      cudaGetErrorString(err));
                  dim3 block(THREADS);
                  kernel<<<grid, block, smem_bytes, stream>>>(
                      out.mutable_data_ptr<fp8_t>(),
                      scale.mutable_data_ptr<float>(),
                      input.const_data_ptr<scalar_t>(), d, (float)beta,
                      (float)linear_beta, fp8_max, valid_rows_ptr);
                  return;
                }
              }
              dim3 block(std::min(d, 1024));
              vllm::situ_and_mul_quant_scalar_kernel<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.mutable_data_ptr<fp8_t>(),
                      scale.mutable_data_ptr<float>(),
                      input.const_data_ptr<scalar_t>(), d, (float)beta,
                      (float)linear_beta, fp8_max, valid_rows_ptr);
            });
      });
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
