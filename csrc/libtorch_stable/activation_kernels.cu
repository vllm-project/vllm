#include <cuda.h>
#include <torch/csrc/stable/tensor.h>

#include <cmath>

#include "../cuda_compat.h"
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
template <typename scalar_t>
__global__ void situ_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d, const float beta, const float linear_beta) {
  const int64_t row = blockIdx.x;
  const scalar_t* gate_ptr = input + row * 2 * d;
  const scalar_t* up_ptr = gate_ptr + d;
  scalar_t* out_ptr = out + row * d;
  const bool clamp_up = linear_beta > 0.0f;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const float g = (float)VLLM_LDG(&gate_ptr[idx]);
    const float u = (float)VLLM_LDG(&up_ptr[idx]);
    const float gate_out = beta * tanhf(g * inv_beta) / (1.0f + expf(-g));
    const float up_out =
        clamp_up ? linear_beta * tanhf(u * inv_linear_beta) : u;
    out_ptr[idx] = (scalar_t)(gate_out * up_out);
  }
}

template <bool BATCHED_EXPERTS>
__device__ __forceinline__ bool get_masked_row_range(
    const int* __restrict__ valid_token_counts, const int max_num_tokens,
    int64_t& first_row, int64_t& end_row) {
  if constexpr (BATCHED_EXPERTS) {
    // [E, T, *]: each block handles one expert's valid token prefix.
    const int expert = blockIdx.y;
    const int num_tokens =
        max(0, min(valid_token_counts[expert], max_num_tokens));
    first_row = static_cast<int64_t>(expert) * max_num_tokens;
    end_row = first_row + num_tokens;
  } else {
    // [T, *]: each block handles one row, masked by a single valid prefix.
    const int row = blockIdx.x;
    const int num_tokens = max(0, min(valid_token_counts[0], max_num_tokens));
    if (row >= num_tokens) {
      return false;
    }
    first_row = row;
    end_row = row + 1;
  }
  return first_row < end_row;
}

template <bool BATCHED_EXPERTS>
__device__ __forceinline__ int get_masked_feature_index() {
  // Feature tiles occupy x for batched layouts and y for flat layouts.
  if constexpr (BATCHED_EXPERTS) {
    return blockIdx.x * blockDim.x + threadIdx.x;
  } else {
    return blockIdx.y * blockDim.x + threadIdx.x;
  }
}

template <typename scalar_t, bool BATCHED_EXPERTS>
__global__ void masked_situ_and_mul_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const int* __restrict__ valid_token_counts, const int max_num_tokens,
    const int d, const float beta, const float linear_beta) {
  int64_t first_row, end_row;
  const int idx = get_masked_feature_index<BATCHED_EXPERTS>();
  if (idx >= d || !get_masked_row_range<BATCHED_EXPERTS>(
                      valid_token_counts, max_num_tokens, first_row, end_row)) {
    return;
  }

  const bool clamp_up = linear_beta > 0.0f;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
  for (int64_t row = first_row; row < end_row; ++row) {
    const scalar_t* gate_ptr = input + row * 2 * d;
    const scalar_t* up_ptr = gate_ptr + d;
    scalar_t* out_ptr = out + row * d;
    const float g = (float)VLLM_LDG(&gate_ptr[idx]);
    const float u = (float)VLLM_LDG(&up_ptr[idx]);
    const float gate_out = beta * tanhf(g * inv_beta) / (1.0f + expf(-g));
    const float up_out =
        clamp_up ? linear_beta * tanhf(u * inv_linear_beta) : u;
    out_ptr[idx] = (scalar_t)(gate_out * up_out);
  }
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&, const float),
          bool act_first, bool HAS_CLAMP, bool BATCHED_EXPERTS>
__global__ void masked_act_and_mul_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const int* __restrict__ valid_token_counts, const int max_num_tokens,
    const int d, const float limit, const float alpha, const float beta) {
  int64_t first_row, end_row;
  const int idx = get_masked_feature_index<BATCHED_EXPERTS>();
  if (idx >= d || !get_masked_row_range<BATCHED_EXPERTS>(
                      valid_token_counts, max_num_tokens, first_row, end_row)) {
    return;
  }

  for (int64_t row = first_row; row < end_row; ++row) {
    const scalar_t* x_ptr = input + row * 2 * d;
    const scalar_t* y_ptr = x_ptr + d;
    scalar_t* out_ptr = out + row * d;
    const scalar_t x = VLLM_LDG(&x_ptr[idx]);
    const scalar_t y = VLLM_LDG(&y_ptr[idx]);
    out_ptr[idx] = compute<scalar_t, ACT_FN, act_first, HAS_CLAMP>(x, y, limit,
                                                                   alpha, beta);
  }
}

template <typename scalar_t, bool BATCHED_EXPERTS>
__global__ void masked_swigluoai_and_mul_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const int* __restrict__ valid_token_counts, const int max_num_tokens,
    const int d, const float alpha, const float limit) {
  int64_t first_row, end_row;
  const int idx = get_masked_feature_index<BATCHED_EXPERTS>();
  if (idx >= d || !get_masked_row_range<BATCHED_EXPERTS>(
                      valid_token_counts, max_num_tokens, first_row, end_row)) {
    return;
  }

  for (int64_t row = first_row; row < end_row; ++row) {
    const scalar_t* in_ptr = input + row * 2 * d;
    scalar_t* out_ptr = out + row * d;
    out_ptr[idx] =
        swigluoai_and_mul(in_ptr[2 * idx], in_ptr[2 * idx + 1], alpha, limit);
  }
}

template <typename scalar_t, bool BATCHED_EXPERTS>
__global__ void masked_swiglustep_and_mul_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const int* __restrict__ valid_token_counts, const int max_num_tokens,
    const int d, const float limit) {
  int64_t first_row, end_row;
  const int idx = get_masked_feature_index<BATCHED_EXPERTS>();
  if (idx >= d || !get_masked_row_range<BATCHED_EXPERTS>(
                      valid_token_counts, max_num_tokens, first_row, end_row)) {
    return;
  }

  for (int64_t row = first_row; row < end_row; ++row) {
    const scalar_t* gate_ptr = input + row * 2 * d;
    const scalar_t* up_ptr = gate_ptr + d;
    scalar_t* out_ptr = out + row * d;
    const float gate = (float)VLLM_LDG(&gate_ptr[idx]);
    const float up = (float)VLLM_LDG(&up_ptr[idx]);
    const float gate_silu = gate / (1.0f + expf(-gate));
    const float gate_clamped = fminf(gate_silu, limit);
    const float up_clamped = fmaxf(fminf(up, limit), -limit);
    out_ptr[idx] = (scalar_t)(gate_clamped * up_clamped);
  }
}

template <typename T>
__device__ __forceinline__ T silu_no_mul_kernel(const T& x) {
  return silu_kernel(x, 1.0f);
}

template <typename T>
__device__ __forceinline__ T gelu_no_mul_kernel(const T& x) {
  return gelu_kernel(x, 1.0f);
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_no_mul_kernel(const T& x) {
  return gelu_tanh_kernel(x, 1.0f);
}

template <typename T>
__device__ __forceinline__ T relu2_no_mul_kernel(const T& x) {
  const float value = fmaxf((float)x, 0.0f);
  return (T)(value * value);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool BATCHED_EXPERTS>
__global__ void masked_activation_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const int* __restrict__ valid_token_counts, const int max_num_tokens,
    const int d) {
  int64_t first_row, end_row;
  const int idx = get_masked_feature_index<BATCHED_EXPERTS>();
  if (idx >= d || !get_masked_row_range<BATCHED_EXPERTS>(
                      valid_token_counts, max_num_tokens, first_row, end_row)) {
    return;
  }

  for (int64_t row = first_row; row < end_row; ++row) {
    const int64_t offset = row * d + idx;
    out[offset] = ACT_FN(VLLM_LDG(&input[offset]));
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
                  double beta, double linear_beta) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  if (num_tokens == 0) {
    return;
  }
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "situ_and_mul_kernel", [&] {
        vllm::situ_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.mutable_data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(),
            d, (float)beta, (float)linear_beta);
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
        vllm::masked_situ_and_mul_kernel<scalar_t, true>
            <<<grid, block, 0, stream>>>(
                out.mutable_data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(),
                expert_num_tokens.const_data_ptr<int>(), max_num_tokens, d,
                (float)beta, (float)linear_beta);
      });
}

#define LAUNCH_MASKED_ACT_AND_MUL(KERNEL, ACT_FIRST, HAS_CLAMP, LIMIT, ALPHA, \
                                  BETA)                                       \
  vllm::masked_act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST,      \
                                  HAS_CLAMP, BATCHED_EXPERTS>                 \
      <<<grid, block, 0, stream>>>(out.mutable_data_ptr<scalar_t>(),          \
                                   input.const_data_ptr<scalar_t>(),          \
                                   valid_token_counts.const_data_ptr<int>(),  \
                                   max_num_tokens, d, LIMIT, ALPHA, BETA)

#define LAUNCH_MASKED_ACTIVATION(KERNEL)                                      \
  vllm::masked_activation_kernel<scalar_t, KERNEL<scalar_t>, BATCHED_EXPERTS> \
      <<<grid, block, 0, stream>>>(                                           \
          out.mutable_data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(), \
          valid_token_counts.const_data_ptr<int>(), max_num_tokens, d)

void masked_moe_activation(
    torch::stable::Tensor& out,    // [T, d] or [E, T, d]
    torch::stable::Tensor& input,  // [T, d] or [E, T, d], gated has 2 * d
    const torch::stable::Tensor& valid_token_counts,
    const std::string& activation, double clamp_limit, double alpha,
    double beta, double situ_beta, double situ_linear_beta) {
  const bool batched_experts = input.dim() == 3;
  const int num_experts = batched_experts ? input.size(0) : 1;
  const int max_num_tokens = batched_experts ? input.size(1) : input.size(0);
  const int d = out.size(-1);
  if (num_experts == 0 || max_num_tokens == 0) {
    return;
  }

  constexpr int block_size = 256;
  const int feature_blocks = (d + block_size - 1) / block_size;
  // Batched grid: (feature tile, expert); flat grid: (row, feature tile).
  dim3 grid = batched_experts ? dim3(feature_blocks, num_experts)
                              : dim3(max_num_tokens, feature_blocks);
  dim3 block(block_size);
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  auto launch = [&]<bool BATCHED_EXPERTS>() {
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "masked_moe_activation", [&] {
          if (activation == "silu") {
            LAUNCH_MASKED_ACT_AND_MUL(vllm::silu_kernel, true, false, 0.0f,
                                      1.0f, 0.0f);
          } else if (activation == "silu_with_clamp") {
            LAUNCH_MASKED_ACT_AND_MUL(vllm::silu_kernel, true, true,
                                      (float)clamp_limit, (float)alpha,
                                      (float)beta);
          } else if (activation == "gelu") {
            LAUNCH_MASKED_ACT_AND_MUL(vllm::gelu_kernel, true, false, 0.0f,
                                      1.0f, 0.0f);
          } else if (activation == "gelu_tanh") {
            LAUNCH_MASKED_ACT_AND_MUL(vllm::gelu_tanh_kernel, true, false, 0.0f,
                                      1.0f, 0.0f);
          } else if (activation == "situ") {
            vllm::masked_situ_and_mul_kernel<scalar_t, BATCHED_EXPERTS>
                <<<grid, block, 0, stream>>>(
                    out.mutable_data_ptr<scalar_t>(),
                    input.const_data_ptr<scalar_t>(),
                    valid_token_counts.const_data_ptr<int>(), max_num_tokens, d,
                    (float)situ_beta, (float)situ_linear_beta);
          } else if (activation == "swigluoai") {
            vllm::masked_swigluoai_and_mul_kernel<scalar_t, BATCHED_EXPERTS>
                <<<grid, block, 0, stream>>>(
                    out.mutable_data_ptr<scalar_t>(),
                    input.const_data_ptr<scalar_t>(),
                    valid_token_counts.const_data_ptr<int>(), max_num_tokens, d,
                    (float)alpha, (float)clamp_limit);
          } else if (activation == "swigluoai_uninterleave") {
            LAUNCH_MASKED_ACT_AND_MUL(vllm::silu_kernel, true, true,
                                      (float)clamp_limit, (float)alpha,
                                      (float)beta);
          } else if (activation == "swiglustep") {
            vllm::masked_swiglustep_and_mul_kernel<scalar_t, BATCHED_EXPERTS>
                <<<grid, block, 0, stream>>>(
                    out.mutable_data_ptr<scalar_t>(),
                    input.const_data_ptr<scalar_t>(),
                    valid_token_counts.const_data_ptr<int>(), max_num_tokens, d,
                    (float)clamp_limit);
          } else if (activation == "silu_no_mul") {
            LAUNCH_MASKED_ACTIVATION(vllm::silu_no_mul_kernel);
          } else if (activation == "gelu_no_mul") {
            LAUNCH_MASKED_ACTIVATION(vllm::gelu_no_mul_kernel);
          } else if (activation == "gelu_tanh_no_mul") {
            LAUNCH_MASKED_ACTIVATION(vllm::gelu_tanh_no_mul_kernel);
          } else if (activation == "relu2_no_mul") {
            LAUNCH_MASKED_ACTIVATION(vllm::relu2_no_mul_kernel);
          } else {
            STD_TORCH_CHECK(false,
                            "Unsupported masked MoE activation: ", activation);
          }
        });
  };
  if (batched_experts) {
    launch.template operator()<true>();
  } else {
    launch.template operator()<false>();
  }
}

#undef LAUNCH_MASKED_ACT_AND_MUL
#undef LAUNCH_MASKED_ACTIVATION
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
