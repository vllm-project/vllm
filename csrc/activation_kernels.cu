#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

struct alignas(32) u32x8_t {
  uint32_t u0, u1, u2, u3, u4, u5, u6, u7;
};

__device__ __forceinline__ void ld256(u32x8_t& val, const u32x8_t* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("ld.global.nc.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
               : "=r"(val.u0), "=r"(val.u1), "=r"(val.u2), "=r"(val.u3),
                 "=r"(val.u4), "=r"(val.u5), "=r"(val.u6), "=r"(val.u7)
               : "l"(ptr));
#else
  const uint4* uint_ptr = reinterpret_cast<const uint4*>(ptr);
  uint4 top_half = __ldg(&uint_ptr[0]);
  uint4 bottom_half = __ldg(&uint_ptr[1]);
  val.u0 = top_half.x;
  val.u1 = top_half.y;
  val.u2 = top_half.z;
  val.u3 = top_half.w;
  val.u4 = bottom_half.x;
  val.u5 = bottom_half.y;
  val.u6 = bottom_half.z;
  val.u7 = bottom_half.w;
#endif
}

__device__ __forceinline__ void st256(u32x8_t& val, u32x8_t* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n"
               :
               : "l"(ptr), "r"(val.u0), "r"(val.u1), "r"(val.u2), "r"(val.u3),
                 "r"(val.u4), "r"(val.u5), "r"(val.u6), "r"(val.u7)
               : "memory");
#else
  uint4* uint_ptr = reinterpret_cast<uint4*>(ptr);
  uint_ptr[0] = make_uint4(val.u0, val.u1, val.u2, val.u3);
  uint_ptr[1] = make_uint4(val.u4, val.u5, val.u6, val.u7);
#endif
}

template <bool support_256>
struct VecTraits;

template <>
struct VecTraits<true> {
  static constexpr int ARCH_MAX_VEC_SIZE = 32;
  using vec_t = u32x8_t;
};

template <>
struct VecTraits<false> {
  static constexpr int ARCH_MAX_VEC_SIZE = 16;
  using vec_t = int4;
};

template <typename T>
struct PackedTraits;

template <>
struct PackedTraits<c10::BFloat16> {
  using packed_t = __nv_bfloat162;
};

template <>
struct PackedTraits<c10::Half> {
  using packed_t = __half2;
};

template <>
struct PackedTraits<float> {
  using packed_t = float2;
};

template <typename packed_t>
__device__ __forceinline__ float2 cast_to_float2(const packed_t& val) {
  if constexpr (std::is_same_v<packed_t, __nv_bfloat162>) {
    return __bfloat1622float2(val);
  } else if constexpr (std::is_same_v<packed_t, __half2>) {
    return __half22float2(val);
  } else if constexpr (std::is_same_v<packed_t, float2>) {
    return float2(val);
  }
}

template <typename packed_t>
__device__ __forceinline__ packed_t cast_to_packed(const float2& val) {
  if constexpr (std::is_same_v<packed_t, __nv_bfloat162>) {
    return __float22bfloat162_rn(val);
  } else if constexpr (std::is_same_v<packed_t, __half2>) {
    return __float22half2_rn(val);
  } else if constexpr (std::is_same_v<packed_t, float2>) {
    return float2(val);
  }
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_mul(const packed_t& x,
                                               const packed_t& y) {
  if constexpr (std::is_same_v<packed_t, __nv_bfloat162> ||
                std::is_same_v<packed_t, __half2>) {
    return __hmul2(x, y);
  } else if constexpr (std::is_same_v<packed_t, float2>) {
    return make_float2(x.x * y.x, x.y * y.y);
  }
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

template <typename packed_t, packed_t (*PACKED_ACT_FN)(const packed_t&),
          bool act_first>
__device__ __forceinline__ packed_t packed_compute(const packed_t& x,
                                                   const packed_t& y) {
  return act_first ? packed_mul(PACKED_ACT_FN(x), y)
                   : packed_mul(x, PACKED_ACT_FN(y));
}

// Check if all pointers are 16-byte aligned for int4 vectorized access
__host__ __device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

// Check if all pointers are 16-byte aligned for longlong4_32a vectorized access
__host__ __device__ __forceinline__ bool is_32byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 31) == 0;
}

// Activation and gating kernel template.
template <typename scalar_t, typename packed_t,
          scalar_t (*ACT_FN)(const scalar_t&),
          packed_t (*PACKED_ACT_FN)(const packed_t&), bool act_first,
          bool use_vec, bool use_256b = false>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const scalar_t* x_ptr = input + blockIdx.x * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + blockIdx.x * d;

  if constexpr (use_vec) {
    // Fast path: 128-bit/256-bit vectorized loop
    using vec_t = typename VecTraits<use_256b>::vec_t;
    constexpr int ARCH_MAX_VEC_SIZE = VecTraits<use_256b>::ARCH_MAX_VEC_SIZE;
    constexpr int VEC_SIZE = ARCH_MAX_VEC_SIZE / sizeof(packed_t);

    const vec_t* x_vec = reinterpret_cast<const vec_t*>(x_ptr);
    const vec_t* y_vec = reinterpret_cast<const vec_t*>(y_ptr);
    vec_t* out_vec = reinterpret_cast<vec_t*>(out_ptr);
    const int num_vecs = d / 2 / VEC_SIZE;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      vec_t x, y;
      if constexpr (use_256b) {
        ld256(x, &x_vec[i]);
        ld256(y, &y_vec[i]);
      } else {
        x = VLLM_LDG(&x_vec[i]);
        y = VLLM_LDG(&y_vec[i]);
      }
      auto* xp = reinterpret_cast<packed_t*>(&x);
      auto* yp = reinterpret_cast<packed_t*>(&y);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        xp[j] =
            packed_compute<packed_t, PACKED_ACT_FN, act_first>(xp[j], yp[j]);
      }
      if constexpr (use_256b) {
        st256(x, &out_vec[i]);
      } else {
        out_vec[i] = x;
      }
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

template <typename packed_t>
__device__ __forceinline__ packed_t packed_silu_kernel(const packed_t& val) {
  // x * sigmoid(x)
  float2 fval = cast_to_float2(val);
  fval.x = fval.x / (1.0f + expf(fval.x));
  fval.y = fval.y / (1.0f + expf(fval.y));
  return cast_to_packed<packed_t>(fval);
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

template <typename packed_t>
__device__ __forceinline__ packed_t packed_gelu_kernel(const packed_t& val) {
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

template <typename packed_t>
__device__ __forceinline__ packed_t
packed_gelu_tanh_kernel(const packed_t& val) {
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
// first.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, PACKED_KERNEL, ACT_FIRST)     \
  auto dtype = input.scalar_type();                                         \
  int d = input.size(-1) / 2;                                               \
  int64_t num_tokens = input.numel() / input.size(-1);                      \
  if (num_tokens == 0) {                                                    \
    return;                                                                 \
  }                                                                         \
  dim3 grid(num_tokens);                                                    \
  int cc_major = at::cuda::getCurrentDeviceProperties()->major;             \
  int support_vec = (cc_major >= 10 && num_tokens >= 128) ? 32 : 16;        \
  int vec_size = support_vec / at::elementSize(dtype);                      \
  const bool use_vec = (d % vec_size == 0);                                 \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));         \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();             \
  if (use_vec) {                                                            \
    dim3 block(std::min(d / vec_size, 1024));                               \
    if (cc_major >= 10 && num_tokens >= 128) {                              \
      VLLM_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel", [&] {       \
        vllm::act_and_mul_kernel<                                           \
            scalar_t, typename vllm::PackedTraits<scalar_t>::packed_t,      \
            KERNEL<scalar_t>,                                               \
            PACKED_KERNEL<typename vllm::PackedTraits<scalar_t>::packed_t>, \
            ACT_FIRST, true, true><<<grid, block, 0, stream>>>(             \
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);       \
      });                                                                   \
    } else {                                                                \
      VLLM_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel", [&] {       \
        vllm::act_and_mul_kernel<                                           \
            scalar_t, typename vllm::PackedTraits<scalar_t>::packed_t,      \
            KERNEL<scalar_t>,                                               \
            PACKED_KERNEL<typename vllm::PackedTraits<scalar_t>::packed_t>, \
            ACT_FIRST, true, false><<<grid, block, 0, stream>>>(            \
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);       \
      });                                                                   \
    }                                                                       \
  } else {                                                                  \
    dim3 block(std::min(d, 1024));                                          \
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel", [&] {         \
      vllm::act_and_mul_kernel<                                             \
          scalar_t, typename vllm::PackedTraits<scalar_t>::packed_t,        \
          KERNEL<scalar_t>,                                                 \
          PACKED_KERNEL<typename vllm::PackedTraits<scalar_t>::packed_t>,   \
          ACT_FIRST, false><<<grid, block, 0, stream>>>(                    \
          out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);         \
    });                                                                     \
  }

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, vllm::packed_silu_kernel,
                                true);
}

void mul_and_silu(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  // The difference between mul_and_silu and silu_and_mul is that mul_and_silu
  // applies the silu to the latter half of the input.
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, vllm::packed_silu_kernel,
                                false);
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, vllm::packed_gelu_kernel,
                                true);
}

void gelu_tanh_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel,
                                vllm::packed_gelu_tanh_kernel, true);
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
  const scalar_t* x_ptr = input + blockIdx.x * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + blockIdx.x * d;

  if constexpr (use_vec) {
    // Fast path: 128-bit/256-bit vectorized loop
    using vec_t = typename VecTraits<use_256b>::vec_t;
    constexpr int ARCH_MAX_VEC_SIZE = VecTraits<use_256b>::ARCH_MAX_VEC_SIZE;
    constexpr int VEC_SIZE = ARCH_MAX_VEC_SIZE / sizeof(packed_t);

    const vec_t* x_vec = reinterpret_cast<const vec_t*>(x_ptr);
    const vec_t* y_vec = reinterpret_cast<const vec_t*>(y_ptr);
    vec_t* out_vec = reinterpret_cast<vec_t*>(out_ptr);
    const int num_vecs = d / 2 / VEC_SIZE;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      vec_t x, y;
      if constexpr (use_256b) {
        ld256(x, &x_vec[i]);
        ld256(y, &y_vec[i]);
      } else {
        x = VLLM_LDG(&x_vec[i]);
        y = VLLM_LDG(&y_vec[i]);
      }
      auto* xp = reinterpret_cast<packed_t*>(&x);
      auto* yp = reinterpret_cast<packed_t*>(&y);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        xp[j] = packed_mul(PACKED_ACT_FN(xp[j], param), yp[j]);
      }
      if constexpr (use_256b) {
        st256(x, &out_vec[i]);
      } else {
        out_vec[i] = x;
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

}  // namespace vllm

#define LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(KERNEL, PACKED_KERNEL, PARAM) \
  auto dtype = input.scalar_type();                                            \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  if (num_tokens == 0) {                                                       \
    return;                                                                    \
  }                                                                            \
  dim3 grid(num_tokens);                                                       \
  int cc_major = at::cuda::getCurrentDeviceProperties()->major;                \
  int support_vec = (cc_major >= 10 && num_tokens >= 128) ? 32 : 16;           \
  int vec_size = support_vec / at::elementSize(dtype);                         \
  const bool use_vec = (d % vec_size == 0);                                    \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  if (use_vec) {                                                               \
    dim3 block(std::min(d / vec_size, 1024));                                  \
    if (cc_major >= 10 && num_tokens >= 128) {                                 \
      VLLM_DISPATCH_FLOATING_TYPES(                                            \
          dtype, "act_and_mul_kernel_with_param", [&] {                        \
            vllm::act_and_mul_kernel_with_param<                               \
                scalar_t, typename vllm::PackedTraits<scalar_t>::packed_t,     \
                KERNEL<scalar_t>,                                              \
                PACKED_KERNEL<                                                 \
                    typename vllm::PackedTraits<scalar_t>::packed_t>,          \
                true, true><<<grid, block, 0, stream>>>(                       \
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d,       \
                PARAM);                                                        \
          });                                                                  \
    } else {                                                                   \
      VLLM_DISPATCH_FLOATING_TYPES(                                            \
          dtype, "act_and_mul_kernel_with_param", [&] {                        \
            vllm::act_and_mul_kernel_with_param<                               \
                scalar_t, typename vllm::PackedTraits<scalar_t>::packed_t,     \
                KERNEL<scalar_t>,                                              \
                PACKED_KERNEL<                                                 \
                    typename vllm::PackedTraits<scalar_t>::packed_t>,          \
                true, false><<<grid, block, 0, stream>>>(                      \
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d,       \
                PARAM);                                                        \
          });                                                                  \
    }                                                                          \
  } else {                                                                     \
    dim3 block(std::min(d, 1024));                                             \
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel_with_param", [&] { \
      vllm::act_and_mul_kernel_with_param<                                     \
          scalar_t, typename vllm::PackedTraits<scalar_t>::packed_t,           \
          KERNEL<scalar_t>,                                                    \
          PACKED_KERNEL<typename vllm::PackedTraits<scalar_t>::packed_t>,      \
          false><<<grid, block, 0, stream>>>(                                  \
          out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d, PARAM);     \
    });                                                                        \
  }

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

void fatrelu_and_mul(torch::Tensor& out,    // [..., d],
                     torch::Tensor& input,  // [..., 2 * d]
                     double threshold) {
  LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(
      vllm::fatrelu_kernel, vllm::packed_fatrelu_kernel, threshold);
}
void swigluoai_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input,  // [..., 2 * d]
                       double alpha, double limit) {
  LAUNCH_SIGLUOAI_AND_MUL(vllm::swigluoai_and_mul, alpha, limit);
}
namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, typename packed_t,
          scalar_t (*ACT_FN)(const scalar_t&),
          packed_t (*PACKED_ACT_FN)(const packed_t&), bool use_vec,
          bool use_256b = false>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {
  const scalar_t* in_ptr = input + blockIdx.x * d;
  scalar_t* out_ptr = out + blockIdx.x * d;

  if constexpr (use_vec) {
    // Fast path: 128-bit/256-bit vectorized loop
    using vec_t = typename VecTraits<use_256b>::vec_t;
    constexpr int ARCH_MAX_VEC_SIZE = VecTraits<use_256b>::ARCH_MAX_VEC_SIZE;
    constexpr int VEC_SIZE = ARCH_MAX_VEC_SIZE / sizeof(packed_t);
    const vec_t* in_vec = reinterpret_cast<const vec_t*>(in_ptr);
    vec_t* out_vec = reinterpret_cast<vec_t*>(out_ptr);
    const int num_vecs = d / 2 / VEC_SIZE;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      vec_t v;
      if constexpr (use_256b) {
        ld256(v, &in_vec[i]);
      } else {
        v = VLLM_LDG(&in_vec[i]);
      }
      auto* vp = reinterpret_cast<packed_t*>(&v);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        vp[j] = PACKED_ACT_FN(vp[j]);
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
#define LAUNCH_ACTIVATION_KERNEL(KERNEL, PACKED_KERNEL)                       \
  auto dtype = input.scalar_type();                                           \
  int d = input.size(-1);                                                     \
  int64_t num_tokens = input.numel() / input.size(-1);                        \
  if (num_tokens == 0) {                                                      \
    return;                                                                   \
  }                                                                           \
  dim3 grid(num_tokens);                                                      \
  int cc_major = at::cuda::getCurrentDeviceProperties()->major;               \
  int support_vec = (cc_major >= 10 && num_tokens >= 128) ? 32 : 16;          \
  int vec_size = support_vec / at::elementSize(dtype);                        \
  const bool use_vec = (d % vec_size == 0);                                   \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));           \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();               \
  if (use_vec) {                                                              \
    dim3 block(std::min(d / vec_size, 1024));                                 \
    if (cc_major >= 10 && num_tokens >= 128) {                                \
      VLLM_DISPATCH_FLOATING_TYPES(dtype, "activation_kernel", [&] {          \
        vllm::activation_kernel<                                              \
            scalar_t, vllm::PackedTraits<scalar_t>::packed_t,                 \
            KERNEL<scalar_t>,                                                 \
            PACKED_KERNEL<typename vllm::PackedTraits<scalar_t>::packed_t>,   \
            true, true><<<grid, block, 0, stream>>>(                          \
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);         \
      });                                                                     \
    } else {                                                                  \
      VLLM_DISPATCH_FLOATING_TYPES(dtype, "activation_kernel", [&] {          \
        vllm::activation_kernel<                                              \
            scalar_t, vllm::PackedTraits<scalar_t>::packed_t,                 \
            KERNEL<scalar_t>,                                                 \
            PACKED_KERNEL<typename vllm::PackedTraits<scalar_t>::packed_t>,   \
            true, false><<<grid, block, 0, stream>>>(                         \
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);         \
      });                                                                     \
    }                                                                         \
  } else {                                                                    \
    dim3 block(std::min(d, 1024));                                            \
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "activation_kernel", [&] {            \
      vllm::activation_kernel<                                                \
          scalar_t, vllm::PackedTraits<scalar_t>::packed_t, KERNEL<scalar_t>, \
          PACKED_KERNEL<typename vllm::PackedTraits<scalar_t>::packed_t>,     \
          false><<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),        \
                                             input.data_ptr<scalar_t>(), d);  \
    });                                                                       \
  }

namespace vllm {

template <typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename packed_t>
__device__ __forceinline__ packed_t
packed_gelu_new_kernel(const packed_t& val) {
  float2 fval = cast_to_float2(val);
  float x3 = fval.x * fval.x * fval.x;
  float tx = tanhf(0.79788456f * (fval.x + (0.044715f * x3)));
  fval.x = 0.5f * fval.x * (1.0f + tx);
  float y3 = fval.y * fval.y * fval.y;
  float ty = tanhf(0.79788456f * (fval.y + (0.044715f * y3)));
  fval.y = 0.5f * fval.y * (1.0f + ty);
  return cast_to_packed<packed_t>(fval);
}

template <typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename packed_t>
__device__ __forceinline__ packed_t
packed_gelu_fast_kernel(const packed_t& val) {
  float2 fval = cast_to_float2(val);
  float tx =
      tanhf((fval.x * 0.79788456f) * (1.0f + (0.044715f * fval.x) * fval.x));
  fval.x = 0.5f * fval.x * (1.0f + tx);
  float ty =
      tanhf((fval.y * 0.79788456f) * (1.0f + (0.044715f * fval.y) * fval.y));
  fval.y = 0.5f * fval.y * (1.0f + ty);
  return cast_to_packed<packed_t>(fval);
}

template <typename T>
__device__ __forceinline__ T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + expf(-1.702f * (float)x)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t
packed_gelu_quick_kernel(const packed_t& val) {
  // x * sigmoid(1.702 * x)
  float2 fval = cast_to_float2(val);
  fval.x = fval.x / (1.0f + expf(-1.702f * fval.x));
  fval.y = fval.y / (1.0f + expf(-1.702f * fval.y));
  return cast_to_packed<packed_t>(fval);
}

}  // namespace vllm

void gelu_new(torch::Tensor& out,    // [..., d]
              torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel, vllm::packed_gelu_new_kernel);
}

void gelu_fast(torch::Tensor& out,    // [..., d]
               torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel,
                           vllm::packed_gelu_fast_kernel);
}

void gelu_quick(torch::Tensor& out,    // [..., d]
                torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel,
                           vllm::packed_gelu_quick_kernel);
}
