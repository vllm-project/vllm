#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#include "attention/attention_dtypes.h"
#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <cub/util_type.cuh>
  #include <cub/cub.cuh>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
  #include <hipcub/util_type.hpp>
  #include <hipcub/hipcub.hpp>
  #include "quantization/fp8/amd/hip_float8.h"
  #include "quantization/fp8/amd/quant_utils.cuh"

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
#endif

#ifdef USE_ROCM
  #include "quantization/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx940__) || \
                           defined(__gfx941__) || defined(__gfx942__))
  #define __HIP__MI300_MI250__
#endif

namespace vllm {

template <typename scalar_t>
struct __align__(16) vec8_t {
  scalar_t x, y, z, w, u, v, s, t;

  __device__ vec8_t() : x(0), y(0), z(0), w(0), u(0), v(0), s(0), t(0) {}
  __device__ vec8_t(scalar_t x, scalar_t y, scalar_t z, scalar_t w, scalar_t u,
                    scalar_t v, scalar_t s, scalar_t t)
      : x(x), y(y), z(z), w(w), u(u), v(v), s(s), t(t) {}

  __device__ vec8_t operator*(const vec8_t& other) const {
    return vec8_t(x * other.x, y * other.y, z * other.z, w * other.w,
                  u * other.u, v * other.v, s * other.s, t * other.t);
  }

  __device__ vec8_t operator*(const float& scale) const {
    return vec8_t(x * scale, y * scale, z * scale, w * scale, u * scale,
                  v * scale, s * scale, t * scale);
  }

  __device__ vec8_t operator+(const vec8_t& other) const {
    return vec8_t(x + other.x, y + other.y, z + other.z, w + other.w,
                  u + other.u, v + other.v, s + other.s, t + other.t);
  }

  __device__ void operator+=(const vec8_t& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    w += other.w;
    u += other.u;
    v += other.v;
    s += other.s;
    t += other.t;
  }

  __device__ scalar_t sum() const { return x + y + z + w + u + v + s + t; }
};

#ifdef __HIP__MI300_MI250__

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;

  vec8_t<scalar_t> v8_variance = {0, 0, 0, 0, 0, 0, 0, 0};

  vec8_t<scalar_t>* vectorized_out = reinterpret_cast<vec8_t<scalar_t>*>(out);
  vec8_t<scalar_t> const* vectorized_in =
      reinterpret_cast<vec8_t<scalar_t> const*>(input);
  vec8_t<scalar_t> const* vectorized_weight =
      reinterpret_cast<vec8_t<scalar_t> const*>(weight);
  const int vec_hidden_size = hidden_size >> 3;

  // Compute variance. Be careful, hidden_size should multiple of 4.
  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    vec8_t<scalar_t> x = vectorized_in[blockIdx.x * vec_hidden_size + idx];
    v8_variance += x * x;
  }
  float v8_variance_sum = v8_variance.sum();

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  float variance =
      BlockReduce(reduceStore).Reduce(v8_variance_sum, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    vec8_t<scalar_t> v8_in = vectorized_in[blockIdx.x * vec_hidden_size + idx];
    vec8_t<scalar_t> v8_w = vectorized_weight[idx];
    vectorized_out[blockIdx.x * vec_hidden_size + idx] =
        v8_in * s_variance * v8_w;
  }
}

#else

// TODO(maleksan): Investigate why vectorization doesn't work for Navi.
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

#endif

template <typename scalar_t>
__global__ void scaled_rms_norm_kernel(
    c10::Float8_e4m3fnuz* __restrict__ out,  // [..., hidden_size]
    const scalar_t* __restrict__ input,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,     // [hidden_size]
    const float scale, const float epsilon, const int num_tokens,
    const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    float r = (x * s_variance) * weight[idx] * scale;
    out[blockIdx.x * hidden_size + idx] = c10::Float8_e4m3fnuz(
        hip_fp8(r).data, c10::Float8_e4m3fnuz::from_bits());
  }
}

/* Converter structs for the conversion from torch types to HIP/CUDA types,
   and the associated type conversions within HIP/CUDA. These helpers need
   to be implemented for now because/error the relevant type conversion
   operators/constructors are not consistently implemented by HIP/CUDA, so
   a generic conversion via type casts cannot be implemented.

   Each struct should have the member static constexpr bool `exists`:
   If false, the optimized kernel is not used for the corresponding torch type.
   If true, the struct should be fully defined as shown in the examples below.
 */
template <typename torch_type>
struct _typeConvert {
  static constexpr bool exists = false;
};

#if defined(USE_ROCM) || (defined(CUDA_VERSION) && (CUDA_VERSION >= 12000))
// CUDA < 12.0 runs into issues with packed type conversion
template <>
struct _typeConvert<c10::Half> {
  static constexpr bool exists = true;
  using hip_type = __half;
  using packed_hip_type = __half2;

  __device__ static inline float convert(hip_type x) { return __half2float(x); }
  __device__ static inline float2 convert(packed_hip_type x) {
    return __half22float2(x);
  }
  __device__ static inline hip_type convert(float x) {
    return __float2half_rn(x);
  }
  __device__ static inline packed_hip_type convert(float2 x) {
    return __float22half2_rn(x);
  }
};

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// CUDA_ARCH < 800 does not have BF16 support
// TODO: Add in ROCm support once public headers handle bf16 maturely
template <>
struct _typeConvert<c10::BFloat16> {
  static constexpr bool exists = true;
  using hip_type = __nv_bfloat16;
  using packed_hip_type = __nv_bfloat162;

  __device__ static inline float convert(hip_type x) {
    return __bfloat162float(x);
  }
  __device__ static inline float2 convert(packed_hip_type x) {
    return __bfloat1622float2(x);
  }
  __device__ static inline hip_type convert(float x) {
    return __float2bfloat16(x);
  }
  __device__ static inline packed_hip_type convert(float2 x) {
    return __float22bfloat162_rn(x);
  }
};
  #endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#endif    // defined(USE_ROCM) || (defined(CUDA_VERSION) && (CUDA_VERSION >=
          // 12000))

/* Vector POD struct to generate vectorized and packed FP16/BF16 ops
   for appropriate specializations of fused_add_rms_norm_kernel.
   Only functions that are necessary in that kernel are implemented.
   Alignment to 16 bytes is required to use 128-bit global memory ops.
 */
template <typename scalar_t, int width>
struct alignas(16) _f16Vec {
  /* Not theoretically necessary that width is a power of 2 but should
     almost always be the case for optimization purposes */
  static_assert(width > 0 && (width & (width - 1)) == 0,
                "Width is not a positive power of 2!");
  using Converter = _typeConvert<scalar_t>;
  using T1 = typename Converter::hip_type;
  using T2 = typename Converter::packed_hip_type;
  T1 data[width];

  __device__ _f16Vec& operator+=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        T2 temp{data[i], data[i + 1]};
        temp += T2{other.data[i], other.data[i + 1]};
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) data[i] += other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        T2 temp{data[i], data[i + 1]};
        temp *= T2{other.data[i], other.data[i + 1]};
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) data[i] *= other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const float scale) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 temp_f = Converter::convert(T2{data[i], data[i + 1]});
        temp_f.x *= scale;
        temp_f.y *= scale;
        T2 temp = Converter::convert(temp_f);
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float temp = Converter::convert(data[i]) * scale;
        data[i] = Converter::convert(temp);
      }
    }
    return *this;
  }

  __device__ float sum_squares() const {
    float result = 0.0f;
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 z = Converter::convert(T2{data[i], data[i + 1]});
        result += z.x * z.x + z.y * z.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float x = Converter::convert(data[i]);
        result += x * x;
      }
    }
    return result;
  }
};

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,         // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = input_v[id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,         // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * hidden_size + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */

template <>
struct Vec<c10::Float8_e4m3fnuz, 8> {
  using Type = uint2;
};

template <>
struct Vec<c10::Half, 8> {
  using Type = uint4;
};

template <>
struct Vec<c10::BFloat16, 8> {
  using Type = bf16_8_t;
};

template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
scaled_fused_add_rms_norm_kernel(
    c10::Float8_e4m3fnuz* __restrict__ out,  // [..., hidden_size]
    scalar_t* __restrict__ input,            // [..., hidden_size]
    scalar_t* __restrict__ residual,         // [..., hidden_size]
    const scalar_t* __restrict__ weight,     // [hidden_size]
    const float epsilon, const float scale, const int num_tokens,
    const int hidden_size) {
  using in_v_t = typename Vec<scalar_t, width>::Type;
  using out_v_t = typename Vec<c10::Float8_e4m3fnuz, width>::Type;
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ out_v = reinterpret_cast<out_v_t*>(out);
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = input_v[id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    out_v_t temp_quant = fp8::scaled_vec_conversion<out_v_t, in_v_t>(
        *reinterpret_cast<in_v_t*>(&temp), scale);
    out_v[id] = temp_quant;
  }
}

/* Generic scaled_fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
scaled_fused_add_rms_norm_kernel(
    c10::Float8_e4m3fnuz* __restrict__ out,  // [..., hidden_size]
    scalar_t* __restrict__ input,            // [..., hidden_size]
    scalar_t* __restrict__ residual,         // [..., hidden_size]
    const scalar_t* __restrict__ weight,     // [hidden_size]
    const float epsilon, const float scale, const int num_tokens,
    const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * hidden_size + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    float r = (x * s_variance) * (float)weight[idx] / scale;
    out[blockIdx.x * hidden_size + idx] = c10::Float8_e4m3fnuz(
        hip_fp8(r).data, c10::Float8_e4m3fnuz::from_bits());
  }
}

}  // namespace vllm

void rms_norm(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
    vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
  });
}

void scaled_rms_norm(torch::Tensor& out,     // [..., hidden_size]
                     torch::Tensor& input,   // [..., hidden_size]
                     torch::Tensor& weight,  // [hidden_size]
                     torch::Tensor& scale, double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_rms_norm_kernel", [&] {
        vllm::scaled_rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<c10::Float8_e4m3fnuz>(), input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(), 1.0 / (*scale.data_ptr<float>()),
            epsilon, num_tokens, hidden_size);
      });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                       \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {                  \
        vllm::fused_add_rms_norm_kernel<scalar_t, width>                       \
            <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),           \
                                         residual.data_ptr<scalar_t>(),        \
                                         weight.data_ptr<scalar_t>(), epsilon, \
                                         num_tokens, hidden_size);             \
      });

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}

#define LAUNCH_SCALED_FUSED_ADD_RMS_NORM(width)                            \
  VLLM_DISPATCH_FLOATING_TYPES(                                            \
      input.scalar_type(), "scaled_fused_add_rms_norm_kernel", [&] {       \
        vllm::scaled_fused_add_rms_norm_kernel<scalar_t, width>            \
            <<<grid, block, 0, stream>>>(                                  \
                out.data_ptr<c10::Float8_e4m3fnuz>(),                      \
                input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(), \
                weight.data_ptr<scalar_t>(), epsilon,                      \
                *scale.data_ptr<float>(), num_tokens, hidden_size);        \
      });

void scaled_fused_add_rms_norm(torch::Tensor& out,       // [..., hidden_size]
                               torch::Tensor& input,     // [..., hidden_size]
                               torch::Tensor& residual,  // [..., hidden_size]
                               torch::Tensor& weight,    // [hidden_size]
                               torch::Tensor& scale, double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0) {
    LAUNCH_SCALED_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_SCALED_FUSED_ADD_RMS_NORM(0);
  }
}
