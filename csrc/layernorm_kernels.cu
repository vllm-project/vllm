#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#include "reduction_utils.cuh"
#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

  using __nv_bfloat16 = __hip_bfloat16;
  using __nv_bfloat162 = __hip_bfloat162;
#endif

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
__global__ void rms_norm_kernel(
  scalar_t* __restrict__ out,             // [..., hidden_size]
  const scalar_t* __restrict__ input,     // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float) input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}


/* Converter structs for the conversion from torch types to HIP/CUDA types,
   and the associated type conversions within HIP/CUDA. These helpers need
   to be implemented for now because the relevant type conversion
   operators/constructors are not consistently implemented by HIP/CUDA, so
   a generic conversion via type casts cannot be implemented.

   Each struct should have the member static constexpr bool `exists`:
   If false, the optimized kernel is not used for the corresponding torch type.
   If true, the struct should be fully defined as shown in the examples below. 
 */
template<typename torch_type>
struct _typeConvert { static constexpr bool exists = false; };

#if defined(USE_ROCM) || (defined(CUDA_VERSION) && (CUDA_VERSION >= 12000))
// CUDA < 12.0 runs into issues with packed type conversion
template<>
struct _typeConvert<c10::Half> {
  static constexpr bool exists = true;
  using hip_type = __half;
  using packed_hip_type = __half2;

  __device__ static inline float convert(hip_type x) { return __half2float(x); }
  __device__ static inline float2 convert(packed_hip_type x) { return __half22float2(x); }
  __device__ static inline hip_type convert(float x) { return __float2half_rn(x); }
  __device__ static inline packed_hip_type convert(float2 x) { return __float22half2_rn(x); }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// CUDA_ARCH < 800 does not have BF16 support
// TODO: Add in ROCm support once public headers handle bf16 maturely
template<>
struct _typeConvert<c10::BFloat16> {
  static constexpr bool exists = true;
  using hip_type = __nv_bfloat16;
  using packed_hip_type = __nv_bfloat162;

  __device__ static inline float convert(hip_type x) { return __bfloat162float(x); }
  __device__ static inline float2 convert(packed_hip_type x) { return __bfloat1622float2(x); }
  __device__ static inline hip_type convert(float x) { return __float2bfloat16(x); }
  __device__ static inline packed_hip_type convert(float2 x) { return __float22bfloat162_rn(x); }
};
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#endif // defined(USE_ROCM) || (defined(CUDA_VERSION) && (CUDA_VERSION >= 12000))

/* Vector POD struct to generate vectorized and packed FP16/BF16 ops
   for appropriate specializations of fused_add_rms_norm_kernel.
   Only functions that are necessary in that kernel are implemented.
   Alignment to 16 bytes is required to use 128-bit global memory ops.
 */
template<typename scalar_t, int width>
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
        T2 temp{data[i], data[i+1]};
        temp += T2{other.data[i], other.data[i+1]};
        data[i] = temp.x;
        data[i+1] = temp.y;
      }
    } else {
      #pragma unroll
      for (int i = 0; i < width; ++i)
        data[i] += other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
      #pragma unroll
      for (int i = 0; i < width; i += 2) {
        T2 temp{data[i], data[i+1]};
        temp *= T2{other.data[i], other.data[i+1]};
        data[i] = temp.x;
        data[i+1] = temp.y;
      }
    } else {
      #pragma unroll
      for (int i = 0; i < width; ++i)
        data[i] *= other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const float scale) {
    if constexpr (width % 2 == 0) {
      #pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 temp_f = Converter::convert(T2{data[i], data[i+1]});
        temp_f.x *= scale;
        temp_f.y *= scale;
        T2 temp = Converter::convert(temp_f);
        data[i] = temp.x;
        data[i+1] = temp.y;
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
        float2 z = Converter::convert(T2{data[i], data[i+1]});
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
template<typename scalar_t, int width>
__global__ std::enable_if_t<
  (width > 0) && _typeConvert<scalar_t>::exists> fused_add_rms_norm_kernel(
  scalar_t* __restrict__ input,           // [..., hidden_size]
  scalar_t* __restrict__ residual,        // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v = reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v = reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v = reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = input_v[id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }
  /* Keep the following if-else block in sync with the
     calculation of max_block_size in fused_add_rms_norm */ 
  if (num_tokens < 256) {
    variance = blockReduceSum<float, 1024>(variance);
  } else variance = blockReduceSum<float, 256>(variance);
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
template<typename scalar_t, int width>
__global__ std::enable_if_t<
  (width == 0) || !_typeConvert<scalar_t>::exists> fused_add_rms_norm_kernel(
  scalar_t* __restrict__ input,           // [..., hidden_size]
  scalar_t* __restrict__ residual,        // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * hidden_size + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float) z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }
  /* Keep the following if-else block in sync with the
     calculation of max_block_size in fused_add_rms_norm */ 
  if (num_tokens < 256) {
    variance = blockReduceSum<float, 1024>(variance);
  } else variance = blockReduceSum<float, 256>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

} // namespace vllm

void rms_norm(
  torch::Tensor& out,      // [..., hidden_size]
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "rms_norm_kernel",
    [&] {
      vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size);
    });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)              \
  VLLM_DISPATCH_FLOATING_TYPES(                       \
    input.scalar_type(),                              \
    "fused_add_rms_norm_kernel",                      \
    [&] {                                             \
      vllm::fused_add_rms_norm_kernel                 \
      <scalar_t, width><<<grid, block, 0, stream>>>(  \
        input.data_ptr<scalar_t>(),                   \
        residual.data_ptr<scalar_t>(),                \
        weight.data_ptr<scalar_t>(),                  \
        epsilon,                                      \
        num_tokens,                                   \
        hidden_size);                                 \
    });

void fused_add_rms_norm(
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& residual, // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
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
  bool ptrs_are_aligned = inp_ptr % 16 == 0 && res_ptr % 16 == 0 \
                          && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}
