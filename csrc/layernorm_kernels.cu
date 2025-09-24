#include "type_convert.cuh"
#include "dispatch_utils.h"
#include "cub_helpers.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,          // [..., hidden_size]
    const scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * input_stride + idx];
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * input_stride + idx];
    out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;
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
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[strided_id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * input_stride + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck.

   _f16VecPN struct extends _f16Vec to add operations specifically required for
   polynomial normalization (poly norm).
   The original _f16Vec does not include the sum-of-powers computation or
   in-place polynomial normalization logic. */
template <typename scalar_t, int width>
struct alignas(16) _f16VecPN : _f16Vec<scalar_t, width> {
  using Base = _f16Vec<scalar_t, width>;
  using Converter = typename Base::Converter;
  using T1 = typename Base::T1;
  using T2 = typename Base::T2;
  using Base::data;

  __device__ auto sum_pows() const {
    float s2 = 0.0f, s4 = 0.0f, s6 = 0.0f;

#pragma unroll
    for (int i = 0; i < width; i += 2) {
      float2 z = Converter::convert(T2{data[i], data[i + 1]});
      float x2 = z.x * z.x;
      float x4 = x2 * x2;
      float x6 = x4 * x2;

      float y2 = z.y * z.y;
      float y4 = y2 * y2;
      float y6 = y4 * y2;

      s2 += x2 + y2;
      s4 += x4 + y4;
      s6 += x6 + y6;
    }
    return std::make_tuple(s2, s4, s6);
  }

  __device__ void poly_norm_inplace(const float w2_inv_std,
                                    const float w1_inv_std2,
                                    const float w0_inv_std3, const float bias) {
#pragma unroll
    for (int i = 0; i < width; i += 2) {
      float2 z = Converter::convert(T2{data[i], data[i + 1]});

      float x2 = z.x * z.x;
      float x3 = x2 * z.x;
      z.x = w2_inv_std * z.x + w1_inv_std2 * x2 + w0_inv_std3 * x3 + bias;

      float y2 = z.y * z.y;
      float y3 = y2 * z.y;
      z.y = w2_inv_std * z.y + w1_inv_std2 * y2 + w0_inv_std3 * y3 + bias;

      auto out = Converter::convert(z);
      data[i] = out.x;
      data[i + 1] = out.y;
    }
  }
};

template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
poly_norm_kernel(scalar_t* __restrict__ out,           // [..., hidden_size]
                 const scalar_t* __restrict__ input,   // [..., hidden_size]
                 const scalar_t* __restrict__ weight,  // [3]
                 const scalar_t* __restrict__ bias,    // [1]
                 const float epsilon, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16VecPN<scalar_t, width>>);
  static_assert(sizeof(_f16VecPN<scalar_t, width>) == sizeof(scalar_t) * width);

  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<const _f16VecPN<scalar_t, width>*>(input);
  const int vec_hidden_size = hidden_size / width;
  float variance = 0.0f;
  float variance2 = 0.0f;
  float variance3 = 0.0f;

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16VecPN<scalar_t, width> temp = input_v[id];
    auto [x2, x4, x6] = temp.sum_pows();

    variance += x2;
    variance2 += x4;
    variance3 += x6;
  }

  float3 thread_variances = make_float3(variance, variance2, variance3);

  struct SumOp {
    __device__ float3 operator()(const float3& a, const float3& b) const {
      return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
  };

  using BlockReduce = cub::BlockReduce<float3, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  float3 block_variances =
      BlockReduce(reduceStore).Reduce(thread_variances, SumOp{}, blockDim.x);

  variance = block_variances.x;
  variance2 = block_variances.y;
  variance3 = block_variances.z;

  __shared__ float s_w2_inv_std;
  __shared__ float s_w1_inv_std2;
  __shared__ float s_w0_inv_std3;
  __shared__ float s_bias;

  if (threadIdx.x == 0) {
    float w0 = (float)weight[0];
    float w1 = (float)weight[1];
    float w2 = (float)weight[2];
    s_bias = (float)bias[0];

    s_w2_inv_std = w2 * rsqrtf(variance / hidden_size + epsilon);
    s_w1_inv_std2 = w1 * rsqrtf(variance2 / hidden_size + epsilon);
    s_w0_inv_std3 = w0 * rsqrtf(variance3 / hidden_size + epsilon);
  }
  __syncthreads();

  auto* __restrict__ out_v = reinterpret_cast<_f16VecPN<scalar_t, width>*>(out);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16VecPN<scalar_t, width> temp = input_v[id];
    temp.poly_norm_inplace(s_w2_inv_std, s_w1_inv_std2, s_w0_inv_std3, s_bias);
    out_v[id] = temp;
  }
}

/* Generic poly_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
poly_norm_kernel(scalar_t* __restrict__ out,           // [..., hidden_size]
                 const scalar_t* __restrict__ input,   // [..., hidden_size]
                 const scalar_t* __restrict__ weight,  // [3]
                 const scalar_t* __restrict__ bias,    // [1]
                 const float epsilon, const int hidden_size) {
  float variance = 0.0f;
  float variance2 = 0.0f;
  float variance3 = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    float x2 = x * x;
    float x4 = x2 * x2;
    float x6 = x4 * x2;

    variance += x2;
    variance2 += x4;
    variance3 += x6;
  }

  float3 thread_variances = make_float3(variance, variance2, variance3);

  struct SumOp {
    __device__ float3 operator()(const float3& a, const float3& b) const {
      return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
  };

  using BlockReduce = cub::BlockReduce<float3, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  float3 block_variances =
      BlockReduce(reduceStore).Reduce(thread_variances, SumOp{}, blockDim.x);

  variance = block_variances.x;
  variance2 = block_variances.y;
  variance3 = block_variances.z;

  __shared__ float s_w2_inv_std;
  __shared__ float s_w1_inv_std2;
  __shared__ float s_w0_inv_std3;
  __shared__ float s_bias;

  if (threadIdx.x == 0) {
    float w0 = (float)weight[0];
    float w1 = (float)weight[1];
    float w2 = (float)weight[2];
    s_bias = (float)bias[0];

    s_w2_inv_std = w2 * rsqrtf(variance / hidden_size + epsilon);
    s_w1_inv_std2 = w1 * rsqrtf(variance2 / hidden_size + epsilon);
    s_w0_inv_std3 = w0 * rsqrtf(variance3 / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    float x2 = x * x;
    float x3 = x2 * x;

    out[blockIdx.x * hidden_size + idx] =
        (scalar_t)(x * s_w2_inv_std + x2 * s_w1_inv_std2 + x3 * s_w0_inv_std3 +
                   s_bias);
  }
}

}  // namespace vllm

void rms_norm(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(weight.is_contiguous());

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int64_t input_stride = input.stride(-2);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
    vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), input_stride,
        weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
  });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                    \
  VLLM_DISPATCH_FLOATING_TYPES(                                             \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {               \
        vllm::fused_add_rms_norm_kernel<scalar_t, width>                    \
            <<<grid, block, 0, stream>>>(                                   \
                input.data_ptr<scalar_t>(), input_stride,                   \
                residual.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), \
                epsilon, num_tokens, hidden_size);                          \
      });

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
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
  constexpr int vector_width = 8;
  constexpr int req_alignment_bytes =
      vector_width * 2;  // vector_width * sizeof(bfloat16 or float16) (float32
                         // falls back to non-vectorized version anyway)
  bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 &&
                          res_ptr % req_alignment_bytes == 0 &&
                          wt_ptr % req_alignment_bytes == 0;
  bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input_stride % vector_width == 0;
  if (ptrs_are_aligned && offsets_are_multiple_of_vector_width) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}

#define LAUNCH_FUSED_POLY_NORM(width)                                         \
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "poly_norm_kernel", [&] { \
    vllm::poly_norm_kernel<scalar_t, width><<<grid, block, 0, stream>>>(      \
        out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),                 \
        weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), epsilon,      \
        hidden_size);                                                         \
  });

void poly_norm(torch::Tensor& out,     // [..., hidden_size]
               torch::Tensor& input,   // [..., hidden_size]
               torch::Tensor& weight,  // [3]
               torch::Tensor& bias,    // [1]
               double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.data_ptr() != input.data_ptr());

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
  auto out_ptr = reinterpret_cast<std::uintptr_t>(out.data_ptr());
  bool ptrs_are_aligned = inp_ptr % 16 == 0 && out_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0) {
    LAUNCH_FUSED_POLY_NORM(8);
  } else {
    LAUNCH_FUSED_POLY_NORM(0);
  }
}
