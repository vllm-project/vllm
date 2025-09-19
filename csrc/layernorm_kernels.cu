#include "type_convert.cuh"
#include "dispatch_utils.h"
#include "cub_helpers.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>
#include "quantization/vectorization_utils.cuh"
#include "quantization/fused_kernels/layernorm_utils.cuh"

namespace vllm {

constexpr int kVecBytes = 16;  // 128-bit phase

template <typename T>
__device__ __forceinline__ bool same_phase(const T* a, const T* b, int bytes) {
  const auto ai = reinterpret_cast<uintptr_t>(a);
  const auto bi = reinterpret_cast<uintptr_t>(b);
  return ((ai ^ bi) & (bytes - 1)) == 0;
}

// copy input row to shared with 16B phase when possible
template <typename T>
__device__ __forceinline__ void copy_row_to_shared_aligned(
    const T* __restrict__ src, T* __restrict__ dst, int n_elems, int tid) {
  const auto sa = reinterpret_cast<uintptr_t>(src);
  const auto da = reinterpret_cast<uintptr_t>(dst);
  const bool same = (((sa ^ da) & (kVecBytes - 1)) == 0);

  if (!same) {
    for (int i = tid; i < n_elems; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
    return;
  }

  const int ebytes = sizeof(T);
  const int perVec = kVecBytes / ebytes;

  int prefix = 0;
  const int mis = sa & (kVecBytes - 1);
  if (mis) prefix = (kVecBytes - mis) / ebytes;
  if (prefix > n_elems) prefix = n_elems;

  for (int i = tid; i < prefix; i += blockDim.x) dst[i] = src[i];

  const int remain = n_elems - prefix;
  const int main_elems = (remain / perVec) * perVec;
  if (main_elems > 0) {
    const uint4* __restrict__ vsrc =
        reinterpret_cast<const uint4*>(src + prefix);
#if defined(__HIP_PLATFORM_AMD__)
    uint32_t* __restrict__ s32 = reinterpret_cast<uint32_t*>(dst + prefix);
    const int nvec = main_elems / perVec;
    constexpr int WORDS_PER_PKT = kVecBytes / sizeof(uint32_t);  // 4
    for (int v = tid; v < nvec; v += blockDim.x) {
      const uint4 p = vsrc[v];
      const int base = v * WORDS_PER_PKT;
      s32[base + 0] = p.x;
      s32[base + 1] = p.y;
      s32[base + 2] = p.z;
      s32[base + 3] = p.w;
    }
#else
    uint4* __restrict__ vdst = reinterpret_cast<uint4*>(dst + prefix);
    const int nvec = main_elems / perVec;
    for (int v = tid; v < nvec; v += blockDim.x) {
      vdst[v] = vsrc[v];
    }
#endif
  }

  const int tail = prefix + main_elems;
  for (int i = tid + tail; i < n_elems; i += blockDim.x) dst[i] = src[i];
  __syncthreads();
}

template <int V, typename T>
struct VecMulNormWeight {
  const vec_n_t<T, V>* __restrict__ wv;
  float inv_rms;
  int stride_vec;
  mutable int64_t vec_idx;
  __device__ __forceinline__ void operator()(vec_n_t<T, V>& dst,
                                             const vec_n_t<T, V>& src) const {
    const vec_n_t<T, V> w = wv[vec_idx];
#pragma unroll
    for (int j = 0; j < V; ++j) {
      const T xn = static_cast<T>(static_cast<float>(src.val[j]) * inv_rms);
      dst.val[j] = xn * w.val[j];
    }
    vec_idx += stride_vec;
  }
};

template <typename T>
struct ScalarMulNormWeight {
  const T* __restrict__ w_base;
  T* __restrict__ out_base;
  float inv_rms;
  __device__ __forceinline__ void operator()(T& dst, const T src) const {
    const int i = static_cast<int>(&dst - out_base);
    const T xn = static_cast<T>(static_cast<float>(src) * inv_rms);
    dst = xn * w_base[i];
  }
};

template <int V, typename T>
struct VecNormMulWeightScalarW {
  const T* __restrict__ w_base;  // offset by prefix
  float inv_rms;
  int stride_vec;
  mutable int vec_idx;
  __device__ __forceinline__ void operator()(vec_n_t<T, V>& dst,
                                             const vec_n_t<T, V>& src) const {
    const int base = vec_idx * V;
#pragma unroll
    for (int j = 0; j < V; ++j) {
      const float x = static_cast<float>(src.val[j]) * inv_rms;
      dst.val[j] = static_cast<T>(x * static_cast<float>(w_base[base + j]));
    }
    vec_idx += stride_vec;
  }
};

template <typename scalar_t>
__global__ void rms_norm_kernel(scalar_t* __restrict__ out,
                                const scalar_t* __restrict__ input,
                                const int64_t input_stride,
                                const scalar_t* __restrict__ weight,
                                const float epsilon, const int /*num_tokens*/,
                                const int hidden_size, int smem_elems) {
  const scalar_t* __restrict__ in_row = input + blockIdx.x * input_stride;
  scalar_t* __restrict__ out_row = out + blockIdx.x * hidden_size;

  extern __shared__ unsigned char smem_raw[];
  scalar_t* s_in = reinterpret_cast<scalar_t*>(smem_raw);

#ifdef __HIP_PLATFORM_AMD__
  constexpr bool kAllowCache = false;
#else
  constexpr bool kAllowCache = true;
#endif
  const bool use_cached =
      kAllowCache && (sizeof(scalar_t) == 2) && (smem_elems > 0);

#if !defined(__HIP_PLATFORM_AMD__)
  if (use_cached)
    copy_row_to_shared_aligned(in_row, s_in, hidden_size, threadIdx.x);
#endif

  float sumsq = 0.f;
  {
    const scalar_t* base = use_cached ? s_in : in_row;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      const float x = static_cast<float>(base[i]);
      sumsq += x * x;
    }
  }

  float wsum = warp_sum<float>(sumsq);
  __shared__ float warp_sums_sh[32];
  if ((threadIdx.x & 31) == 0) warp_sums_sh[threadIdx.x >> 5] = wsum;
  __syncthreads();

  if (threadIdx.x < 32) {
    const int nwarps = (blockDim.x + 31) / 32;
    const float v = (threadIdx.x < nwarps) ? warp_sums_sh[threadIdx.x] : 0.f;
    const float total = warp_sum<float>(v);
    if (threadIdx.x == 0) warp_sums_sh[0] = total;
  }
  __syncthreads();

  const float inv_rms =
      rsqrtf(warp_sums_sh[0] / static_cast<float>(hidden_size) + epsilon);

  if (hidden_size == blockDim.x) {
    const int i = threadIdx.x;
    const float x = static_cast<float>(use_cached ? s_in[i] : in_row[i]);
    const scalar_t xn = static_cast<scalar_t>(x * inv_rms);
    out_row[i] = xn * weight[i];
    return;
  }

  constexpr int V = (sizeof(scalar_t) == 2) ? 8 : 4;  // 16B
  constexpr int WIDTH = V * sizeof(scalar_t);
  const bool vec_store_ok =
      (hidden_size % V == 0) && same_phase(in_row, out_row, WIDTH);

  const bool s_same = use_cached && same_phase(in_row, s_in, kVecBytes);
  const scalar_t* vin = s_same ? s_in : in_row;

  if (vec_store_ok) {
    ScalarMulNormWeight<scalar_t> sca_op{weight, out_row, inv_rms};

    const auto addr = reinterpret_cast<uintptr_t>(vin);
    const int mis = addr & (WIDTH - 1);
    const int prefix =
        mis ? (WIDTH - mis) / static_cast<int>(sizeof(scalar_t)) : 0;

    if (same_phase(in_row, weight, WIDTH)) {
      using VecT = vec_n_t<scalar_t, V>;
      const VecT* __restrict__ wv =
          reinterpret_cast<const VecT*>(weight + prefix);
      VecMulNormWeight<V, scalar_t> vec_op{wv, inv_rms, (int)blockDim.x,
                                           (int64_t)threadIdx.x};
      vectorize_with_alignment<V>(vin, out_row, hidden_size, threadIdx.x,
                                  blockDim.x, vec_op, sca_op);
    } else {
      VecNormMulWeightScalarW<V, scalar_t> vec_op{
          weight + prefix, inv_rms, (int)blockDim.x, (int)threadIdx.x};
      vectorize_with_alignment<V>(vin, out_row, hidden_size, threadIdx.x,
                                  blockDim.x, vec_op, sca_op);
    }
    return;
  }

  // scalar fallback (keeps op order identical to fused path)
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float x = static_cast<float>(use_cached ? s_in[i] : in_row[i]);
    const scalar_t xn = static_cast<scalar_t>(x * inv_rms);
    out_row[i] = xn * weight[i];
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

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int64_t in_stride = input.stride(-2);

  dim3 grid(num_tokens);
  dim3 block(vllm::ln_block_threads_unified(hidden_size));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Optional per-block row cache in dynamic shared memory.
  // If enabled (FP16 and HS <= 4096), the kernel copies the row to smem once
  // and reuses it on the second pass to cut a global read. If shmem_bytes == 0,
  // the kernel takes the non-cached path
  size_t shmem_bytes = 0;
  int smem_elems = 0;
  if (input.scalar_type() == at::kHalf && hidden_size <= 4096) {
    shmem_bytes = static_cast<size_t>(hidden_size) * sizeof(at::Half);
    smem_elems = hidden_size;  // flag to kernel that shmem was provisioned
  }

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
    vllm::rms_norm_kernel<scalar_t><<<grid, block, shmem_bytes, stream>>>(
        out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), in_stride,
        weight.data_ptr<scalar_t>(), static_cast<float>(epsilon), num_tokens,
        hidden_size, smem_elems);
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
