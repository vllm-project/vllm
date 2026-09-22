#include "../../../../quantization/w8a8/fp8/common.cuh"
#include "../../../../cuda_compat.h"
#include "../../../../cuda_utils.h"
#include "../../../dispatch_utils.h"
#include "../../../cub_helpers.h"
#include "../../vectorization_utils.cuh"
#include "../../../torch_utils.h"
#include <torch/csrc/stable/macros.h>

#include <algorithm>

namespace vllm {

namespace {

// 16 fp8 outputs form one 16-byte store; the matching 16-element input load
// is 32 B for 16-bit types and 64 B for fp32.
constexpr int kFp8QuantVecSize = 16;
constexpr int kFp8QuantBlockSize = 256;

template <typename T>
using fp8_quant_vec_t = vec_n_t<T, kFp8QuantVecSize>;

// Vectors per thread on the single-read per-token path. 16-bit rows longer
// than one vector per thread (hidden > 4096) stay on the two-pass kernel: its
// second read hits L2 and it already runs at HBM bandwidth there, while
// holding more vectors measured 2-7% slower on H100. fp32 rows gain up to 4.
template <typename T>
constexpr int kFp8QuantMaxVecsPerThread = sizeof(T) == 4 ? 4 : 1;

// Picks the fewest vectors per thread whose thread count, rounded up to whole
// warps, fits in one block, so every thread holds the same number of vectors.
// Returns false when the row is too long for the single-read path.
inline bool select_single_read_launch(int num_vecs, int max_vecs, int warp_size,
                                      int block_size, int& threads,
                                      int& vecs_per_thread) {
  for (int v = 1; v <= max_vecs; v *= 2) {
    const int t =
        cuda_utils::ceil_div(cuda_utils::ceil_div(num_vecs, v), warp_size) *
        warp_size;
    if (t <= block_size) {
      threads = t;
      vecs_per_thread = v;
      return true;
    }
  }
  return false;
}

template <typename T>
bool ptr_vec_aligned(const T* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % alignof(fp8_quant_vec_t<T>) == 0;
}

// True when every row of a [tokens, hidden] view starts on a
// fp8_quant_vec_t<T> boundary, so rows can be read or written as whole vectors.
template <typename T>
bool rows_vec_aligned(const T* ptr, int64_t row_stride) {
  return ptr_vec_aligned(ptr) &&
         (row_stride * sizeof(T)) % alignof(fp8_quant_vec_t<T>) == 0;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, VLLM_SHFL_XOR_SYNC(v, offset));
  }
  return v;
}

// Block-wide max, returned to every thread. Warp partials go through one
// shared-memory exchange, so the whole reduction costs a single
// __syncthreads. blockDim.x must be a multiple of WARP_SIZE and at most
// kFp8QuantBlockSize.
__device__ __forceinline__ float block_reduce_max(float v, float* warp_max) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;
  v = warp_reduce_max(v);
  if (lane == 0) {
    warp_max[warp] = v;
  }
  __syncthreads();
  v = warp_max[0];
  for (int w = 1; w < num_warps; ++w) {
    v = fmaxf(v, warp_max[w]);
  }
  return v;
}

}  // namespace

// STRIDE_I_ZERO: true if scale_stride_i == 0 (per-tensor or per-channel)
// STRIDE_J_ZERO: true if scale_stride_j == 0 (per-tensor or per-token)
template <typename scalar_t, typename fp8_type, bool STRIDE_I_ZERO,
          bool STRIDE_J_ZERO>
__global__ void scaled_fp8_quant_kernel_strided_group_shape(
    fp8_type* __restrict__ out, const scalar_t* __restrict__ input,
    const float* __restrict__ scale, int hidden_size, int64_t in_row_stride,
    int64_t out_row_stride, int group_m, int group_n, int64_t scale_stride_i,
    int64_t scale_stride_j) {
  const int64_t token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const scalar_t* token_in = input + token_idx * in_row_stride;
  fp8_type* token_out = out + token_idx * out_row_stride;

  // Precompute row-level base offset for scale access (compile-time eliminated
  // when STRIDE_I_ZERO)
  const int64_t scale_row_base =
      STRIDE_I_ZERO ? 0
                    : static_cast<int>(token_idx) / group_m * scale_stride_i;

  auto get_inv_scale = [&](int gj) {
    return 1.0f / scale[scale_row_base + gj * scale_stride_j];
  };

  int cached_gj = -1;
  float cached_inv_scale = 0.0f;
  auto get_inv_scale_cached = [&](int gj) {
    if (gj != cached_gj) {
      cached_inv_scale = 1.0f / scale[scale_row_base + gj * scale_stride_j];
      cached_gj = gj;
    }
    return cached_inv_scale;
  };

  constexpr int VEC_SIZE = 16;  // FP8 so vectorize to 128 bits
  auto scaled_fp8_conversion_vectorized = [&](const scalar_t* in, fp8_type* out,
                                              int size, float inv_scale) {
    vectorize_with_alignment<VEC_SIZE>(
        in, out, size, tid, blockDim.x,
        [=] __device__(fp8_type & dst, const scalar_t& src) {
          dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                      inv_scale);
        });
  };

  if (STRIDE_J_ZERO && hidden_size % VEC_SIZE == 0) {
    // Per-tensor or per-token: single scale per row, vectorize full row
    scaled_fp8_conversion_vectorized(token_in, token_out, hidden_size,
                                     get_inv_scale(0));
  } else if (group_n % VEC_SIZE == 0) {
    // Multiple column groups with vectorization
    const int num_groups_n = hidden_size / group_n;

    for (int gj = 0; gj < num_groups_n; gj++) {
      scaled_fp8_conversion_vectorized(token_in + gj * group_n,
                                       token_out + gj * group_n, group_n,
                                       get_inv_scale(gj));
    }
  } else {
    // Scalar path for small column groups (group_n < VEC_SIZE)
    for (int n = tid; n < hidden_size; n += blockDim.x) {
      const int gj = n / group_n;
      token_out[n] = scaled_fp8_conversion<true, fp8_type>(
          static_cast<float>(token_in[n]), get_inv_scale_cached(gj));
    }
  }
}

// Per-tensor abs-max of a contiguous tensor viewed as a flat array of
// 16-element vectors: grid-stride loop, one atomic per block.
template <typename scalar_t, typename fp8_type>
__global__ void __launch_bounds__(kFp8QuantBlockSize)
    absmax_reduction_flat(float* __restrict__ scale,
                          const scalar_t* __restrict__ input,
                          int64_t num_vecs) {
  using in_vec_t = fp8_quant_vec_t<scalar_t>;
  const auto* v_in = reinterpret_cast<const in_vec_t*>(input);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  float thread_max = 0.0f;
#pragma unroll 4
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < num_vecs; v += stride) {
    const in_vec_t vec = v_in[v];
#pragma unroll
    for (int i = 0; i < kFp8QuantVecSize; ++i) {
      thread_max = fmaxf(thread_max, fabsf(static_cast<float>(vec.val[i])));
    }
  }

  __shared__ float warp_max[kFp8QuantBlockSize / 32];
  const float block_max = block_reduce_max(thread_max, warp_max);
  if (threadIdx.x == 0) {
    atomicMaxFloat(scale, block_max / quant_type_max_v<fp8_type>);
  }
}

// Per-tensor abs-max for strided or unaligned rows: one block per token.
template <typename scalar_t, typename fp8_type>
__global__ void __launch_bounds__(kFp8QuantBlockSize)
    segmented_max_reduction_strided(float* __restrict__ scale,
                                    const scalar_t* __restrict__ input,
                                    int hidden_size, int64_t in_row_stride,
                                    int64_t num_tokens) {
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const scalar_t* row_ptr = input + token_idx * in_row_stride;

  float thread_max = 0.0f;
  vectorize_read_with_alignment<kFp8QuantVecSize>(
      row_ptr, hidden_size, threadIdx.x, blockDim.x,
      [&] __device__(const scalar_t& v) {
        thread_max = fmaxf(thread_max, fabsf(static_cast<float>(v)));
      });

  __shared__ float warp_max[kFp8QuantBlockSize / 32];
  const float block_max = block_reduce_max(thread_max, warp_max);
  if (threadIdx.x == 0) {
    atomicMaxFloat(scale, block_max / quant_type_max_v<fp8_type>);
  }
}

template <typename scalar_t, typename fp8_type>
__global__ void scaled_fp8_quant_kernel_strided_dynamic(
    fp8_type* __restrict__ out, const scalar_t* __restrict__ input,
    const float* __restrict__ scale, int hidden_size, int64_t in_row_stride,
    int64_t out_row_stride) {
  const int64_t token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const scalar_t* token_in = input + token_idx * in_row_stride;
  fp8_type* token_out = out + token_idx * out_row_stride;

  const float reciprocal_scale = 1.0f / (*scale);
  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                    reciprocal_scale);
      });
}

// One block per token, one read of the row. Each thread keeps its
// kVecsPerThread vectors in registers across the abs-max reduction, so the row
// is read from global memory once instead of twice. Thread t owns vectors
// t, t + blockDim.x, ..., so a warp always touches consecutive vectors and
// every load and store is coalesced.
template <typename scalar_t, typename fp8_type, int kVecsPerThread>
__global__ void __launch_bounds__(kFp8QuantBlockSize)
    dynamic_per_token_scaled_fp8_quant_kernel_single_read(
        fp8_type* __restrict__ out, float* __restrict__ scale,
        const scalar_t* __restrict__ input, const float* __restrict__ scale_ub,
        int num_vecs, int64_t in_row_stride, int64_t out_row_stride) {
  using in_vec_t = fp8_quant_vec_t<scalar_t>;
  using out_vec_t = fp8_quant_vec_t<fp8_type>;

  const int64_t token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const auto* token_in =
      reinterpret_cast<const in_vec_t*>(input + token_idx * in_row_stride);
  auto* token_out =
      reinterpret_cast<out_vec_t*>(out + token_idx * out_row_stride);

  in_vec_t vecs[kVecsPerThread];
#pragma unroll
  for (int k = 0; k < kVecsPerThread; ++k) {
    const int v = tid + k * blockDim.x;
    if (v < num_vecs) {
      vecs[k] = token_in[v];
    }
  }

  float thread_max = 0.0f;
#pragma unroll
  for (int k = 0; k < kVecsPerThread; ++k) {
    const int v = tid + k * blockDim.x;
    if (v < num_vecs) {
#pragma unroll
      for (int i = 0; i < kFp8QuantVecSize; ++i) {
        thread_max =
            fmaxf(thread_max, fabsf(static_cast<float>(vecs[k].val[i])));
      }
    }
  }

  __shared__ float warp_max[kFp8QuantBlockSize / 32];
  const float block_max = block_reduce_max(thread_max, warp_max);

  // Every thread derives the same scale, so no broadcast is needed.
  float token_scale = scale_ub ? fminf(block_max, *scale_ub) : block_max;
  token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>,
                      min_scaling_factor<fp8_type>::val());
  if (tid == 0) {
    scale[token_idx] = token_scale;
  }

#pragma unroll
  for (int k = 0; k < kVecsPerThread; ++k) {
    const int v = tid + k * blockDim.x;
    if (v < num_vecs) {
      out_vec_t q;
#pragma unroll
      for (int i = 0; i < kFp8QuantVecSize; ++i) {
        q.val[i] = scaled_fp8_conversion<false, fp8_type>(
            static_cast<float>(vecs[k].val[i]), token_scale);
      }
      token_out[v] = q;
    }
  }
}

template <typename scalar_t, typename fp8_type, int kVecsPerThread>
void launch_dynamic_per_token_single_read(
    fp8_type* out, float* scale, const scalar_t* input, const float* scale_ub,
    int num_tokens, int num_vecs, int threads, int64_t in_row_stride,
    int64_t out_row_stride, cudaStream_t stream) {
  dynamic_per_token_scaled_fp8_quant_kernel_single_read<scalar_t, fp8_type,
                                                        kVecsPerThread>
      <<<num_tokens, threads, 0, stream>>>(
          out, scale, input, scale_ub, num_vecs, in_row_stride, out_row_stride);
}

// Generic two-pass path for rows that are unaligned, not a multiple of 16
// elements, or too long to hold in registers.
template <typename scalar_t, typename fp8_type>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel_strided(
    fp8_type* __restrict__ out, float* __restrict__ scale,
    const scalar_t* __restrict__ input, const float* __restrict__ scale_ub,
    int hidden_size, int64_t in_row_stride, int64_t out_row_stride) {
  const int64_t token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  // Use int64 to avoid overflowing an int32 when calculating this offset
  int64_t in_offset = static_cast<int64_t>(token_idx) * in_row_stride;
  int64_t out_offset = static_cast<int64_t>(token_idx) * out_row_stride;
  const scalar_t* token_in = input + in_offset;
  fp8_type* token_out = out + out_offset;

  // 1) per-token absmax
  float absmax_val = 0.f;
  vectorize_read_with_alignment<16>(
      token_in, hidden_size, tid, blockDim.x, [&] __device__(scalar_t v) {
        absmax_val = fmaxf(absmax_val, fabsf(static_cast<float>(v)));
      });

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  const float block_max =
      BlockReduce(tmp).Reduce(absmax_val, CubMaxOp{}, blockDim.x);

  __shared__ float token_scale;
  if (tid == 0) {
    token_scale = scale_ub ? fminf(block_max, *scale_ub) : block_max;
    token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>,
                        min_scaling_factor<fp8_type>::val());
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // 2) quantize
  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<false, fp8_type>(static_cast<float>(src),
                                                     token_scale);
      });
}

}  // namespace vllm

void static_scaled_fp8_quant(
    torch::stable::Tensor& out,          // [..., d]
    torch::stable::Tensor const& input,  // [..., d]
    torch::stable::Tensor const& scale,  // various shapes
    std::optional<torch::headeronly::IntHeaderOnlyArrayRef>
        opt_group_shape)  // optional explicit [group_m, group_n]
{
  STD_TORCH_CHECK(input.stride(-1) == 1,
                  "last dimension of input must be contiguous");
  STD_TORCH_CHECK(out.stride(-1) == 1,
                  "last dimension of output must be contiguous");

  const int hidden_size = input.size(-1);              // N (columns)
  const int num_tokens = input.numel() / hidden_size;  // M (rows)

  // Determine group_m, group_n, and scale strides from scale shape
  // Scale indexing: scale[gi * scale_stride_j + gj * scale_stride_i]
  // where gi = m / group_m, gj = n / group_n
  int group_m, group_n;
  int64_t scale_stride_i, scale_stride_j;

  if (scale.dim() == 0 || scale.numel() == 1) {
    // Per-tensor: one scale for the entire tensor
    group_m = num_tokens;
    group_n = hidden_size;
    scale_stride_i = 0;
    scale_stride_j = 0;
  } else if (scale.dim() == 1) {
    // 1D scale: require explicit group_shape to disambiguate per-channel vs
    // per-token (avoids edge case where num_tokens == hidden_size)
    STD_TORCH_CHECK(
        opt_group_shape.has_value(),
        "1D scale requires explicit group_shape to disambiguate "
        "per-channel vs per-token quantization. "
        "Use group_shape=(-1, 1) for per-channel or group_shape=(1, "
        "-1) for per-token.");
    STD_TORCH_CHECK(opt_group_shape->size() == 2,
                    "group_shape must have exactly 2 elements, got ",
                    opt_group_shape->size());

    const auto opt_group_m = (*opt_group_shape)[0];
    const auto opt_group_n = (*opt_group_shape)[1];
    group_m = opt_group_m == -1 ? num_tokens : static_cast<int>(opt_group_m);
    group_n = opt_group_n == -1 ? hidden_size : static_cast<int>(opt_group_n);

    // Validate the explicit group shape matches the 1D scale
    const int64_t scale_len = scale.numel();
    const int64_t expected_scale_m = num_tokens / group_m;
    const int64_t expected_scale_n = hidden_size / group_n;
    const int64_t expected_scale_numel = expected_scale_m * expected_scale_n;

    STD_TORCH_CHECK(scale_len == expected_scale_numel, "1D scale length (",
                    scale_len, ") does not match expected size (",
                    expected_scale_numel, ") for group_shape (", opt_group_m,
                    ", ", opt_group_n, ") with input shape (", num_tokens, ", ",
                    hidden_size, ")");

    // For 1D scale, determine strides based on which dim is trivial
    // Scale indexing: scale[gi * scale_stride_i + gj * scale_stride_j]
    // where gi = m / group_m (row group), gj = n / group_n (col group)
    if (expected_scale_m == 1) {
      // Per-channel style: one scale in M dim, scale varies along N
      // gi = 0 always, gj varies, so stride_1 traverses the scale
      scale_stride_i = 0;
      scale_stride_j = scale.stride(0);
    } else if (expected_scale_n == 1) {
      // Per-token style: one scale in N dim, scale varies along M
      // gj = 0 always, gi varies, so stride_0 traverses the scale
      scale_stride_i = scale.stride(0);
      scale_stride_j = 0;
    } else {
      STD_TORCH_CHECK(
          false,
          "1D scale can only be used when one of the scale dimensions is 1. "
          "For 2D group scaling, use a 2D scale tensor.");
    }
  } else if (scale.dim() == 2) {
    // 2D scale: infer group sizes from scale dimensions (or use explicit if
    // provided)
    const int64_t scale_size_0 = scale.size(0);
    const int64_t scale_size_1 = scale.size(1);

    STD_TORCH_CHECK(num_tokens % scale_size_0 == 0, "num_tokens (", num_tokens,
                    ") must be divisible by scale.size(0) (", scale_size_0,
                    ")");
    STD_TORCH_CHECK(hidden_size % scale_size_1 == 0, "hidden_size (",
                    hidden_size, ") must be divisible by scale.size(1) (",
                    scale_size_1, ")");

    // Infer from 2D scale shape
    int inferred_group_m = num_tokens / scale_size_0;
    int inferred_group_n = hidden_size / scale_size_1;

    // Use explicit if provided, otherwise use inferred
    if (opt_group_shape.has_value()) {
      STD_TORCH_CHECK(opt_group_shape->size() == 2,
                      "group_shape must have exactly 2 elements, got ",
                      opt_group_shape->size());
      const auto opt_group_m = (*opt_group_shape)[0];
      const auto opt_group_n = (*opt_group_shape)[1];
      group_m = opt_group_m == -1 ? num_tokens : static_cast<int>(opt_group_m);
      group_n = opt_group_n == -1 ? hidden_size : static_cast<int>(opt_group_n);

      // Validate explicit matches inferred
      STD_TORCH_CHECK(
          group_m == inferred_group_m && group_n == inferred_group_n,
          "Explicit group_shape (", opt_group_m, ", ", opt_group_n,
          ") does not match inferred group shape (", inferred_group_m, ", ",
          inferred_group_n, ") from 2D scale tensor shape (", scale_size_0,
          ", ", scale_size_1, ")");
    } else {
      group_m = inferred_group_m;
      group_n = inferred_group_n;
    }

    scale_stride_i = scale.stride(0);
    scale_stride_j = scale.stride(1);
  } else {
    STD_TORCH_CHECK(false, "scale must be 0D, 1D, or 2D tensor, but got ",
                    scale.dim(), "D");
  }

  const int block_size = 256;
  dim3 grid(num_tokens);
  dim3 block(block_size);

  const int64_t in_row_stride = input.stride(-2);
  const int64_t out_row_stride = out.stride(-2);

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  // Dispatch to template-specialized kernel based on stride pattern
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_STABLE_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              VLLM_STABLE_DISPATCH_BOOL(scale_stride_i == 0, S0_ZERO, [&] {
                VLLM_STABLE_DISPATCH_BOOL(scale_stride_j == 0, S1_ZERO, [&] {
                  vllm::scaled_fp8_quant_kernel_strided_group_shape<
                      scalar_t, fp8_t, S0_ZERO, S1_ZERO>
                      <<<grid, block, 0, stream>>>(
                          out.mutable_data_ptr<fp8_t>(),
                          input.const_data_ptr<scalar_t>(),
                          scale.const_data_ptr<float>(), hidden_size,
                          in_row_stride, out_row_stride, group_m, group_n,
                          scale_stride_i, scale_stride_j);
                });
              });
            });
      });
}

void dynamic_scaled_fp8_quant(torch::stable::Tensor& out,          // [..., d]
                              torch::stable::Tensor const& input,  // [..., d]
                              torch::stable::Tensor& scale)        // [1]
{
  STD_TORCH_CHECK(input.stride(-1) == 1,
                  "last dimension of input must be contiguous");
  STD_TORCH_CHECK(out.stride(-1) == 1,
                  "last dimension of output must be contiguous");

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int block_size = vllm::kFp8QuantBlockSize;
  dim3 grid(num_tokens);
  dim3 block(block_size);

  const int64_t in_row_stride = input.stride(-2);
  const int64_t out_row_stride = out.stride(-2);

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  // scale tensor should be initialised to <=0 before reduction
  STD_CUDA_CHECK(cudaMemsetAsync(scale.mutable_data_ptr<float>(), 0,
                                 sizeof(float), stream));

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_STABLE_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              const scalar_t* in_ptr = input.const_data_ptr<scalar_t>();
              const bool flat = in_row_stride == hidden_size &&
                                hidden_size % vllm::kFp8QuantVecSize == 0 &&
                                vllm::ptr_vec_aligned(in_ptr);
              if (flat) {
                const int64_t num_vecs = input.numel() / vllm::kFp8QuantVecSize;
                const int64_t max_blocks =
                    static_cast<int64_t>(
                        get_device_prop()->multiProcessorCount) *
                    8;
                const int64_t num_blocks = std::max<int64_t>(
                    1,
                    std::min<int64_t>(
                        max_blocks, (num_vecs + block_size - 1) / block_size));
                vllm::absmax_reduction_flat<scalar_t, fp8_t>
                    <<<num_blocks, block, 0, stream>>>(
                        scale.mutable_data_ptr<float>(), in_ptr, num_vecs);
              } else {
                vllm::segmented_max_reduction_strided<scalar_t, fp8_t>
                    <<<grid, block, 0, stream>>>(
                        scale.mutable_data_ptr<float>(), in_ptr, hidden_size,
                        in_row_stride, static_cast<int64_t>(num_tokens));
              }

              vllm::scaled_fp8_quant_kernel_strided_dynamic<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(out.mutable_data_ptr<fp8_t>(),
                                               input.const_data_ptr<scalar_t>(),
                                               scale.const_data_ptr<float>(),
                                               hidden_size, in_row_stride,
                                               out_row_stride);
            });
      });
}

void dynamic_per_token_scaled_fp8_quant(
    torch::stable::Tensor& out,          // [..., d]
    torch::stable::Tensor const& input,  // [..., d]
    torch::stable::Tensor& scales,
    std::optional<torch::stable::Tensor> const& scale_ub) {
  STD_TORCH_CHECK(input.stride(-1) == 1,
                  "last dimension of input must be contiguous");
  STD_TORCH_CHECK(out.stride(-1) == 1,
                  "last dimension of output must be contiguous");

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int block_size = vllm::kFp8QuantBlockSize;
  dim3 grid(num_tokens);

  const int64_t in_row_stride = input.stride(-2);
  const int64_t out_row_stride = out.stride(-2);

  const int num_vecs = hidden_size / vllm::kFp8QuantVecSize;
  const int warp_size = WARP_SIZE;

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "dynamic_per_token_scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_STABLE_DISPATCH_FP8_TYPES(
            out.scalar_type(),
            "dynamic_per_token_scaled_fp8_quant_kernel_fp8_type", [&] {
              const scalar_t* in_ptr = input.const_data_ptr<scalar_t>();
              fp8_t* out_ptr = out.mutable_data_ptr<fp8_t>();
              float* scales_ptr = scales.mutable_data_ptr<float>();
              const float* scale_ub_ptr =
                  scale_ub.has_value() ? scale_ub->const_data_ptr<float>()
                                       : nullptr;

              int single_read_threads = 0;
              int vecs_per_thread = 0;
              const bool single_read =
                  hidden_size % vllm::kFp8QuantVecSize == 0 && num_vecs > 0 &&
                  vllm::rows_vec_aligned(in_ptr, in_row_stride) &&
                  vllm::rows_vec_aligned(out_ptr, out_row_stride) &&
                  vllm::select_single_read_launch(
                      num_vecs, vllm::kFp8QuantMaxVecsPerThread<scalar_t>,
                      warp_size, block_size, single_read_threads,
                      vecs_per_thread);
              if (!single_read) {
                vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<
                    scalar_t, fp8_t>
                    <<<grid, dim3(std::min(hidden_size, block_size)), 0,
                       stream>>>(out_ptr, scales_ptr, in_ptr, scale_ub_ptr,
                                 hidden_size, in_row_stride, out_row_stride);
                return;
              }

              if (vecs_per_thread == 1) {
                vllm::launch_dynamic_per_token_single_read<scalar_t, fp8_t, 1>(
                    out_ptr, scales_ptr, in_ptr, scale_ub_ptr, num_tokens,
                    num_vecs, single_read_threads, in_row_stride,
                    out_row_stride, stream);
              } else if (vecs_per_thread == 2) {
                vllm::launch_dynamic_per_token_single_read<scalar_t, fp8_t, 2>(
                    out_ptr, scales_ptr, in_ptr, scale_ub_ptr, num_tokens,
                    num_vecs, single_read_threads, in_row_stride,
                    out_row_stride, stream);
              } else {
                vllm::launch_dynamic_per_token_single_read<scalar_t, fp8_t, 4>(
                    out_ptr, scales_ptr, in_ptr, scale_ub_ptr, num_tokens,
                    num_vecs, single_read_threads, in_row_stride,
                    out_row_stride, stream);
              }
            });
      });
}
