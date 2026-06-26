#include <numeric>

#include "torch_utils.h"

#include "../cub_helpers.h"
#include "../core/batch_invariant.hpp"
#include "../type_convert.cuh"
#include "dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS, bool HasWeight>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const int64_t input_stride_d2,        // input.stride(-2)
    const int64_t input_stride_d3,        // input.stride(-3)
    const int64_t input_stride_d4,        // input.stride(-4)
    const int64_t input_shape_d2,         // input.size(-2)
    const int64_t input_shape_d3,         // input.size(-3)
    const scalar_t* __restrict__ weight,  // [hidden_size] or
                                          // [num_groups, hidden_size];
                                          // null if !HasWeight
    const int64_t weight_stride,          // 0 or weight.stride(0)
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  const scalar_t* input_row;
  const scalar_t* weight_row;
  int64_t weight_row_off = 0;
  if constexpr (NUM_DIMS == 2) {
    // 2D for layernorm normal case [batch_size, hidden]
    input_row = input + blockIdx.x * input_stride_d2;
    weight_row = weight + blockIdx.x * weight_stride;
  } else if constexpr (NUM_DIMS == 3) {
    // 3D for q/k norm [batch_size, num_heads, head_size]
    int batch_idx = blockIdx.x / input_shape_d2;
    int head_idx = blockIdx.x % input_shape_d2;
    input_row =
        input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
    weight_row = weight + batch_idx * weight_stride;
  } else if constexpr (NUM_DIMS == 4) {
    // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
    int batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
    int remaining = blockIdx.x % (input_shape_d3 * input_shape_d2);
    int seq_idx = remaining / input_shape_d2;
    int head_idx = remaining % input_shape_d2;
    input_row = input + batch_idx * input_stride_d4 +
                seq_idx * input_stride_d3 + head_idx * input_stride_d2;
    weight_row = weight + batch_idx * weight_stride;
  }

  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };
  vllm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t* out_row = out + blockIdx.x * hidden_size;
  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight_row);
  auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(out_row);
  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
    vec_n_t<scalar_t, VEC_SIZE> dst;
    vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
    vec_n_t<scalar_t, VEC_SIZE> src2;
    if constexpr (HasWeight) {
      src2 = v_w[i];
    }
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(src1.val[j]);
      if constexpr (HasWeight) {
        float w = static_cast<float>(src2.val[j]);
        dst.val[j] = static_cast<scalar_t>(x * s_variance * w);
      } else {
        dst.val[j] = static_cast<scalar_t>(x * s_variance);
      }
    }
    v_out[i] = dst;
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width, bool HasWeight>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size], null if !HasWeight
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
    _f16Vec<scalar_t, width> res = residual_v[id];
    _f16Vec<scalar_t, width> out;
    using Converter = _typeConvert<scalar_t>;
    if constexpr (HasWeight) {
      _f16Vec<scalar_t, width> w = weight_v[idx];
#pragma unroll
      for (int j = 0; j < width; ++j) {
        float x = Converter::convert(res.data[j]);
        float wf = Converter::convert(w.data[j]);
        out.data[j] = Converter::convert(x * s_variance * wf);
      }
    } else {
#pragma unroll
      for (int j = 0; j < width; ++j) {
        float x = Converter::convert(res.data[j]);
        out.data[j] = Converter::convert(x * s_variance);
      }
    }
    input_v[strided_id] = out;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width, bool HasWeight>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size], null if !HasWeight
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
    if constexpr (HasWeight) {
      float w = (float)weight[idx];
      input[blockIdx.x * input_stride + idx] = (scalar_t)(x * s_variance * w);
    } else {
      input[blockIdx.x * input_stride + idx] = (scalar_t)(x * s_variance);
    }
  }
}

}  // namespace vllm

void rms_norm(torch::stable::Tensor& out,    // [..., hidden_size]
              torch::stable::Tensor& input,  // [..., hidden_size]
              std::optional<torch::stable::Tensor> weight, double epsilon) {
  STD_TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1) {
    input = torch::stable::contiguous(input);
  }
  STD_TORCH_CHECK(input.stride(-1) == 1);
  int64_t weight_stride = 0;
  if (weight.has_value()) {
    STD_TORCH_CHECK(weight->is_contiguous());
    if (weight->dim() == 1) {
      STD_TORCH_CHECK(weight->size(0) == input.size(-1));
    } else if (weight->dim() == 2) {
      STD_TORCH_CHECK(weight->size(0) == input.size(0));
      STD_TORCH_CHECK(weight->size(-1) == input.size(-1));
      weight_stride = weight->stride(0);
    } else {
      STD_TORCH_CHECK(false, "rms_norm weight must be 1D or 2D");
    }
  }

  int hidden_size = input.size(-1);

  int num_tokens = input.numel() / hidden_size;
  int num_dims = input.dim();
  int64_t input_stride_d2 = input.stride(-2);
  int64_t input_stride_d3 = (num_dims >= 3) ? input.stride(-3) : 0;
  int64_t input_stride_d4 = (num_dims >= 4) ? input.stride(-4) : 0;
  int64_t input_shape_d2 = (num_dims >= 3) ? input.size(-2) : 0;
  int64_t input_shape_d3 = (num_dims >= 4) ? input.size(-3) : 0;

  // For large num_tokens, use smaller blocks to increase SM concurrency.
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  const bool has_weight = weight.has_value();
  VLLM_STABLE_DISPATCH_RANK234(num_dims, [&] {
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "rms_norm_kernel", [&] {
          const scalar_t* weight_ptr =
              has_weight ? weight->const_data_ptr<scalar_t>() : nullptr;
          const int calculated_vec_size =
              std::gcd(16 / sizeof(scalar_t), hidden_size);
          const int block_size =
              std::min(hidden_size / calculated_vec_size, max_block_size);
          dim3 block(block_size);
          VLLM_STABLE_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {
            if (has_weight) {
              vllm::rms_norm_kernel<scalar_t, vec_size, tensor_rank, true>
                  <<<grid, block, 0, stream>>>(
                      out.mutable_data_ptr<scalar_t>(),
                      input.const_data_ptr<scalar_t>(), input_stride_d2,
                      input_stride_d3, input_stride_d4, input_shape_d2,
                      input_shape_d3, weight_ptr, weight_stride, epsilon,
                      num_tokens, hidden_size);
            } else {
              vllm::rms_norm_kernel<scalar_t, vec_size, tensor_rank, false>
                  <<<grid, block, 0, stream>>>(
                      out.mutable_data_ptr<scalar_t>(),
                      input.const_data_ptr<scalar_t>(), input_stride_d2,
                      input_stride_d3, input_stride_d4, input_shape_d2,
                      input_shape_d3, weight_ptr, /*weight_stride=*/0, epsilon,
                      num_tokens, hidden_size);
            }
          });
        });
  });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width, has_weight)                       \
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                     \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {              \
        if (has_weight) {                                                  \
          vllm::fused_add_rms_norm_kernel<scalar_t, width, true>           \
              <<<grid, block, 0, stream>>>(                                \
                  input.mutable_data_ptr<scalar_t>(), input_stride,        \
                  residual.mutable_data_ptr<scalar_t>(),                   \
                  weight->const_data_ptr<scalar_t>(), epsilon, num_tokens, \
                  hidden_size);                                            \
        } else {                                                           \
          vllm::fused_add_rms_norm_kernel<scalar_t, width, false>          \
              <<<grid, block, 0, stream>>>(                                \
                  input.mutable_data_ptr<scalar_t>(), input_stride,        \
                  residual.mutable_data_ptr<scalar_t>(), nullptr, epsilon, \
                  num_tokens, hidden_size);                                \
        }                                                                  \
      });

void fused_add_rms_norm(torch::stable::Tensor& input,     // [..., hidden_size]
                        torch::stable::Tensor& residual,  // [..., hidden_size]
                        std::optional<torch::stable::Tensor> weight,
                        double epsilon) {
  STD_TORCH_CHECK(input.scalar_type() == residual.scalar_type());
  STD_TORCH_CHECK(residual.is_contiguous());
  if (weight.has_value()) {
    STD_TORCH_CHECK(weight->scalar_type() == input.scalar_type());
    STD_TORCH_CHECK(weight->is_contiguous());
  }
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
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  constexpr int vector_width = 8;
  constexpr int req_alignment_bytes = vector_width * 2;
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input_stride % vector_width == 0;
  bool batch_invariant_launch = vllm::vllm_is_batch_invariant();
  const bool has_weight = weight.has_value();
  if (has_weight) {
    auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight->data_ptr());
    bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 &&
                            res_ptr % req_alignment_bytes == 0 &&
                            wt_ptr % req_alignment_bytes == 0;
    if (ptrs_are_aligned && offsets_are_multiple_of_vector_width &&
        !batch_invariant_launch) {
      LAUNCH_FUSED_ADD_RMS_NORM(8, true);
    } else {
      LAUNCH_FUSED_ADD_RMS_NORM(0, true);
    }
  } else {
    bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 &&
                            res_ptr % req_alignment_bytes == 0;
    if (ptrs_are_aligned && offsets_are_multiple_of_vector_width &&
        !batch_invariant_launch) {
      LAUNCH_FUSED_ADD_RMS_NORM(8, false);
    } else {
      LAUNCH_FUSED_ADD_RMS_NORM(0, false);
    }
  }
}
