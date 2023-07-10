#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Half.h>
#include <torch/extension.h>

#include "attention/attention_utils.cuh"
#include "reduction_utils.cuh"

namespace vllm {

template <typename scalar_t, int pack_size>
__global__ void rms_norm_kernel_impl(scalar_t *__restrict__ out,
                                     const scalar_t *__restrict__ input,
                                     const scalar_t *__restrict__ weight,
                                     const float epsilon, const int num_tokens,
                                     const int hidden_size) {
  int thread_group_width = hidden_size / pack_size;

  __shared__ float s_variance;
  float variance = 0.0f;
  const int vec_offset = blockIdx.x * hidden_size;

  using pack_vec = typename Vec<scalar_t, pack_size>::Type;
  scalar_t *out_ptr = out + vec_offset;
  const scalar_t *input_ptr = input + vec_offset;

  for (int idx = threadIdx.x; idx < thread_group_width; idx += blockDim.x) {
    pack_vec vec_input =
        *reinterpret_cast<const pack_vec *>(input_ptr + idx * pack_size);
    variance += dot(vec_input, vec_input);
  }

  variance = blockReduceSum<float>(variance);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < thread_group_width; idx += blockDim.x) {
    const int offset = idx * pack_size;
    pack_vec vec_input =
        *reinterpret_cast<const pack_vec *>(input_ptr + offset);
    pack_vec vec_weight = *reinterpret_cast<const pack_vec *>(weight + offset);
    pack_vec *vec_out = reinterpret_cast<pack_vec *>(out_ptr + offset);
    pack_vec hidden_states;
    if constexpr (std::is_same<scalar_t, __nv_bfloat16>::value) {
      hidden_states = mul<pack_vec, scalar_t, pack_vec>(
          __float2bfloat16(s_variance), vec_input);
    } else if constexpr (std::is_same<scalar_t, uint16_t>::value) {
      hidden_states = mul<pack_vec, scalar_t, pack_vec>(
          float_to_half(s_variance), vec_input);
    } else if constexpr (std::is_same<scalar_t, float>::value) {
      hidden_states = mul<pack_vec, scalar_t, pack_vec>(s_variance, vec_input);
    }
    *vec_out = mul<pack_vec>(hidden_states, vec_weight);
  }
}

template <typename scalar_t>
void rms_norm_kernel_launcher(torch::Tensor &out, torch::Tensor &input,
                              torch::Tensor &weight, float epsilon) {

  int num_tokens = input.size(0);
  int hidden_size = input.size(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int pack_size = 16 / sizeof(scalar_t);

  scalar_t *out_ptr = reinterpret_cast<scalar_t *>(out.data_ptr());
  scalar_t *input_ptr = reinterpret_cast<scalar_t *>(input.data_ptr());
  scalar_t *weight_ptr = reinterpret_cast<scalar_t *>(weight.data_ptr());

  if (hidden_size % pack_size == 0) {
    dim3 grid(num_tokens);
    dim3 block(min(((hidden_size / pack_size + 31) / 32) * 32, 1024));
    vllm::rms_norm_kernel_impl<scalar_t, pack_size><<<grid, block, 0, stream>>>(
        out_ptr, input_ptr, weight_ptr, epsilon, num_tokens, hidden_size);
  } else {
    dim3 grid(num_tokens);
    dim3 block(min(((hidden_size / 1 + 31) / 32) * 32, 1024));
    vllm::rms_norm_kernel_impl<scalar_t, 1><<<grid, block, 0, stream>>>(
        out_ptr, input_ptr, weight_ptr, epsilon, num_tokens, hidden_size);
  }
}

} // namespace vllm

// TODO(sleepcoo): opt use shared memory.
#define CALL_RMSNORM_LAUNCHER_BY_TYPE(scalar_t)                                \
  vllm::rms_norm_kernel_launcher<scalar_t>(out, input, weight, epsilon);

void rms_norm(torch::Tensor &out,    // [num_tokens, hidden_size]
              torch::Tensor &input,  // [num_tokens, hidden_size]
              torch::Tensor &weight, // [hidden_size]
              float epsilon) {

  if (input.dtype() == at::ScalarType::Float) {
    CALL_RMSNORM_LAUNCHER_BY_TYPE(float);
  } else if (input.dtype() == at::ScalarType::Half) {
    CALL_RMSNORM_LAUNCHER_BY_TYPE(uint16_t);
  } else if (input.dtype() == at::ScalarType::BFloat16) {
    CALL_RMSNORM_LAUNCHER_BY_TYPE(__nv_bfloat16);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", input.dtype());
  }
}
