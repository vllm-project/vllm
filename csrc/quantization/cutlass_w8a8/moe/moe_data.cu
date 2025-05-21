#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <iostream>

constexpr uint64_t THREADS_PER_EXPERT = 512;

__global__ void compute_problem_sizes(const int* __restrict__ topk_ids,
                                      int32_t* problem_sizes1,
                                      int32_t* problem_sizes2,
                                      int32_t* atomic_buffer,
                                      const int topk_length, const int n,
                                      const int k) {
  int expert_id = blockIdx.x;

  int occurrences = 0;
  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    occurrences += (topk_ids[i] == expert_id);
  }
  atomicAdd(&atomic_buffer[expert_id], occurrences);
  __syncthreads();

  if (threadIdx.x == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    problem_sizes1[expert_id * 3] = final_occurrences;
    problem_sizes1[expert_id * 3 + 1] = 2 * n;
    problem_sizes1[expert_id * 3 + 2] = k;
    problem_sizes2[expert_id * 3] = final_occurrences;
    problem_sizes2[expert_id * 3 + 1] = k;
    problem_sizes2[expert_id * 3 + 2] = n;
  }
}

__global__ void compute_expert_offsets(
    const int32_t* __restrict__ problem_sizes1, int32_t* expert_offsets,
    int32_t* atomic_buffer, const int num_experts) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += problem_sizes1[i * 3];
    expert_offsets[i + 1] = tot_offset;
  }
}

__global__ void compute_arg_sorts(const int* __restrict__ topk_ids,
                                  const int32_t* __restrict__ expert_offsets,
                                  int32_t* input_permutation,
                                  int32_t* output_permutation,
                                  int32_t* atomic_buffer, const int topk_length,
                                  const int topk) {
  int const blk_expert_id = blockIdx.x;
  int const num_experts = gridDim.x;
  int32_t const num_tokens = expert_offsets[num_experts];

  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    int const expert_id = topk_ids[i];
    if (expert_id == -1 && blockIdx.x == 0) {
      // output_permutation is used to re-order the moe outputs. It is
      // used as c2 = c2[c_map], where c2 is a torch.tensor that is the
      // output of the cutlass kernels and c_map is the output_permutation.
      // c2 is initialized to zeros, therefore by setting the output_permutation
      // to num_tokens, we are guaranteed to fill the moe outputs to zero
      // for "invalid" topk_ids.
      output_permutation[i] = num_tokens;
    } else if (expert_id == blk_expert_id) {
      int start = atomicAdd(&atomic_buffer[expert_id], 1);
      input_permutation[start] = i / topk;
      output_permutation[i] = start;
    }
  }
}

void get_cutlass_moe_mm_data_caller(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k) {
  auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
  auto options_int32 =
      torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor atomic_buffer = torch::zeros(num_experts, options_int32);

  int num_threads = min(THREADS_PER_EXPERT, topk_ids.numel());
  compute_problem_sizes<<<num_experts, num_threads, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(problem_sizes2.data_ptr()),
      static_cast<int32_t*>(atomic_buffer.data_ptr()), topk_ids.numel(), n, k);
  compute_expert_offsets<<<1, 1, 0, stream>>>(
      static_cast<const int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(expert_offsets.data_ptr()),
      static_cast<int32_t*>(atomic_buffer.data_ptr()), num_experts);
  compute_arg_sorts<<<num_experts, num_threads, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<const int32_t*>(expert_offsets.data_ptr()),
      static_cast<int32_t*>(input_permutation.data_ptr()),
      static_cast<int32_t*>(output_permutation.data_ptr()),
      static_cast<int32_t*>(atomic_buffer.data_ptr()), topk_ids.numel(),
      topk_ids.size(1));
}
