#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "dispatch_utils.h"

#include <iostream>

constexpr uint64_t THREADS_PER_EXPERT = 512;
// threshold must match the dispatch logic in run_cutlass_moe_mm_sm90()
constexpr int SWAP_AB_THRESHOLD = 64;

template <bool SWAP_AB>
__global__ void compute_problem_sizes(const int32_t* __restrict__ topk_ids,
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
    if constexpr (!SWAP_AB) {
      problem_sizes1[expert_id * 3] = final_occurrences;
      problem_sizes1[expert_id * 3 + 1] = 2 * n;
      problem_sizes1[expert_id * 3 + 2] = k;
      problem_sizes2[expert_id * 3] = final_occurrences;
      problem_sizes2[expert_id * 3 + 1] = k;
      problem_sizes2[expert_id * 3 + 2] = n;
    } else {
      problem_sizes1[expert_id * 3] = 2 * n;
      problem_sizes1[expert_id * 3 + 1] = final_occurrences;
      problem_sizes1[expert_id * 3 + 2] = k;
      problem_sizes2[expert_id * 3] = k;
      problem_sizes2[expert_id * 3 + 1] = final_occurrences;
      problem_sizes2[expert_id * 3 + 2] = n;
    }
  }
}

__global__ void compute_expert_offsets(
    const int32_t* __restrict__ problem_sizes1, int32_t* expert_offsets,
    int32_t* atomic_buffer, const int num_experts, const bool swap_ab) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += swap_ab ? problem_sizes1[i * 3 + 1] : problem_sizes1[i * 3];
    expert_offsets[i + 1] = tot_offset;
  }
}

__global__ void compute_expert_blockscale_offsets(
    const int32_t* __restrict__ problem_sizes1, int32_t* expert_offsets,
    int32_t* blockscale_offsets, int32_t* atomic_buffer, const int num_experts,
    const bool swap_ab) {
  int32_t tot_offset = 0;
  int32_t tot_offset_round = 0;
  expert_offsets[0] = 0;
  blockscale_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    int32_t cur_offset =
        swap_ab ? problem_sizes1[i * 3 + 1] : problem_sizes1[i * 3];
    atomic_buffer[i] = tot_offset;
    tot_offset += cur_offset;
    expert_offsets[i + 1] = tot_offset;
    tot_offset_round += (cur_offset + (128 - 1)) / 128 * 128;
    blockscale_offsets[i + 1] = tot_offset_round;
  }
}

__global__ void compute_arg_sorts(const int32_t* __restrict__ topk_ids,
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

namespace {
inline void launch_compute_problem_sizes(const torch::Tensor& topk_ids,
                                         torch::Tensor& problem_sizes1,
                                         torch::Tensor& problem_sizes2,
                                         torch::Tensor& atomic_buffer,
                                         int64_t num_experts, int64_t n,
                                         int64_t k, cudaStream_t stream,
                                         const bool swap_ab) {
  int num_threads = min(THREADS_PER_EXPERT, topk_ids.numel());

  auto const* topk_ptr = topk_ids.data_ptr<int32_t>();
  auto* ps1_ptr = problem_sizes1.data_ptr<int32_t>();
  auto* ps2_ptr = problem_sizes2.data_ptr<int32_t>();
  auto* atomic_ptr = atomic_buffer.data_ptr<int32_t>();

  VLLM_DISPATCH_BOOL(swap_ab, SwapAB, [&] {
    compute_problem_sizes<SwapAB><<<num_experts, num_threads, 0, stream>>>(
        topk_ptr, ps1_ptr, ps2_ptr, atomic_ptr,
        static_cast<int>(topk_ids.numel()), static_cast<int>(n),
        static_cast<int>(k));
  });
}
}  // namespace

template <bool SWAP_AB>
__global__ void compute_problem_sizes_from_expert_offsets(
    const int64_t* __restrict__ expert_first_token_offset,
    int32_t* __restrict__ problem_sizes1, int32_t* __restrict__ problem_sizes2,
    const int num_experts, const int n, const int k) {
  int const expert_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert_id >= num_experts) {
    return;
  }

  int64_t const m64 = expert_first_token_offset[expert_id + 1] -
                      expert_first_token_offset[expert_id];
  int32_t const m = static_cast<int32_t>(m64);

  int32_t* ps1 = problem_sizes1 + expert_id * 3;
  int32_t* ps2 = problem_sizes2 + expert_id * 3;

  if constexpr (!SWAP_AB) {
    // [M, 2*N, K]
    ps1[0] = m;
    ps1[1] = 2 * n;
    ps1[2] = k;
    // [M, K, N]
    ps2[0] = m;
    ps2[1] = k;
    ps2[2] = n;
  } else {
    // swap logical M/N in the problem shape
    // [2*N, M, K]
    ps1[0] = 2 * n;
    ps1[1] = m;
    ps1[2] = k;
    // [K, M, N]
    ps2[0] = k;
    ps2[1] = m;
    ps2[2] = n;
  }
}

void get_cutlass_moe_mm_problem_sizes_from_expert_offsets_caller(
    const torch::Tensor& expert_first_token_offset,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    const int64_t n, const int64_t k, const bool swap_ab) {
  TORCH_CHECK(expert_first_token_offset.is_cuda(),
              "expert_first_token_offset must be a CUDA tensor");
  TORCH_CHECK(expert_first_token_offset.dtype() == torch::kInt64,
              "expert_first_token_offset must be int64");

  TORCH_CHECK(problem_sizes1.is_cuda() && problem_sizes2.is_cuda(),
              "problem_sizes must be CUDA tensors");
  TORCH_CHECK(problem_sizes1.dtype() == torch::kInt32 &&
                  problem_sizes2.dtype() == torch::kInt32,
              "problem_sizes must be int32");
  TORCH_CHECK(problem_sizes1.is_contiguous() && problem_sizes2.is_contiguous(),
              "problem_sizes must be contiguous");
  TORCH_CHECK(problem_sizes1.dim() == 2 && problem_sizes2.dim() == 2,
              "problem_sizes must be 2D tensors");
  TORCH_CHECK(problem_sizes1.size(1) == 3 && problem_sizes2.size(1) == 3,
              "problem_sizes second dim must be 3");
  TORCH_CHECK(problem_sizes1.sizes() == problem_sizes2.sizes(),
              "problem_sizes1 and problem_sizes2 must have same shape");

  int64_t const num_experts64 = problem_sizes1.size(0);
  TORCH_CHECK(expert_first_token_offset.numel() == num_experts64 + 1,
              "expert_first_token_offset must have num_experts + 1 elements");
  TORCH_CHECK(num_experts64 <= INT32_MAX, "num_experts must fit in int32");
  TORCH_CHECK(n <= INT32_MAX && k <= INT32_MAX, "n and k must fit in int32");

  int const num_experts = static_cast<int>(num_experts64);
  auto stream = at::cuda::getCurrentCUDAStream(
      expert_first_token_offset.device().index());

  int const threads = (num_experts < 256) ? num_experts : 256;
  int const blocks = (num_experts + threads - 1) / threads;

  auto const* offsets_ptr = expert_first_token_offset.data_ptr<int64_t>();
  auto* ps1_ptr = problem_sizes1.data_ptr<int32_t>();
  auto* ps2_ptr = problem_sizes2.data_ptr<int32_t>();

  VLLM_DISPATCH_BOOL(swap_ab, SwapAB, [&] {
    compute_problem_sizes_from_expert_offsets<SwapAB>
        <<<blocks, threads, 0, stream>>>(offsets_ptr, ps1_ptr, ps2_ptr,
                                         num_experts, static_cast<int>(n),
                                         static_cast<int>(k));
  });
}

void get_cutlass_moe_mm_data_caller(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k,
    const std::optional<torch::Tensor>& blockscale_offsets) {
  auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
  auto options_int32 =
      torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor atomic_buffer = torch::zeros(num_experts, options_int32);

  int num_threads = min(THREADS_PER_EXPERT, topk_ids.numel());

  // Swap-AB should be disabled for FP4 path
  bool may_swap_ab = (!blockscale_offsets.has_value()) &&
                     (topk_ids.numel() <= SWAP_AB_THRESHOLD);

  launch_compute_problem_sizes(topk_ids, problem_sizes1, problem_sizes2,
                               atomic_buffer, num_experts, n, k, stream,
                               may_swap_ab);

  if (blockscale_offsets.has_value()) {
    // fp4 path
    compute_expert_blockscale_offsets<<<1, 1, 0, stream>>>(
        static_cast<const int32_t*>(problem_sizes1.data_ptr()),
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(blockscale_offsets.value().data_ptr()),
        static_cast<int32_t*>(atomic_buffer.data_ptr()), num_experts,
        may_swap_ab);
  } else {
    compute_expert_offsets<<<1, 1, 0, stream>>>(
        static_cast<const int32_t*>(problem_sizes1.data_ptr()),
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(atomic_buffer.data_ptr()), num_experts,
        may_swap_ab);
  }
  compute_arg_sorts<<<num_experts, num_threads, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<const int32_t*>(expert_offsets.data_ptr()),
      static_cast<int32_t*>(input_permutation.data_ptr()),
      static_cast<int32_t*>(output_permutation.data_ptr()),
      static_cast<int32_t*>(atomic_buffer.data_ptr()), topk_ids.numel(),
      topk_ids.size(1));
}

template <bool SWAP_AB>
__global__ void compute_pplx_data(int32_t* expert_offsets,
                                  int32_t* problem_sizes1,
                                  int32_t* problem_sizes2,
                                  const int32_t* __restrict__ expert_num_tokens,
                                  const int padded_m, const int n,
                                  const int k) {
  int expert_idx = threadIdx.x;
  expert_offsets[expert_idx] = expert_idx * padded_m;

  if constexpr (!SWAP_AB) {
    problem_sizes1[expert_idx * 3] = expert_num_tokens[expert_idx];
    problem_sizes1[expert_idx * 3 + 1] = 2 * n;
    problem_sizes1[expert_idx * 3 + 2] = k;
    problem_sizes2[expert_idx * 3] = expert_num_tokens[expert_idx];
    problem_sizes2[expert_idx * 3 + 1] = k;
    problem_sizes2[expert_idx * 3 + 2] = n;
  } else {
    problem_sizes1[expert_idx * 3] = 2 * n;
    problem_sizes1[expert_idx * 3 + 1] = expert_num_tokens[expert_idx];
    problem_sizes1[expert_idx * 3 + 2] = k;
    problem_sizes2[expert_idx * 3] = k;
    problem_sizes2[expert_idx * 3 + 1] = expert_num_tokens[expert_idx];
    problem_sizes2[expert_idx * 3 + 2] = n;
  }
}

void get_cutlass_pplx_moe_mm_data_caller(torch::Tensor& expert_offsets,
                                         torch::Tensor& problem_sizes1,
                                         torch::Tensor& problem_sizes2,
                                         const torch::Tensor& expert_num_tokens,
                                         const int64_t num_local_experts,
                                         const int64_t padded_m,
                                         const int64_t n, const int64_t k) {
  auto stream = at::cuda::getCurrentCUDAStream(expert_offsets.device().index());

  if (num_local_experts * padded_m > SWAP_AB_THRESHOLD) {
    compute_pplx_data<false><<<1, num_local_experts, 0, stream>>>(
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(problem_sizes1.data_ptr()),
        static_cast<int32_t*>(problem_sizes2.data_ptr()),
        static_cast<const int32_t*>(expert_num_tokens.data_ptr()), padded_m, n,
        k);
  } else {
    compute_pplx_data<true><<<1, num_local_experts, 0, stream>>>(
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(problem_sizes1.data_ptr()),
        static_cast<int32_t*>(problem_sizes2.data_ptr()),
        static_cast<const int32_t*>(expert_num_tokens.data_ptr()), padded_m, n,
        k);
  }
}