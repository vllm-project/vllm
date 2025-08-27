#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

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

  const int32_t* topk_ptr = static_cast<const int32_t*>(topk_ids.data_ptr());
  int32_t* ps1_ptr = static_cast<int32_t*>(problem_sizes1.data_ptr());
  int32_t* ps2_ptr = static_cast<int32_t*>(problem_sizes2.data_ptr());
  int32_t* atomic_ptr = static_cast<int32_t*>(atomic_buffer.data_ptr());

  if (swap_ab) {
    compute_problem_sizes<true><<<num_experts, num_threads, 0, stream>>>(
        topk_ptr, ps1_ptr, ps2_ptr, atomic_ptr,
        static_cast<int>(topk_ids.numel()), static_cast<int>(n),
        static_cast<int>(k));
  } else {
    compute_problem_sizes<false><<<num_experts, num_threads, 0, stream>>>(
        topk_ptr, ps1_ptr, ps2_ptr, atomic_ptr,
        static_cast<int>(topk_ids.numel()), static_cast<int>(n),
        static_cast<int>(k));
  }
}
}  // namespace

void get_cutlass_moe_mm_problem_sizes_caller(
    const torch::Tensor& topk_ids, torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2, const int64_t num_experts, const int64_t n,
    const int64_t k, const std::optional<torch::Tensor>& blockscale_offsets) {
  auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
  auto options_int32 =
      torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor atomic_buffer = torch::zeros(num_experts, options_int32);

  // Swap-AB should be disabled for FP4 path
  bool may_swap_ab = (!blockscale_offsets.has_value()) &&
                     (topk_ids.numel() <= SWAP_AB_THRESHOLD);

  launch_compute_problem_sizes(topk_ids, problem_sizes1, problem_sizes2,
                               atomic_buffer, num_experts, n, k, stream,
                               may_swap_ab);
}

void get_cutlass_moe_mm_data_caller(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k,
    const std::optional<torch::Tensor>& blockscale_offsets, bool force_no_swap,
    bool should_fuse) {
  auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
  auto options_int32 =
      torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor atomic_buffer = torch::zeros(num_experts, options_int32);

  int num_threads = min(THREADS_PER_EXPERT, topk_ids.numel());

  // Swap-AB should be disabled for FP4 path
  bool may_swap_ab = !force_no_swap && (!blockscale_offsets.has_value()) &&
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

__device__ inline void cp_async1_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  const int BYTES = 4;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// Async copy fence.
__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

template <int THREAD_COUNT>
__global__ void transpose_a_scales(float* __restrict__ a_scales_t,
                                   const float* __restrict__ a_scales,
                                   const int32_t* __restrict__ expert_offsets,
                                   const int32_t* __restrict__ problem_sizes,
                                   uint32_t k_scaled) {
  uint32_t expert_idx = blockIdx.x;

  static constexpr uint32_t WARP_SIZE = 32;
  static constexpr uint32_t WARP_COUNT = THREAD_COUNT / WARP_SIZE;

  __shared__ float s_block[THREAD_COUNT];

  uint32_t lane_id = threadIdx.x & 0x1fu;
  uint32_t warp_id = threadIdx.x / WARP_SIZE;
  uint32_t* s_32 = reinterpret_cast<uint32_t*>(s_block);

  if (!lane_id) {
    if (warp_id == 0) {
      s_32[0] = expert_offsets[expert_idx];
    } else {
      s_32[1] = problem_sizes[expert_idx * 3];
    }
  }

  __syncthreads();

  uint32_t expert_offset_scaled = s_32[0] * k_scaled;
  const uint32_t num_tokens = s_32[1];

  auto a_scales_t_ptr = a_scales_t + expert_offset_scaled;
  auto _a_scales_ptr = a_scales + expert_offset_scaled;

  float* s_block_load_ptr = s_block + warp_id * WARP_SIZE + lane_id;
  const float* s_block_write_ptr = s_block + lane_id * WARP_COUNT + warp_id;

  auto t = warp_id;
  uint32_t t_base = 0;
  auto transpose = [&]() {
    uint32_t k = lane_id;
    auto a_scales_ptr = _a_scales_ptr + t * k_scaled + k;

    auto tile_x = t / WARP_SIZE;
    auto y = tile_x * WARP_SIZE + lane_id;
    bool pred_y = y < num_tokens;

    auto num_k_tiles = CEILDIV(k_scaled, WARP_SIZE);

    auto x = warp_id;
    for (uint32_t k_tile = 0; k_tile < num_k_tiles; k_tile++, k += WARP_SIZE) {
      bool pred = k < k_scaled && t < num_tokens;
      cp_async1_pred(s_block_load_ptr, a_scales_ptr, pred);
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();

      if (x < k_scaled && pred_y) {
        auto a_idx = x * num_tokens + y;
        a_scales_t_ptr[a_idx] = *s_block_write_ptr;
      }

      __syncthreads();
      x += WARP_SIZE;
      a_scales_ptr += WARP_SIZE;
    }
  };

  while (t - warp_id < num_tokens) {
    // All threads are able to execute this.
    transpose();
    t_base += WARP_COUNT;
    t += WARP_COUNT;
  }

  if (t >= num_tokens) {
    return;
  }

  transpose();
}

torch::Tensor transpose_cutlass_moe_a_scales_caller(
    torch::Tensor& a_scales, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes) {
  const int64_t num_experts = expert_offsets.size(0);
  const int64_t num_tokens = a_scales.size(0);
  const int32_t k_scaled = a_scales.size(1);

  auto options =
      torch::TensorOptions().dtype(a_scales.dtype()).device(a_scales.device());
  torch::Tensor a_scales_t = torch::empty(num_tokens * k_scaled, options);

  auto stream = at::cuda::getCurrentCUDAStream(expert_offsets.device().index());

  static constexpr int MAX_THREADS = 1024;

  auto num_threads = MAX_THREADS;

  transpose_a_scales<MAX_THREADS><<<num_experts, num_threads, 0, stream>>>(
      static_cast<float*>(a_scales_t.data_ptr()),
      static_cast<const float*>(a_scales.data_ptr()),
      static_cast<const int32_t*>(expert_offsets.data_ptr()),
      static_cast<const int32_t*>(problem_sizes.data_ptr()), k_scaled);
  return a_scales_t;
}
