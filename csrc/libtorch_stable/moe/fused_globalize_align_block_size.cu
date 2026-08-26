// Single-launch fusion of the DeepEP-v2 decode metadata chain
// (_globalize_recv_topk_idx + moe_align_block_size + count_and_sort) for the
// humming INDEXED / CUDA-graph decode path. recv values are LOCAL expert ids;
// the kernel bins by local id while storing back the global id, sized by local
// experts. Hybrid launch: one non-cooperative block for small decode, else a
// cooperative multi-SM grid (launched via cudaLaunchKernelEx so grid.sync() is
// CUDA-graph capturable). Barriers: zero+sentinel -> shared histogram
// (distributed reduce) -> block-0 prefix + expert_ids fill -> scatter+globalize.

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "libtorch_stable/torch_utils.h"

namespace cg = cooperative_groups;

namespace vllm {
namespace moe {

static constexpr int FGAS_THREADS = 256;
static constexpr int FGAS_SB_THREADS = 1024;
static constexpr long FGAS_SB_MAX_WORK = 8192;

// COOP selects the phase barrier: grid.sync() (multi-block) vs __syncthreads().
// topk / block_size are runtime args: any (topk, block_size) is supported, and
// the kernel is barrier/memory-bound so the divisions are not on the hot path.
template <bool COOP>
__device__ __forceinline__ void fused_gas_body(
    long* __restrict__ topk_idx, int* __restrict__ sorted_ids,
    int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_pad,
    int* __restrict__ counts, int num_recv, int rank_expert_offset,
    int global_num_experts, int numel, int max_num_tokens_padded,
    int max_num_m_blocks, int local_num_experts, int topk, int block_size) {
  extern __shared__ int sh[];  // [local_num_experts] per-block histogram
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  auto BAR = [&]() {
    if constexpr (COOP)
      cg::this_grid().sync();
    else
      __syncthreads();
  };

  // Phase 0: zero the histogram + sentinel-fill sorted_ids.
  for (int e = tid; e < local_num_experts; e += stride) counts[e] = 0;
  for (int i = tid; i < max_num_tokens_padded; i += stride) sorted_ids[i] = numel;
  BAR();

  // Phase 1: per-block shared histogram, one global atomic per (block, expert).
  // Reduce stays distributed across blocks (centralizing it in block 0 is slower).
  for (int e = threadIdx.x; e < local_num_experts; e += blockDim.x) sh[e] = 0;
  __syncthreads();
  for (int i = tid; i < numel; i += stride) {
    int val = (int)topk_idx[i];
    int row = i / topk;
    bool valid = (val >= 0) && (val < local_num_experts) &&
                 (val + rank_expert_offset < global_num_experts) &&
                 (row < num_recv);
    if (valid) atomicAdd(&sh[val], 1);
  }
  __syncthreads();
  for (int e = threadIdx.x; e < local_num_experts; e += blockDim.x)
    if (sh[e]) atomicAdd(&counts[e], sh[e]);
  BAR();

  // Phase 2 (block 0). Thread 0 runs the tiny serial prefix scan; then all of
  // block 0's threads split the expert_ids fill grid-stride.
  if (blockIdx.x == 0) {
    int* sh_start = sh;  // [local_num_experts] padded start offsets
    int& sh_run = sh[local_num_experts];
    if (threadIdx.x == 0) {
      int run = 0;
      for (int e = 0; e < local_num_experts; e++) {
        int c = counts[e];
        int nb = (c + block_size - 1) / block_size;
        sh_start[e] = run;
        counts[e] = run;  // exclusive prefix; empty experts skip
        run += nb * block_size;
      }
      sh_run = run;
      *num_tokens_post_pad = run;
    }
    __syncthreads();
    int run = sh_run;
    // owner(j) = highest expert whose padded start <= j*block_size (skips empty
    // experts); blocks past the last token get -1.
    for (int j = threadIdx.x; j < max_num_m_blocks; j += blockDim.x) {
      int off = j * block_size;
      if (off >= run) {
        expert_ids[j] = -1;
        continue;
      }
      int owner = 0;
      for (int e = 1; e < local_num_experts; e++)
        if (sh_start[e] <= off) owner = e;
      expert_ids[j] = owner;
    }
  }
  BAR();

  // Phase 3: scatter token indices at cursor[val]++ and globalize in place.
  for (int i = tid; i < numel; i += stride) {
    int val = (int)topk_idx[i];
    int g = val + rank_expert_offset;
    int row = i / topk;
    bool valid = (val >= 0) && (val < local_num_experts) &&
                 (g < global_num_experts) && (row < num_recv);
    if (valid) {
      int pos = atomicAdd(&counts[val], 1);
      sorted_ids[pos] = i;
    }
    topk_idx[i] = valid ? g : -1;
  }
}

template <bool COOP>
__global__ void fused_gas_kernel(long* __restrict__ topk_idx,
                                 const int* __restrict__ psum,
                                 int* __restrict__ sorted_ids,
                                 int* __restrict__ expert_ids,
                                 int* __restrict__ num_tokens_post_pad,
                                 int* __restrict__ counts, int P,
                                 int rank_expert_offset, int global_num_experts,
                                 int numel, int max_num_tokens_padded,
                                 int max_num_m_blocks, int local_num_experts,
                                 int topk, int block_size) {
  // num_recv read on-device (psum[P-1]) -> baked into replay, cudagraph-safe.
  fused_gas_body<COOP>(
      topk_idx, sorted_ids, expert_ids, num_tokens_post_pad, counts, psum[P - 1],
      rank_expert_offset, global_num_experts, numel, max_num_tokens_padded,
      max_num_m_blocks, local_num_experts, topk, block_size);
}

static int fgas_sm_count() {
  // Device-constant: query the driver once, never on the capture/replay path.
  static int sm_count = -1;
  if (sm_count < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (sm_count < 1) sm_count = 1;
  }
  return sm_count;
}

static int fgas_cooperative_blocks(int bps, long work) {
  int max_blocks = bps * fgas_sm_count();
  if (max_blocks < 1) max_blocks = 1;
  long need = (work + FGAS_THREADS - 1) / FGAS_THREADS;  // no over-launch
  if (need < 1) need = 1;
  return need < (long)max_blocks ? (int)need : max_blocks;
}

static void fgas_launch(long* p_topk, const int* p_psum, int* p_sorted,
                        int* p_expert, int* p_num, int* p_counts, int P,
                        int reo, int gne, int numel, int mntp, int mnmb,
                        int local_e, int topk, int block_size,
                        cudaStream_t stream) {
  long work = numel > mntp ? numel : mntp;
  // Histogram [local_e] + one scalar (phase-2 padded-run total).
  size_t smem = (size_t)(local_e + 1) * sizeof(int);

  // Small decode: one non-cooperative block, __syncthreads barriers. Skips the
  // cooperative-launch + grid.sync floor that dominates when work is tiny.
  if (work <= FGAS_SB_MAX_WORK) {
    fused_gas_kernel<false><<<1, FGAS_SB_THREADS, smem, stream>>>(
        p_topk, p_psum, p_sorted, p_expert, p_num, p_counts, P, reo, gne, numel,
        mntp, mnmb, local_e, topk, block_size);
    STD_CUDA_CHECK(cudaGetLastError());
    return;
  }

  // Large decode: cooperative multi-SM. Occupancy (fixed per rank) queried once.
  static int bps = -1;
  if (bps < 0)
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bps, (const void*)fused_gas_kernel<true>, FGAS_THREADS, smem);
  int blocks = fgas_cooperative_blocks(bps, work);

  // Cooperative launch via cudaLaunchKernelEx + the cooperative launch
  // attribute: unlike the legacy cudaLaunchCooperativeKernel, this form IS
  // stream-capturable, so grid.sync() works inside a captured CUDA graph.
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(blocks);
  config.blockDim = dim3(FGAS_THREADS);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeCooperative;
  attr.val.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  STD_CUDA_CHECK(cudaLaunchKernelEx(&config, fused_gas_kernel<true>, p_topk,
                                    p_psum, p_sorted, p_expert, p_num, p_counts,
                                    P, reo, gne, numel, mntp, mnmb, local_e,
                                    topk, block_size));
}

}  // namespace moe
}  // namespace vllm

// recv_topk_idx (int64, [N, topk]) is globalized in place and the align/sort
// outputs (int32) are filled. counts is an internal write-cursor allocated here.
void fused_globalize_align_block_size(
    torch::stable::Tensor topk_idx, torch::stable::Tensor psum,
    int64_t rank_expert_offset, int64_t global_num_experts,
    int64_t local_num_experts, int64_t block_size,
    torch::stable::Tensor sorted_ids, torch::stable::Tensor expert_ids,
    torch::stable::Tensor num_tokens_post_pad) {
  STD_TORCH_CHECK(topk_idx.scalar_type() == torch::headeronly::ScalarType::Long,
                  "fused_globalize_align_block_size: topk_idx must be int64");
  STD_TORCH_CHECK(local_num_experts <= 1024,
                  "fused_globalize_align_block_size: local_num_experts <= 1024");

  const torch::stable::accelerator::DeviceGuard device_guard(
      topk_idx.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(topk_idx.get_device_index());

  int numel = (int)topk_idx.numel();
  int topk = (int)topk_idx.size(1);
  int mntp = (int)sorted_ids.size(0);
  int mnmb = (int)expert_ids.size(0);
  int local_e = (int)local_num_experts;
  int reo = (int)rank_expert_offset;
  int gne = (int)global_num_experts;
  int P = (int)psum.size(0);

  // Internal per-local-expert histogram / write-cursor.
  torch::stable::Tensor counts = torch::stable::new_empty(
      topk_idx, {local_num_experts}, torch::headeronly::ScalarType::Int);

  long* p_topk = reinterpret_cast<long*>(topk_idx.mutable_data_ptr());
  const int* p_psum = reinterpret_cast<const int*>(psum.const_data_ptr());
  int* p_sorted = reinterpret_cast<int*>(sorted_ids.mutable_data_ptr());
  int* p_expert = reinterpret_cast<int*>(expert_ids.mutable_data_ptr());
  int* p_num = reinterpret_cast<int*>(num_tokens_post_pad.mutable_data_ptr());
  int* p_counts = reinterpret_cast<int*>(counts.mutable_data_ptr());

  vllm::moe::fgas_launch(p_topk, p_psum, p_sorted, p_expert, p_num, p_counts, P,
                         reo, gne, numel, mntp, mnmb, local_e, topk,
                         (int)block_size, stream);
}
