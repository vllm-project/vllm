/**
 * C++ bindings for HierarchicalAllreduce.
 * Exposes init, allreduce, dispose, and signal_size to Python via PyTorch ops.
 */

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include "hierarchical_allreduce.cuh"

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

/**
 * Initialize a HierarchicalAllreduce instance.
 *
 * Args:
 *   intra_ar_ptr: Pointer to existing CustomAllreduce (from init_custom_ar)
 *   broadcast_ptrs: IPC pointers for broadcast buffer (within node)
 *   hier_signal_ptrs: IPC pointers for hierarchical signals
 *   local_rank, local_world_size, node_id, num_nodes: topology info
 *   gateway_local_rank: which local GPU is the gateway (default: 0)
 *   max_size: max allreduce message size in bytes
 *   num_proxy_threads: CPU proxy threads per gateway GPU
 */
fptr_t init_hierarchical_ar(fptr_t intra_ar_ptr,
                            const std::vector<fptr_t>& broadcast_ptrs,
                            const std::vector<fptr_t>& hier_signal_ptrs,
                            int64_t local_rank, int64_t local_world_size,
                            int64_t node_id, int64_t num_nodes,
                            int64_t gateway_local_rank, int64_t max_size,
                            int64_t num_proxy_threads) {
  auto intra_ar = reinterpret_cast<vllm::CustomAllreduce*>(intra_ar_ptr);

  TORCH_CHECK(local_world_size > 0 && local_world_size <= 8,
              "HierarchicalAllreduce: local_world_size (GPUs per node) must be "
              "1-8 (inherited from CustomAllreduce's fixed-size arrays), got ",
              local_world_size);

  // Convert broadcast pointers
  void* bcast_ptrs[8];
  vllm::HierSignal* sig_ptrs[8];
  for (int i = 0; i < local_world_size; i++) {
    bcast_ptrs[i] = reinterpret_cast<void*>(broadcast_ptrs[i]);
    sig_ptrs[i] = reinterpret_cast<vllm::HierSignal*>(hier_signal_ptrs[i]);
  }

  vllm::HierarchicalConfig config;
  config.local_rank = local_rank;
  config.local_world_size = local_world_size;
  config.node_id = node_id;
  config.num_nodes = num_nodes;
  config.gateway_local_rank = gateway_local_rank;
  config.max_size = max_size;
  config.num_proxy_threads = num_proxy_threads;

  return (fptr_t) new vllm::HierarchicalAllreduce(intra_ar, bcast_ptrs,
                                                  sig_ptrs, config);
}

/**
 * Perform hierarchical all-reduce.
 */
void hierarchical_all_reduce(fptr_t _har, torch::Tensor& inp,
                             torch::Tensor& out) {
  auto har = reinterpret_cast<vllm::HierarchicalAllreduce*>(_har);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());

  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      har->allreduce<float>(stream, reinterpret_cast<float*>(inp.data_ptr()),
                            reinterpret_cast<float*>(out.data_ptr()),
                            out.numel());
      break;
    case at::ScalarType::Half:
      har->allreduce<half>(stream, reinterpret_cast<half*>(inp.data_ptr()),
                           reinterpret_cast<half*>(out.data_ptr()),
                           out.numel());
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      har->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel());
      break;
#endif
    default:
      throw std::runtime_error(
          "hierarchical allreduce only supports float32, float16, "
          "and bfloat16");
  }
}

/**
 * Dispose (delete) a HierarchicalAllreduce instance.
 */
void dispose_hierarchical_ar(fptr_t _har) {
  delete reinterpret_cast<vllm::HierarchicalAllreduce*>(_har);
}

/**
 * Return sizeof(HierSignal) for Python-side buffer allocation.
 */
int64_t hier_signal_size() { return sizeof(vllm::HierSignal); }
