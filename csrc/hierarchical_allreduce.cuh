/**
 * Hierarchical AllReduce for multi-node vLLM inference.
 *
 * Three-phase algorithm:
 *   Phase 1: Intra-node all-reduce  (existing vLLM custom AR via NVLink)
 *   Phase 2: Inter-node all-reduce  (UCCL-EP FIFO → CPU proxy → RDMA)
 *   Phase 3: Intra-node broadcast   (NVLink IPC read from gateway's buffer)
 *
 * UCCL-EP is an optional dependency gated behind VLLM_USE_UCCL_EP.
 * Without it, the code compiles but only single-node operation is supported.
 */

#pragma once

#include "custom_all_reduce.cuh"

#ifdef VLLM_USE_UCCL_EP
  #include <uccl_ep/fifo_channel.h>
  #include <uccl_ep/proxy.h>
  #include <uccl_ep/symmetric_mem.h>
#endif

#include <stdexcept>
#include <string>

namespace vllm {

// ─── Inter-node signal structure ───
// Lightweight signaling for Phase 2 → 3 transition.
// Allocated as IPC-shared within each node so non-gateway GPUs can
// observe phase2_done via NVLink reads.
struct HierSignal {
  alignas(128) uint32_t phase2_done;   // Set by gateway after inter-node AR
  alignas(128) uint32_t phase3_ready;  // Set after broadcast complete
};

// ─── Configuration ───
struct HierarchicalConfig {
  int local_rank;
  int local_world_size;  // GPUs per node
  int node_id;
  int num_nodes;
  int gateway_local_rank;  // Which local GPU is the gateway (default: 0)
  int max_size;            // Max allreduce message size in bytes
  int num_proxy_threads;   // CPU proxy threads per GPU (default: 4)
};

// ─────────────────────────────────────────────────
// Phase 3 kernel: Intra-node broadcast via NVLink IPC read
// ─────────────────────────────────────────────────
// Non-gateway GPUs read the global result from the gateway's
// IPC-shared broadcast buffer via NVLink.
template <typename T>
__global__ void intra_node_broadcast_kernel(
    T* __restrict__ dst,        // Local output buffer
    const T* __restrict__ src,  // Gateway's IPC broadcast buffer
    int num_elems) {
  using P = typename packed_t<T>::P;
  constexpr int VEC_SIZE = P::size;
  int num_vecs = num_elems / VEC_SIZE;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vecs;
       i += blockDim.x * gridDim.x) {
    reinterpret_cast<P*>(dst)[i] = reinterpret_cast<const P*>(src)[i];
  }
}

// ─── Helper kernels for phase synchronization ───

__global__ void signal_phase2_done(HierSignal* signal) {
  // Use threadfence_system to ensure visibility via NVLink to other GPUs
  __threadfence_system();
  volatile uint32_t* flag = &signal->phase2_done;
  *flag = 1;
  __threadfence_system();
}

__global__ void wait_for_phase2(HierSignal* gateway_signal) {
  volatile uint32_t* flag = &gateway_signal->phase2_done;
  while (*flag != 1) {
    // Spin — gateway is doing inter-node all-reduce
    // Bounded by network RTT + UCCL-EP overhead
  }
  __threadfence_system();
}

__global__ void reset_phase_signal(HierSignal* signal) {
  signal->phase2_done = 0;
  signal->phase3_ready = 0;
  __threadfence_system();
}

// ─────────────────────────────────────────────────
// Phase 2 kernel: Inter-node all-reduce (gateway GPU only)
// ─────────────────────────────────────────────────
// Design: For small messages (target: ≤8MB) with few nodes (2-8),
// use FLAT ALL-TO-ALL pattern:
//   - Each gateway writes its partial sum to every other gateway
//   - Barrier
//   - Each gateway locally reduces all received partial sums
//
// This wastes bandwidth (N× amplification) but minimizes latency.
#ifdef VLLM_USE_UCCL_EP
template <typename T>
__global__ void inter_node_allreduce_kernel(uccl_ep::FIFOChannel* fifos,
                                            int num_remote_nodes, T* send_buf,
                                            T** recv_bufs, T* output,
                                            int num_elems, int my_node_id,
                                            int num_nodes, uint64_t send_offset,
                                            uint64_t recv_base_offset) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int VEC_SIZE = P::size;

  // ─── Step 1: Push Write TransferCmds (single warp) ───
  if (threadIdx.x < 32 && blockIdx.x == 0) {
    int lane = threadIdx.x;

    if (lane == 0) {
      for (int fifo_idx = 0; fifo_idx < num_remote_nodes; fifo_idx++) {
        uccl_ep::TransferCmd cmd;
        cmd.type = uccl_ep::CMD_WRITE;
        cmd.src_offset = send_offset;
        cmd.dst_offset = recv_base_offset + my_node_id * num_elems * sizeof(T);
        cmd.length = num_elems * sizeof(T);
        cmd.seq_num = 0;
        cmd.flags = 0;
        fifos[fifo_idx].Push(cmd);
      }
    }
    __syncwarp();

    // ─── Step 2: Push Barrier and wait for completion ───
    uint32_t barrier_idx;
    if (lane == 0) {
      uccl_ep::TransferCmd barrier_cmd;
      barrier_cmd.type = uccl_ep::CMD_BARRIER;
      barrier_cmd.flags = uccl_ep::BARRIER_ALL_GATEWAYS;
      barrier_idx = fifos[0].Push(barrier_cmd);
    }
    barrier_idx = __shfl_sync(0xFFFFFFFF, barrier_idx, 0);

    if (lane == 0) {
      while (!fifos[0].CheckCompletion(barrier_idx)) {
        // spin — CPU proxy is establishing the barrier
      }
    }
    __syncwarp();
  }

  // Ensure all threads see the barrier completion
  // Note: using 1 block for simplicity (OK for small messages)
  __syncthreads();

  // ─── Step 3: Local reduce ───
  int num_vecs = num_elems / VEC_SIZE;

  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    A acc = upcast(reinterpret_cast<P*>(send_buf)[i]);

    for (int n = 0; n < num_remote_nodes; n++) {
      P remote = reinterpret_cast<P*>(recv_bufs[n])[i];
      packed_assign_add(acc, upcast(remote));
    }

    reinterpret_cast<P*>(output)[i] = downcast<P>(acc);
  }
}
#endif  // VLLM_USE_UCCL_EP

// ═══════════════════════════════════════════════════
// HierarchicalAllreduce class
// ═══════════════════════════════════════════════════

class HierarchicalAllreduce {
 public:
  // ─── Existing vLLM custom AR for intra-node ───
  CustomAllreduce* intra_ar_;

  // ─── Config ───
  HierarchicalConfig config_;
  bool is_gateway_;

#ifdef VLLM_USE_UCCL_EP
  // ─── UCCL-EP state for inter-node ───
  uccl_ep::ProxyHandle* proxy_;
  uccl_ep::FIFOChannel* fifos_;
  int num_fifos_;

  // Symmetric memory (RDMA-registered)
  // Dynamically sized to support arbitrary number of nodes.
  void* uccl_send_buf_;
  std::vector<void*> uccl_recv_bufs_;
  size_t symmetric_buf_size_;

  // Device pointer array for recv bufs (passed to kernel)
  void** recv_bufs_device_ptrs_;
#endif

  // ─── Intra-node broadcast buffer (IPC shared) ───
  void* broadcast_buf_;
  void* broadcast_ptrs_[8];

  // ─── Synchronization ───
  HierSignal* hier_signal_;
  HierSignal* hier_signal_ptrs_[8];

  // ─── Constructor ───
  HierarchicalAllreduce(CustomAllreduce* intra_ar, void* broadcast_ptrs[],
                        HierSignal* hier_signal_ptrs[],
                        const HierarchicalConfig& config)
      : intra_ar_(intra_ar),
        config_(config)
#ifdef VLLM_USE_UCCL_EP
        ,
        proxy_(nullptr),
        fifos_(nullptr),
        num_fifos_(0),
        uccl_send_buf_(nullptr),
        symmetric_buf_size_(0),
        recv_bufs_device_ptrs_(nullptr)
#endif
  {
    is_gateway_ = (config.local_rank == config.gateway_local_rank);

    // Store broadcast IPC pointers (all GPUs need these)
    for (int i = 0; i < config.local_world_size; i++) {
      broadcast_ptrs_[i] = broadcast_ptrs[i];
      hier_signal_ptrs_[i] = hier_signal_ptrs[i];
    }
    broadcast_buf_ = broadcast_ptrs[config.local_rank];
    hier_signal_ = hier_signal_ptrs[config.local_rank];

    // UCCL-EP initialization happens separately via init_uccl_ep()
  }

#ifdef VLLM_USE_UCCL_EP
  /**
   * Initialize UCCL-EP connections. Called only on gateway GPUs.
   */
  void init_uccl_ep(
      const std::vector<uccl_ep::ConnectionInfo>& remote_gateways) {
    if (!is_gateway_) return;

    // Spawn CPU proxy threads
    uccl_ep::ProxyConfig proxy_cfg;
    proxy_cfg.num_threads = config_.num_proxy_threads;
    proxy_cfg.gpu_id =
        config_.local_rank + config_.node_id * config_.local_world_size;
    proxy_ = uccl_ep::create_proxy(proxy_cfg);

    // Allocate symmetric memory (RDMA-registered)
    symmetric_buf_size_ = config_.max_size;
    num_fifos_ = config_.num_nodes - 1;
    uccl_recv_bufs_.resize(num_fifos_);
    uccl_ep::allocate_symmetric_mem(proxy_, symmetric_buf_size_,
                                    &uccl_send_buf_, uccl_recv_bufs_.data(),
                                    remote_gateways);

    // Create FIFO channels (one per remote node)
    fifos_ =
        uccl_ep::create_fifo_channels(proxy_, num_fifos_, /*kMaxInflight=*/64);

    // Connect FIFOs to remote gateways
    for (size_t i = 0; i < remote_gateways.size(); i++) {
      uccl_ep::connect_fifo(fifos_[i], remote_gateways[i]);
    }

    // Allocate device pointer array for recv bufs
    CUDACHECK(cudaMalloc(&recv_bufs_device_ptrs_, num_fifos_ * sizeof(void*)));
    CUDACHECK(cudaMemcpy(recv_bufs_device_ptrs_, uccl_recv_bufs_.data(),
                         num_fifos_ * sizeof(void*), cudaMemcpyHostToDevice));
  }
#endif

  // ─── Main hierarchical allreduce ───
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int num_elems) {
    int input_bytes = num_elems * sizeof(T);

    // ═══════════════════════════════════════════════
    // PHASE 1: Intra-node all-reduce
    // ═══════════════════════════════════════════════
    // Result: every local GPU has local_sum = Σ(all local GPUs' input)
    //
    // For the gateway, direct output to the UCCL-EP send buffer if available.
    // For non-gateway GPUs, output to broadcast_buf_ (temp storage).
    T* phase1_output;
    if (is_gateway_) {
#ifdef VLLM_USE_UCCL_EP
      if (config_.num_nodes > 1 && uccl_send_buf_ != nullptr) {
        phase1_output = reinterpret_cast<T*>(uccl_send_buf_);
      } else {
        phase1_output = output;
      }
#else
      phase1_output = output;
#endif
    } else {
      phase1_output = reinterpret_cast<T*>(broadcast_buf_);
    }

    intra_ar_->allreduce<T>(stream, input, phase1_output, num_elems);

    // If single node, we're done — just copy to output if needed
    if (config_.num_nodes <= 1) {
      if (phase1_output != output) {
        CUDACHECK(cudaMemcpyAsync(output, phase1_output, input_bytes,
                                  cudaMemcpyDeviceToDevice, stream));
      }
      return;
    }

    // ═══════════════════════════════════════════════
    // PHASE 2: Inter-node all-reduce (gateway only)
    // ═══════════════════════════════════════════════
#ifdef VLLM_USE_UCCL_EP
    if (is_gateway_) {
      // Gateway's Phase 1 result is in uccl_send_buf_ (RDMA-registered)
      // Launch inter-node reduce kernel (1 block for small messages)
      int threads =
          std::min(1024, ((num_elems / packed_t<T>::P::size) + 31) & ~31);
      threads = std::max(threads, 32);

      inter_node_allreduce_kernel<T><<<1, threads, 0, stream>>>(
          fifos_, num_fifos_, reinterpret_cast<T*>(uccl_send_buf_),
          reinterpret_cast<T**>(recv_bufs_device_ptrs_),
          reinterpret_cast<T*>(broadcast_ptrs_[config_.gateway_local_rank]),
          num_elems, config_.node_id, config_.num_nodes,
          /*send_offset=*/0,
          /*recv_base_offset=*/0);

      // Signal phase 2 done (visible to other local GPUs via NVLink)
      signal_phase2_done<<<1, 1, 0, stream>>>(hier_signal_);
    } else {
      // Non-gateway GPUs: wait for Phase 2 to complete
      wait_for_phase2<<<1, 1, 0, stream>>>(
          hier_signal_ptrs_[config_.gateway_local_rank]);
    }
#else
    // Without UCCL-EP, multi-node is not supported
    throw std::runtime_error(
        "Hierarchical AllReduce requires UCCL-EP for multi-node. "
        "Build with -DVLLM_USE_UCCL_EP=ON");
#endif

    // ═══════════════════════════════════════════════
    // PHASE 3: Intra-node broadcast
    // ═══════════════════════════════════════════════
    if (is_gateway_) {
      // Result is already in broadcast_buf_ for the gateway
      T* gateway_broadcast =
          reinterpret_cast<T*>(broadcast_ptrs_[config_.gateway_local_rank]);
      if (output != gateway_broadcast) {
        CUDACHECK(cudaMemcpyAsync(output, gateway_broadcast, input_bytes,
                                  cudaMemcpyDeviceToDevice, stream));
      }
    } else {
      // Non-gateway: read from gateway's broadcast buffer via NVLink
      int threads = 256;
      int num_vecs = num_elems / packed_t<T>::P::size;
      int blocks = std::min((num_vecs + threads - 1) / threads, 108);

      intra_node_broadcast_kernel<T><<<blocks, threads, 0, stream>>>(
          output,
          reinterpret_cast<T*>(broadcast_ptrs_[config_.gateway_local_rank]),
          num_elems);
    }

    // Reset signal for next iteration
    reset_phase_signal<<<1, 1, 0, stream>>>(hier_signal_);
  }

  ~HierarchicalAllreduce() {
#ifdef VLLM_USE_UCCL_EP
    if (is_gateway_ && proxy_) {
      if (recv_bufs_device_ptrs_) {
        cudaFree(recv_bufs_device_ptrs_);
      }
      uccl_ep::destroy_proxy(proxy_);
    }
#endif
  }
};

}  // namespace vllm
