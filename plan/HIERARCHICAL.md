

# Enhancing vLLM Custom All-Reduce with UCCL-EP for Hierarchical Inter-Node Communication

## 1. Architectural Alignment: Why UCCL-EP Fits

The key realization is that vLLM's custom all-reduce and UCCL-EP solve complementary halves of the same problem:

```
vLLM Custom AR                          UCCL-EP
──────────────                          ───────
Optimized for: small messages            Optimized for: small messages (tokens)
Latency strategy: waste bandwidth,       Latency strategy: GPU-initiated,
  flat all-to-all read-reduce              fine-grained overlap
Transport: NVLink direct pointer          Transport: GPU→FIFO→CPU proxy→RDMA
  access (intra-node only)                 (inter-node, portable)
Sync: spin on signal flags               Sync: Push/Check-completion + Barrier
  in shared GPU memory                     TransferCmd
```

Both share a critical design principle: **GPU-initiated communication with minimal synchronization overhead**. vLLM's custom AR has the GPU directly dereference remote GPU pointers via NVLink. UCCL-EP has the GPU push 128-bit commands to a FIFO, and the CPU proxy handles the rest. In both cases, the GPU does not block waiting for a CPU to decide what to send.

The hierarchical design combines both:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     HIERARCHICAL ALL-REDUCE                             │
│                                                                         │
│  INTRA-NODE (NVLink)           INTER-NODE (Network)                    │
│  ┌─────────────────────┐       ┌─────────────────────────────────────┐ │
│  │ vLLM Custom AR       │       │ UCCL-EP                             │ │
│  │                      │       │                                     │ │
│  │ GPU₀ ←NVLink→ GPU₁  │       │ GPU₀ →FIFO→ CPU →RDMA→ Remote GPU  │ │
│  │ GPU₂ ←NVLink→ GPU₃  │       │                                     │ │
│  │                      │       │ Direct pointer NOT possible         │ │
│  │ Direct pointer access│       │ 128-bit TransferCmd + CPU proxy     │ │
│  │ IPC shared buffers   │       │ GPUDirect RDMA (data stays on GPU)  │ │
│  └─────────────────────┘       └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Three-Phase Algorithm

```
═══════════════════════════════════════════════════════════════════════
  Node 0 (GPUs 0-3)                    Node 1 (GPUs 4-7)
═══════════════════════════════════════════════════════════════════════

Phase 1: INTRA-NODE REDUCE (vLLM Custom AR, NVLink)
  
  GPU₀: x₀    GPU₁: x₁               GPU₄: x₄    GPU₅: x₅
  GPU₂: x₂    GPU₃: x₃               GPU₆: x₆    GPU₇: x₇
       │           │                        │           │
       └─────┬─────┘                        └─────┬─────┘
             ▼                                    ▼
  All GPUs: S₀ = x₀+x₁+x₂+x₃        All GPUs: S₁ = x₄+x₅+x₆+x₇

  Mechanism: existing vLLM custom_all_reduce.cuh
             (flat read-reduce via NVLink IPC pointers)

═══════════════════════════════════════════════════════════════════════

Phase 2: INTER-NODE ALL-REDUCE (UCCL-EP, Gateway GPUs Only)
  
  Gateway GPU₀                          Gateway GPU₄
  has S₀                                has S₁
       │                                     │
       │  ┌───────────────────────────────┐  │
       └──► UCCL-EP FIFO → CPU Proxy     ◄──┘
          │ → RDMA Write (GPUDirect)     │
          │ → Barrier                     │
          │ → Local Reduce               │
          └───────────────────────────────┘
       │                                     │
       ▼                                     ▼
  Gateway GPU₀: G = S₀+S₁              Gateway GPU₄: G = S₀+S₁

  Mechanism: UCCL-EP TransferCmds
    1. Push Write: send S₀ to remote gateway's receive buffer
    2. Push Barrier: wait for all gateways to finish writing  
    3. Local reduce: G = S₀ + (received S₁)

═══════════════════════════════════════════════════════════════════════

Phase 3: INTRA-NODE BROADCAST (NVLink Direct Write)
  
  Gateway GPU₀: writes G                Gateway GPU₄: writes G
    to IPC broadcast buffer                to IPC broadcast buffer
       │                                     │
  Signal barrier (vLLM Signal)          Signal barrier (vLLM Signal)
       │                                     │
       ▼                                     ▼
  All GPUs on Node 0                    All GPUs on Node 1
  read G from gateway's                 read G from gateway's
  IPC buffer via NVLink                 IPC buffer via NVLink

  Mechanism: reuse vLLM's IPC shared buffer + signal mechanism
             (same cudaIpcOpenMemHandle infrastructure)

═══════════════════════════════════════════════════════════════════════
  Result: All GPUs on all nodes have G = x₀+x₁+...+x₇  ✓
═══════════════════════════════════════════════════════════════════════
```

---

## 3. How to Enhance the Existing Custom All-Reduce

### 3.1 Memory Layout Extension

The existing vLLM custom AR allocates two sets of IPC-shared buffers:
- `meta_ptrs`: Signal metadata + temp storage
- `buffer_ptrs`: Pre-registered data staging buffer

For hierarchical AR, we need additional buffers for the UCCL-EP inter-node step:

```
EXISTING (intra-node only):
┌─────────────────────────────────────────────────────┐
│ meta_ptrs[0..N-1]      Signal + temp allreduce buf  │ IPC shared
│ buffer_ptrs[0..N-1]    Data staging buffer          │ IPC shared
│ rank_data              Pointer lookup table          │ Local
└─────────────────────────────────────────────────────┘

NEW (hierarchical additions):
┌─────────────────────────────────────────────────────┐
│ uccl_ep_send_buf       Registered with UCCL-EP      │ GPUDirect RDMA
│                        (symmetric memory)            │ registered
│ uccl_ep_recv_bufs[0..M-1]  One per remote node      │ GPUDirect RDMA
│                            (symmetric memory)        │ registered
│ broadcast_buf          IPC shared within node        │ IPC shared
│                        Gateway writes, others read   │ (existing infra)
│ fifo_channels[0..M-1] UCCL-EP FIFOs to CPU proxy   │ Host memory
│                        One per remote node           │ (GPU writes,
│                                                      │  CPU reads)
│ hier_signals           Phase sync within node        │ IPC shared
└─────────────────────────────────────────────────────┘

Where N = local_world_size (GPUs per node)
      M = num_nodes
```

### 3.2 C++ Layer: New `HierarchicalAllreduce` Class

```cpp
// hierarchical_allreduce.cuh

#pragma once

#include "custom_all_reduce.cuh"
#include <uccl_ep/fifo_channel.h>    // UCCL-EP FIFO API
#include <uccl_ep/proxy.h>           // UCCL-EP CPU proxy management
#include <uccl_ep/symmetric_mem.h>   // UCCL-EP symmetric memory

namespace vllm {

// ─── Inter-node signal structure ───
// Lightweight signaling for Phase 2→3 transition
struct HierSignal {
  alignas(128) uint32_t phase2_done;  // Set by gateway after inter-node AR
  alignas(128) uint32_t phase3_ready; // Set by all after broadcast complete
};

// ─── Configuration ───
struct HierarchicalConfig {
  int local_rank;
  int local_world_size;     // GPUs per node
  int node_id;
  int num_nodes;
  int gateway_local_rank;   // Which local GPU is the gateway (default: 0)
  int max_size;             // Max allreduce message size
  int num_proxy_threads;    // CPU proxy threads per GPU (default: 4)
};

class HierarchicalAllreduce {
 public:
  // ─── Existing vLLM custom AR for intra-node ───
  CustomAllreduce* intra_ar_;
  
  // ─── UCCL-EP state for inter-node ───
  uccl_ep::ProxyHandle* proxy_;         // CPU proxy handle (runs proxy threads)
  uccl_ep::FIFOChannel* fifos_;         // Array of FIFO channels
  int num_fifos_;                        // One per remote node (for gateway)
  
  // ─── Symmetric memory (RDMA-registered) ───
  void* uccl_send_buf_;                  // Local send buffer (RDMA registered)
  void* uccl_recv_bufs_[8];             // Receive buffers (one per remote node)
  size_t symmetric_buf_size_;
  
  // ─── Intra-node broadcast buffer (IPC shared) ───
  void* broadcast_buf_;                  // Gateway writes here after Phase 2
  void* broadcast_ptrs_[8];             // IPC pointers from all local GPUs
  
  // ─── Synchronization ───
  HierSignal* hier_signal_;             // IPC-shared signal for phase transitions
  HierSignal* hier_signal_ptrs_[8];     // IPC pointers from all local GPUs
  
  // ─── Config ───
  HierarchicalConfig config_;
  bool is_gateway_;
  
  // ─── Constructor ───
  HierarchicalAllreduce(
      // Intra-node: reuse existing custom AR
      CustomAllreduce* intra_ar,
      // UCCL-EP connection info
      const std::vector<uccl_ep::ConnectionInfo>& remote_gateways,
      // Buffers (allocated and IPC-exchanged by Python layer)
      void* broadcast_ptrs[],
      HierSignal* hier_signal_ptrs[],
      // Config
      const HierarchicalConfig& config)
      : intra_ar_(intra_ar), config_(config) {
    
    is_gateway_ = (config.local_rank == config.gateway_local_rank);
    
    if (is_gateway_) {
      // ── Initialize UCCL-EP CPU proxy ──
      // This spawns proxy threads that poll FIFOs and post RDMA
      uccl_ep::ProxyConfig proxy_cfg;
      proxy_cfg.num_threads = config.num_proxy_threads;
      proxy_cfg.gpu_id = config.local_rank + config.node_id * config.local_world_size;
      proxy_ = uccl_ep::create_proxy(proxy_cfg);
      
      // ── Establish RDMA connections to remote gateways ──
      // Symmetric memory: all gateways allocate same-sized buffers
      symmetric_buf_size_ = config.max_size;
      uccl_ep::allocate_symmetric_mem(
          proxy_, symmetric_buf_size_,
          &uccl_send_buf_,       // local send buffer
          uccl_recv_bufs_,       // receive slots (one per remote node)
          remote_gateways);
      
      // ── Create FIFO channels (one per remote node) ──
      num_fifos_ = config.num_nodes - 1;
      fifos_ = uccl_ep::create_fifo_channels(
          proxy_, num_fifos_,
          /* kMaxInflight = */ 64);
      
      // ── Connect FIFOs to remote gateways ──
      for (int i = 0; i < remote_gateways.size(); i++) {
        uccl_ep::connect_fifo(fifos_[i], remote_gateways[i]);
      }
    }
    
    // ── Store broadcast IPC pointers (all GPUs need these) ──
    for (int i = 0; i < config.local_world_size; i++) {
      broadcast_ptrs_[i] = broadcast_ptrs[i];
      hier_signal_ptrs_[i] = hier_signal_ptrs[i];
    }
    broadcast_buf_ = broadcast_ptrs[config.local_rank];
    hier_signal_ = hier_signal_ptrs[config.local_rank];
  }
  
  // ─── Main hierarchical allreduce ───
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int num_elems);
  
  ~HierarchicalAllreduce() {
    if (is_gateway_ && proxy_) {
      uccl_ep::destroy_proxy(proxy_);
    }
  }
};

}  // namespace vllm
```

### 3.3 The GPU Kernel: Three-Phase Orchestration

This is the core kernel. It's launched on **all GPUs**, but only the gateway GPU executes Phase 2.

```cuda
// hierarchical_allreduce_kernels.cuh

namespace vllm {

// ─────────────────────────────────────────────────
// Phase 1: Intra-node all-reduce (reuses existing)
// ─────────────────────────────────────────────────
// This is NOT a separate kernel — we call intra_ar_->allreduce()
// which launches vLLM's existing custom AR kernel.
// After Phase 1, every local GPU has: local_sum = Σ(local GPUs' data)

// ─────────────────────────────────────────────────
// Phase 2: Inter-node all-reduce (gateway GPU only)
// ─────────────────────────────────────────────────
//
// Design decision: For small messages (target: ≤8MB) with few nodes
// (2-8), use FLAT ALL-TO-ALL pattern:
//   - Each gateway writes its partial sum to every other gateway
//   - Barrier
//   - Each gateway locally reduces all received partial sums
//
// This wastes bandwidth (N× amplification) but minimizes latency
// (single round of RDMA writes + one barrier), matching vLLM's
// intra-node philosophy.

template <typename T>
__global__ void inter_node_allreduce_kernel(
    // UCCL-EP FIFO channels (one per remote node)
    uccl_ep::FIFOChannel* fifos,
    int num_remote_nodes,
    
    // Source: result from Phase 1 (in RDMA-registered send buffer)
    T* send_buf,
    
    // Receive buffers: one per remote node (RDMA-registered)
    // Remote gateways write into these via GPUDirect RDMA
    T** recv_bufs,         // Array of pointers
    
    // Output: where to write the global sum
    T* output,
    
    // Metadata
    int num_elems,
    int my_node_id,
    int num_nodes,
    
    // Offsets in symmetric memory
    uint64_t send_offset,
    uint64_t recv_base_offset) {
  
  // ─── Step 1: Push Write TransferCmds (single warp handles this) ───
  // Only warp 0 does FIFO operations; other warps wait.
  if (threadIdx.x < 32 && blockIdx.x == 0) {  // warp 0 of block 0
    int lane = threadIdx.x;
    
    if (lane == 0) {
      // Push one Write command per remote node
      for (int fifo_idx = 0; fifo_idx < num_remote_nodes; fifo_idx++) {
        uccl_ep::TransferCmd cmd;
        cmd.type = uccl_ep::CMD_WRITE;
        // Remote gateway will receive our data in their recv slot for us
        cmd.src_offset = send_offset;
        cmd.dst_offset = recv_base_offset + my_node_id * num_elems * sizeof(T);
        cmd.length = num_elems * sizeof(T);
        cmd.seq_num = 0;  // single round, no ordering needed beyond barrier
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
    
    // All lanes in warp 0 spin on barrier completion
    if (lane == 0) {
      while (!fifos[0].CheckCompletion(barrier_idx)) {
        // spin — CPU proxy is establishing the barrier
        // Other warps/blocks on this GPU are idle during this wait
        // This is acceptable for small messages where the network
        // RTT dominates anyway
      }
    }
    __syncwarp();
  }
  
  // Ensure all threads see the barrier completion
  __syncthreads();
  // Note: Need grid-level sync here. Options:
  //   a) Use cooperative groups grid sync (requires occupancy 1)
  //   b) Split into two kernels (barrier between them)
  //   c) Use only 1 block for the entire operation (OK for small messages)
  // For simplicity and small messages, we use 1 block.
  
  // ─── Step 3: Local reduce — sum my partial sum with all received ones ───
  // Vectorized for performance (16-byte loads)
  constexpr int VEC_SIZE = 16 / sizeof(T);
  int num_vecs = num_elems / VEC_SIZE;
  
  for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    // Start with our own partial sum
    using VecT = typename VecType<T, VEC_SIZE>::type;
    VecT acc = reinterpret_cast<VecT*>(send_buf)[i];
    
    // Add each remote node's contribution
    for (int n = 0; n < num_remote_nodes; n++) {
      VecT remote = reinterpret_cast<VecT*>(recv_bufs[n])[i];
      // Element-wise add (need to unpack VecT)
      T* acc_elems = reinterpret_cast<T*>(&acc);
      T* remote_elems = reinterpret_cast<T*>(&remote);
      #pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        acc_elems[j] += remote_elems[j];
      }
    }
    
    reinterpret_cast<VecT*>(output)[i] = acc;
  }
}


// ─────────────────────────────────────────────────
// Phase 3: Intra-node broadcast (via NVLink IPC)
// ─────────────────────────────────────────────────
// Gateway wrote the global result to broadcast_buf (an IPC-shared buffer).
// All other GPUs read from gateway's broadcast_buf via NVLink.

template <typename T>
__global__ void intra_node_broadcast_kernel(
    T* __restrict__ dst,      // Local output buffer
    const T* __restrict__ src, // Gateway's IPC-shared broadcast buffer
    int num_elems) {
  
  constexpr int VEC_SIZE = 16 / sizeof(T);
  using VecT = typename VecType<T, VEC_SIZE>::type;
  int num_vecs = num_elems / VEC_SIZE;
  
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < num_vecs; 
       i += blockDim.x * gridDim.x) {
    // Direct NVLink read from gateway's buffer
    reinterpret_cast<VecT*>(dst)[i] = 
        reinterpret_cast<const VecT*>(src)[i];
  }
}

}  // namespace vllm
```

### 3.4 The Orchestrator: Putting the Three Phases Together

```cpp
// In hierarchical_allreduce.cuh

namespace vllm {

template <typename T>
void HierarchicalAllreduce::allreduce(
    cudaStream_t stream, T* input, T* output, int num_elems) {
  
  int input_bytes = num_elems * sizeof(T);
  
  // ═══════════════════════════════════════════════
  // PHASE 1: Intra-node all-reduce
  // ═══════════════════════════════════════════════
  // Result: every local GPU has local_sum = Σ(all local GPUs' input)
  //
  // We use an intermediate buffer for the intra-node result.
  // For the gateway, this goes into the UCCL-EP send buffer
  // (RDMA-registered, so the CPU proxy can GPUDirect from it).
  // For non-gateway GPUs, any temp buffer works.
  
  T* phase1_output;
  if (is_gateway_) {
    phase1_output = reinterpret_cast<T*>(uccl_send_buf_);
  } else {
    // Use the existing buffer_ptrs staging area
    phase1_output = reinterpret_cast<T*>(broadcast_buf_);
  }
  
  intra_ar_->allreduce<T>(stream, input, phase1_output, num_elems);
  
  // If single node, we're done
  if (config_.num_nodes == 1) {
    if (phase1_output != output) {
      cudaMemcpyAsync(output, phase1_output, input_bytes,
                      cudaMemcpyDeviceToDevice, stream);
    }
    return;
  }
  
  // ═══════════════════════════════════════════════
  // PHASE 2: Inter-node all-reduce (gateway only)
  // ═══════════════════════════════════════════════
  if (is_gateway_) {
    // The gateway's Phase 1 result is already in uccl_send_buf_
    // (RDMA-registered, ready for GPUDirect)
    
    // Prepare recv buffer pointers for the kernel
    T* recv_buf_ptrs_d;  // Device array of pointers
    // (These are pre-allocated during init and stored on device)
    
    // Launch inter-node allreduce kernel
    // Use 1 block for simplicity (OK for small messages)
    int threads = min(1024, (num_elems / (16/sizeof(T)) + 31) & ~31);
    threads = max(threads, 32);  // At least one warp for FIFO ops
    
    inter_node_allreduce_kernel<T><<<1, threads, 0, stream>>>(
        fifos_,
        num_fifos_,
        reinterpret_cast<T*>(uccl_send_buf_),
        reinterpret_cast<T**>(recv_bufs_device_ptrs_),
        reinterpret_cast<T*>(broadcast_ptrs_[config_.gateway_local_rank]),
        num_elems,
        config_.node_id,
        config_.num_nodes,
        /* send_offset = */ 0,
        /* recv_base_offset = */ 0);
    
    // Gateway's result is now in broadcast_buf_ (IPC-shared)
    
    // Signal that Phase 2 is done
    // Write to hier_signal_ so other local GPUs can see it
    signal_phase2_done<<<1, 1, 0, stream>>>(hier_signal_);
    
  } else {
    // Non-gateway GPUs: wait for Phase 2 to complete
    // Spin on gateway's hier_signal_ (readable via IPC NVLink pointer)
    wait_for_phase2<<<1, 1, 0, stream>>>(
        hier_signal_ptrs_[config_.gateway_local_rank]);
  }
  
  // ═══════════════════════════════════════════════
  // PHASE 3: Intra-node broadcast
  // ═══════════════════════════════════════════════
  if (is_gateway_) {
    // Gateway: result is already in broadcast_buf_, just copy to output
    if (output != reinterpret_cast<T*>(
            broadcast_ptrs_[config_.gateway_local_rank])) {
      cudaMemcpyAsync(output,
                      broadcast_ptrs_[config_.gateway_local_rank],
                      input_bytes,
                      cudaMemcpyDeviceToDevice, stream);
    }
  } else {
    // Non-gateway: read from gateway's broadcast buffer via NVLink
    int threads = 256;
    int blocks = min(
        (num_elems / (16/sizeof(T)) + threads - 1) / threads, 108);
    
    intra_node_broadcast_kernel<T><<<blocks, threads, 0, stream>>>(
        output,
        reinterpret_cast<T*>(
            broadcast_ptrs_[config_.gateway_local_rank]),
        num_elems);
  }
  
  // Reset signal for next iteration
  reset_phase_signal<<<1, 1, 0, stream>>>(hier_signal_);
}

// ─── Helper kernels for phase synchronization ───

__global__ void signal_phase2_done(HierSignal* signal) {
  // Use volatile + threadfence_system to ensure visibility via NVLink
  __threadfence_system();
  volatile uint32_t* flag = &signal->phase2_done;
  *flag = 1;
  __threadfence_system();
}

__global__ void wait_for_phase2(HierSignal* gateway_signal) {
  volatile uint32_t* flag = &gateway_signal->phase2_done;
  while (*flag != 1) {
    // Spin — gateway is doing inter-node all-reduce
    // This wait is bounded by the network RTT + UCCL-EP overhead
  }
  __threadfence_system();
}

__global__ void reset_phase_signal(HierSignal* signal) {
  signal->phase2_done = 0;
  __threadfence_system();
}

}  // namespace vllm
```

### 3.5 C++ Binding Layer

```cpp
// hierarchical_allreduce_ops.cpp

#include "hierarchical_allreduce.cuh"
#include <torch/all.h>

using fptr_t = int64_t;

fptr_t init_hierarchical_ar(
    fptr_t intra_ar_ptr,
    // UCCL-EP connection info (exchanged via Python layer)
    const std::vector<std::vector<int64_t>>& remote_gateway_infos,
    // IPC pointers for broadcast buffer (within node)
    const std::vector<fptr_t>& broadcast_ptrs,
    // IPC pointers for hierarchical signals
    const std::vector<fptr_t>& hier_signal_ptrs,
    // Config
    int64_t local_rank,
    int64_t local_world_size,
    int64_t node_id,
    int64_t num_nodes,
    int64_t gateway_local_rank,
    int64_t max_size,
    int64_t num_proxy_threads) {
  
  auto intra_ar = reinterpret_cast<vllm::CustomAllreduce*>(intra_ar_ptr);
  
  // Deserialize connection info
  std::vector<uccl_ep::ConnectionInfo> connections;
  for (auto& info : remote_gateway_infos) {
    connections.push_back(uccl_ep::ConnectionInfo::deserialize(info));
  }
  
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
  
  return (fptr_t) new vllm::HierarchicalAllreduce(
      intra_ar, connections, bcast_ptrs, sig_ptrs, config);
}

void hierarchical_all_reduce(
    fptr_t _har, torch::Tensor& inp, torch::Tensor& out) {
  
  auto har = reinterpret_cast<vllm::HierarchicalAllreduce*>(_har);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  
  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      har->allreduce<float>(stream,
                            reinterpret_cast<float*>(inp.data_ptr()),
                            reinterpret_cast<float*>(out.data_ptr()),
                            out.numel());
      break;
    case at::ScalarType::Half:
      har->allreduce<half>(stream,
                           reinterpret_cast<half*>(inp.data_ptr()),
                           reinterpret_cast<half*>(out.data_ptr()),
                           out.numel());
      break;
    case at::ScalarType::BFloat16:
      har->allreduce<nv_bfloat16>(
          stream,
          reinterpret_cast<nv_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()),
          out.numel());
      break;
    default:
      throw std::runtime_error("Unsupported dtype");
  }
}

void dispose_hierarchical_ar(fptr_t _har) {
  delete reinterpret_cast<vllm::HierarchicalAllreduce*>(_har);
}
```

### 3.6 Python Layer

```python
# hierarchical_allreduce.py

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from contextlib import contextmanager
from typing import Optional

from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
    is_weak_contiguous,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class HierarchicalAllreduce:
    """
    Hierarchical all-reduce combining:
      - vLLM's custom intra-node all-reduce (NVLink)
      - UCCL-EP inter-node all-reduce (RDMA via CPU proxy)
    
    Optimized for SMALL messages where latency dominates.
    
    Architecture:
      Phase 1: Intra-node reduce   (vLLM custom AR, NVLink direct ptr)
      Phase 2: Inter-node reduce   (UCCL-EP FIFO → CPU proxy → RDMA)
      Phase 3: Intra-node broadcast (NVLink direct ptr read)
    """

    def __init__(
        self,
        # Process groups
        local_group: ProcessGroup,        # Intra-node (e.g., gloo backend)
        global_group: ProcessGroup,       # All ranks across all nodes
        gateway_group: Optional[ProcessGroup],  # Gateway ranks only
        
        # Device info
        device: torch.device,
        local_rank: int,
        global_rank: int,
        node_id: int,
        num_nodes: int,
        local_world_size: int,
        
        # Configuration  
        max_size: int = 8192 * 1024,      # 8 MB max message
        gateway_local_rank: int = 0,
        num_proxy_threads: int = 4,
    ):
        self.local_group = local_group
        self.global_group = global_group
        self.gateway_group = gateway_group
        self.device = device
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.local_world_size = local_world_size
        self.max_size = max_size
        self.gateway_local_rank = gateway_local_rank
        self.is_gateway = (local_rank == gateway_local_rank)
        self.disabled = False
        self._IS_CAPTURING = False
        
        if num_nodes <= 1:
            logger.info("Single node — hierarchical AR not needed")
            self.disabled = True
            return
        
        # ────────────────────────────────────────────
        # Step 1: Initialize intra-node custom AR
        # ────────────────────────────────────────────
        # Reuse vLLM's existing CustomAllreduce for Phase 1 & 3
        self.intra_ar = CustomAllreduce(
            group=local_group,
            device=device,
            max_size=max_size,
        )
        if self.intra_ar.disabled:
            logger.warning("Intra-node custom AR disabled; "
                           "hierarchical AR disabled too")
            self.disabled = True
            return
        
        # ────────────────────────────────────────────
        # Step 2: Create broadcast buffer (IPC shared within node)
        # ────────────────────────────────────────────
        # Gateway writes the global result here after Phase 2.
        # Other local GPUs read from gateway's buffer via NVLink.
        self.broadcast_ptrs = CustomAllreduce.create_shared_buffer(
            max_size, group=local_group
        )
        
        # ────────────────────────────────────────────
        # Step 3: Create hierarchical signal buffer (IPC shared)
        # ────────────────────────────────────────────
        self.hier_signal_ptrs = CustomAllreduce.create_shared_buffer(
            ops.hier_signal_size(),  # sizeof(HierSignal)
            group=local_group
        )
        
        # ────────────────────────────────────────────
        # Step 4: Exchange UCCL-EP connection info (gateways only)
        # ────────────────────────────────────────────
        if self.is_gateway:
            remote_gateway_infos = self._exchange_uccl_ep_connection_info()
        else:
            remote_gateway_infos = []
        
        # ────────────────────────────────────────────
        # Step 5: Initialize C++ HierarchicalAllreduce
        # ────────────────────────────────────────────
        self._ptr = ops.init_hierarchical_ar(
            self.intra_ar._ptr,
            remote_gateway_infos,
            self.broadcast_ptrs,
            self.hier_signal_ptrs,
            local_rank,
            local_world_size,
            node_id,
            num_nodes,
            gateway_local_rank,
            max_size,
            num_proxy_threads,
        )
        
        logger.info(
            "HierarchicalAllreduce initialized: node %d/%d, "
            "local_rank %d/%d, gateway=%s",
            node_id, num_nodes, local_rank, local_world_size,
            self.is_gateway
        )
    
    def _exchange_uccl_ep_connection_info(self):
        """
        Gateway GPUs exchange UCCL-EP connection info 
        (RDMA QP details, symmetric memory base addresses)
        via the gateway process group.
        """
        assert self.is_gateway
        assert self.gateway_group is not None
        
        # Get local UCCL-EP connection info
        # This includes: NIC info, QP numbers, symmetric mem base, etc.
        local_info = ops.get_uccl_ep_connection_info(self.device.index)
        
        # Exchange with all other gateways
        gateway_world_size = dist.get_world_size(group=self.gateway_group)
        all_infos = [None] * gateway_world_size
        dist.all_gather_object(all_infos, local_info, group=self.gateway_group)
        
        # Filter out our own info
        gateway_rank = dist.get_rank(group=self.gateway_group)
        remote_infos = [
            info for i, info in enumerate(all_infos) if i != gateway_rank
        ]
        
        return remote_infos
    
    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        """Check if hierarchical allreduce should be used."""
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if inp_size > self.max_size:
            return False
        return True
    
    def all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None):
        """Perform hierarchical all-reduce."""
        if out is None:
            out = torch.empty_like(inp)
        ops.hierarchical_all_reduce(self._ptr, inp, out)
        return out
    
    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """Main API matching CustomAllreduce interface."""
        if self.disabled or not self.should_custom_ar(input):
            return None
        
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input)
            else:
                return torch.empty_like(input)
        else:
            return self.all_reduce(input)
    
    @contextmanager
    def capture(self):
        """CUDA graph capture context."""
        try:
            self._IS_CAPTURING = True
            if not self.disabled:
                self.intra_ar._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.intra_ar._IS_CAPTURING = False
                self.intra_ar.register_graph_buffers()
    
    def close(self):
        if not self.disabled and self._ptr:
            ops.dispose_hierarchical_ar(self._ptr)
            self._ptr = 0
            self.intra_ar.close()
            CustomAllreduce.free_shared_buffer(
                self.broadcast_ptrs, rank=self.local_rank)
            CustomAllreduce.free_shared_buffer(
                self.hier_signal_ptrs, rank=self.local_rank)
    
    def __del__(self):
        self.close()
```

---

## 4. Integration into vLLM's Parallel State

The integration point is in `vllm/distributed/parallel_state.py`:

```python
# In GroupCoordinator.__init__() or equivalent

class GroupCoordinator:
    def __init__(self, ...):
        # ... existing initialization ...
        
        # Determine topology
        self.num_nodes = world_size // local_world_size
        self.node_id = rank // local_world_size
        self.local_rank = rank % local_world_size
        
        # Create process groups
        if self.num_nodes > 1:
            # Local group: ranks on same node
            local_ranks = list(range(
                self.node_id * local_world_size,
                (self.node_id + 1) * local_world_size
            ))
            self.local_group = dist.new_group(
                ranks=local_ranks, backend="gloo"
            )
            
            # Gateway group: rank 0 from each node
            gateway_ranks = [
                n * local_world_size for n in range(self.num_nodes)
            ]
            self.gateway_group = (
                dist.new_group(ranks=gateway_ranks, backend="gloo")
                if self.local_rank == 0 else None
            )
            
            # Initialize hierarchical allreduce
            self.hier_comm = HierarchicalAllreduce(
                local_group=self.local_group,
                global_group=self.group,
                gateway_group=self.gateway_group,
                device=self.device,
                local_rank=self.local_rank,
                global_rank=self.rank,
                node_id=self.node_id,
                num_nodes=self.num_nodes,
                local_world_size=local_world_size,
            )
        else:
            self.hier_comm = None
        
        # Existing single-node custom AR
        self.ca_comm = CustomAllreduce(
            group=self.group, device=self.device
        )
    
    def all_reduce(self, input_):
        # Priority 1: Hierarchical AR for multi-node small messages
        if self.hier_comm is not None:
            out = self.hier_comm.custom_all_reduce(input_)
            if out is not None:
                return out
        
        # Priority 2: Single-node custom AR
        if self.ca_comm is not None:
            out = self.ca_comm.custom_all_reduce(input_)
            if out is not None:
                return out
        
        # Fallback: NCCL
        dist.all_reduce(input_, group=self.group)
        return input_
```

---

## 5. UCCL-EP Integration Specifics

### 5.1 CPU Proxy Lifecycle

UCCL-EP's CPU proxy is a critical component. In vLLM's process model, each GPU worker is a separate process. The CPU proxy threads live **within the gateway GPU's worker process**:

```
Node 0:
┌──────────────────────────────────────────────────────┐
│ Worker Process 0 (GPU 0 = Gateway)                   │
│ ┌────────────┐  ┌────────────────────────────┐       │
│ │ Main Thread │  │ UCCL-EP CPU Proxy Threads  │       │
│ │ (Python     │  │ Thread 0: FIFO poll + RDMA │       │
│ │  vLLM       │  │ Thread 1: FIFO poll + RDMA │       │
│ │  model      │  │ Thread 2: FIFO poll + RDMA │       │
│ │  runner)    │  │ Thread 3: FIFO poll + RDMA │       │
│ └────────────┘  └────────────────────────────┘       │
│      GPU 0: model forward + allreduce kernels        │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ Worker Process 1 (GPU 1, non-gateway)                │
│ ┌────────────┐                                        │
│ │ Main Thread │  (No UCCL-EP proxy needed)            │
│ │ (Python)    │                                       │
│ └────────────┘                                        │
│      GPU 1: model forward + allreduce kernels        │
└──────────────────────────────────────────────────────┘

   ... (Worker 2, 3 — same as Worker 1)
```

The proxy threads are spawned during `HierarchicalAllreduce.__init__()` on the gateway GPU process, and destroyed in `close()`. They run continuously, polling the FIFO channels.

**CPU overhead**: UCCL-EP uses 4 threads per GPU, but only the gateway GPU (1 per node) needs them. On a server with 128-192 CPU cores and 8 GPUs, 4 proxy threads represent ~2-3% CPU utilization. UCCL-EP's paper reports that even with 4 threads per GPU (32 total for 8 GPUs), CPU utilization increases from 8% to only 22%.

### 5.2 Memory Registration

UCCL-EP uses symmetric memory registered for GPUDirect RDMA. This registration needs to happen during initialization:

```
Data flow during Phase 2 (inter-node):

GPU₀ memory (send_buf)                     GPU₄ memory (recv_buf)
    │                                           ▲
    │ (GPUDirect: NIC reads                     │ (GPUDirect: NIC writes
    │  directly from GPU                        │  directly to GPU
    │  via PCIe switch)                         │  via PCIe switch)
    ▼                                           │
  NIC_A ──────── network ──────────────────── NIC_B
        RDMA Write (posted by CPU proxy_A)

The data NEVER touches CPU memory.
Only the 16-byte TransferCmd crosses GPU→CPU (via FIFO).
```

### 5.3 How UCCL-EP FIFO Connects to the Inter-Node Kernel

The GPU kernel's `Push()` call writes a 128-bit descriptor to host memory. Here's the exact data flow for one inter-node write:

```
TIME   GPU (gateway)                    CPU Proxy                    NIC
─────  ─────────────────────────────   ──────────────────────────   ──────────
T=0    inter_node_allreduce_kernel:
       Push(Write cmd) to FIFO[0]
       → 128-bit store to host mem
       → threadfence_system()
       
T=1    Continue to next Push           Thread 0: Poll(FIFO[0])
       (or start barrier)              → Reads cmd from host mem
                                       → Sees: Write, dst=Node1,
                                         src_off=0, len=32KB

T=2                                    Thread 0: Build RDMA WR       
                                       → ibv_post_send(
                                           RDMA_WRITE_WITH_IMM,
                                           src=gpu_base + src_off,
                                           dst=remote_base + dst_off,
                                           len=32KB)
                                       Pop(FIFO[0])                  
                                                                     
T=3                                                                  NIC reads
                                                                     32KB from
                                                                     GPU mem via
                                                                     PCIe switch
                                                                     
T=4                                                                  Sends over
                                                                     network
                                                                     
T=5    (GPU is now doing the                                         Remote NIC
        barrier spin-wait)                                           writes to
                                                                     remote GPU
```

### 5.4 Why This Matches vLLM's Small-Message Philosophy

vLLM's custom AR and this hierarchical extension both prioritize latency over bandwidth for small messages:

```
Message size: 32 KB (typical small allreduce in LLM inference)
Nodes: 4, GPUs per node: 8, Total: 32 GPUs

LATENCY BREAKDOWN (hierarchical with UCCL-EP):
─────────────────────────────────────────────────
Phase 1: Intra-node custom AR
  • Signal barrier:     ~2 μs
  • NVLink read-reduce: ~5 μs (32KB × 8 GPUs via NVLink)
  Subtotal:             ~7 μs

Phase 2: Inter-node (UCCL-EP, gateway only)
  • FIFO Push (×3):     ~1 μs (3 remote nodes × 16B descriptor)
  • CPU proxy latency:  ~2 μs (poll + build WR + post)
  • RDMA write:         ~5 μs one-way (32KB at 400Gbps)
  • Barrier RTT:        ~10 μs (hierarchical barrier via RDMA imm)
  • Local reduce:       ~1 μs (sum 3 × 32KB vectors)
  Subtotal:             ~19 μs

Phase 3: Intra-node broadcast
  • Signal wait:        ~2 μs
  • NVLink read:        ~3 μs (32KB from gateway's buffer)
  Subtotal:             ~5 μs

TOTAL:                  ~31 μs

COMPARISON:
  NCCL ring allreduce (32 GPUs, 4 nodes): ~100-200 μs
  NCCL tree allreduce:                     ~50-80 μs
  Hierarchical with UCCL-EP:               ~31 μs
  
  Speedup: 2-6× for small messages
```

---

## 6. Complete Data Flow Diagram

```
═══════════════════════════════════════════════════════════════════════════
                    COMPLETE DATA FLOW
        Hierarchical AllReduce with UCCL-EP + vLLM Custom AR
═══════════════════════════════════════════════════════════════════════════

Node 0                                     Node 1
GPU₀(GW) GPU₁ GPU₂ GPU₃                   GPU₄(GW) GPU₅ GPU₆ GPU₇
 x₀       x₁   x₂   x₃                    x₄       x₅   x₆   x₇
 │         │    │    │                      │         │    │    │
 ▼─────────▼────▼────▼                      ▼─────────▼────▼────▼
┌──────────────────────┐                   ┌──────────────────────┐
│ PHASE 1: vLLM Custom │                   │ PHASE 1: vLLM Custom │
│ AllReduce (NVLink)   │                   │ AllReduce (NVLink)   │
│                      │                   │                      │
│ All GPUs write to    │                   │ All GPUs write to    │
│ IPC shared buffers   │                   │ IPC shared buffers   │
│ → Barrier (signals)  │                   │ → Barrier (signals)  │
│ → Each GPU reads all │                   │ → Each GPU reads all │
│   others via NVLink  │                   │   others via NVLink  │
│ → Local reduce       │                   │ → Local reduce       │
└──────┬───────────────┘                   └──────┬───────────────┘
       │                                          │
 S₀=Σ(x₀..x₃) on all local GPUs          S₁=Σ(x₄..x₇) on all local GPUs
       │                                          │
       │ (GW GPU₀ only)                           │ (GW GPU₄ only)
       ▼                                          ▼
┌──────────────────────┐                   ┌──────────────────────┐
│ PHASE 2: UCCL-EP     │                   │ PHASE 2: UCCL-EP     │
│ Inter-Node AllReduce │                   │ Inter-Node AllReduce │
│                      │                   │                      │
│ 1. S₀ already in     │                   │ 1. S₁ already in     │
│    RDMA-reg send_buf │                   │    RDMA-reg send_buf │
│                      │                   │                      │
│ 2. GPU Push(Write):  │ ──FIFO──→ CPU₀   │ 2. GPU Push(Write):  │
│    "Send S₀ to Node1"│          proxy    │    "Send S₁ to Node0"│
│                      │            │      │                      │
│                      │     ibv_post_send │                      │
│                      │            │      │                      │
│    ┌─────────────────┼───GPUDirect┼──────┼─────────────────┐   │
│    │                 │    RDMA    │      │                 │   │
│    │  send_buf(GPU₀) │ ──────────┼──▶   │ recv_buf(GPU₄)  │   │
│    │  (NIC reads     │    Write   │      │ (NIC writes     │   │
│    │   from GPU mem) │           │      │  to GPU mem)    │   │
│    └─────────────────┼───────────┼──────┼─────────────────┘   │
│                      │           │      │                      │
│                      │ ◀─────────┼──────┤ CPU₁ proxy:         │
│    recv_buf(GPU₀)  ◀─┤  GPUDirect│      │ ibv_post_send       │
│    gets S₁           │    RDMA   │      │ → Send S₁ to Node0  │
│                      │           ▼      │                      │
│ 3. GPU Push(Barrier) │          CPU₀    │ 3. GPU Push(Barrier) │
│    → spin-wait       │       establishes│    → spin-wait       │
│                      │    hierarchical  │                      │
│                      │      barrier     │                      │
│ 4. GPU local reduce: │                  │ 4. GPU local reduce: │
│    G = S₀ + S₁       │                  │    G = S₁ + S₀       │
│                      │                  │                      │
│ 5. Write G to IPC    │                  │ 5. Write G to IPC    │
│    broadcast_buf     │                  │    broadcast_buf     │
│ 6. Signal phase2_done│                  │ 6. Signal phase2_done│
└──────┬───────────────┘                   └──────┬───────────────┘
       │                                          │
       ▼                                          ▼
┌──────────────────────┐                   ┌──────────────────────┐
│ PHASE 3: Intra-Node  │                   │ PHASE 3: Intra-Node  │
│ Broadcast (NVLink)   │                   │ Broadcast (NVLink)   │
│                      │                   │                      │
│ Non-GW GPUs:         │                   │ Non-GW GPUs:         │
│ wait_for_phase2()    │                   │ wait_for_phase2()    │
│ → spin on GW's       │                   │ → spin on GW's       │
│   hier_signal via    │                   │   hier_signal via    │
│   NVLink IPC pointer │                   │   NVLink IPC pointer │
│                      │                   │                      │
│ Then: each GPU reads │                   │ Then: each GPU reads │
│ G from GW's IPC      │                   │ G from GW's IPC      │
│ broadcast_buf via    │                   │ broadcast_buf via    │
│ NVLink (vectorized   │                   │ NVLink (vectorized   │
│ 16-byte loads)       │                   │ 16-byte loads)       │
│                      │                   │                      │
│ GW: copies G to out  │                   │ GW: copies G to out  │
└──────────────────────┘                   └──────────────────────┘

RESULT: All 8 GPUs have G = x₀+x₁+x₂+x₃+x₄+x₅+x₆+x₇  ✓
```

---

## 7. Key Design Decisions and Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| **Phase 1 algorithm** | Full all-reduce (not reduce-to-gateway) | Reuses existing vLLM code with zero modifications. Bandwidth waste negligible for small messages. Every GPU already has the local sum, simplifying Phase 3. |
| **Phase 2 algorithm** | Flat all-to-all write + barrier + local reduce | Minimal latency for 2-8 nodes. One round of RDMA writes + one barrier. Matches vLLM's bandwidth-for-latency tradeoff philosophy. |
| **Phase 2 transport** | UCCL-EP FIFO + CPU proxy | Portable (any NIC via libibverbs). GPU-initiated (no CPU in the hot path for initiation). CPU handles RDMA posting and delivery semantics. |
| **Phase 3 algorithm** | NVLink direct read from gateway's IPC buffer | All non-gateway GPUs read in parallel. O(1) latency. Reuses existing IPC infrastructure. |
| **Gateway selection** | Local rank 0 | Simple. Could be configurable for load balancing (e.g., rotate per layer). |
| **CPU proxy location** | Within gateway worker process | No extra daemon. Spawned/destroyed with the vLLM worker. Minimal overhead (4 threads). |
| **FIFO channel count** | One per remote node | For all-reduce with few nodes (2-8), contention is minimal. Each FIFO handles one Write + one Barrier per allreduce call. |
| **Memory registration** | Symmetric memory via UCCL-EP | Offsets in 128-bit descriptor instead of full addresses. Base addresses exchanged once during init. |
| **Barrier implementation** | UCCL-EP hierarchical Barrier TransferCmd | CPU proxy handles intra-node (shared memory) + inter-node (RDMA imm) barrier. GPU just Push + Check-completion. |

---

## 8. CUDA Graph Considerations

UCCL-EP's FIFO operations are graph-capturable because they are standard GPU memory operations (stores to host memory, loads from host memory):

```
CAPTURABLE:
  • Phase 1: vLLM custom AR kernel (already graph-captured in vLLM)
  • Phase 2 GPU-side: 
      Push() = st.global.v4.u32 to fixed host addr  ← capturable
      CheckCompletion() = ld.global from fixed host addr  ← capturable
      Local reduce = standard GPU compute  ← capturable
  • Phase 3: NVLink broadcast kernel  ← capturable

NOT IN GRAPH (runs independently):
  • CPU proxy threads (polling FIFOs, posting RDMA)
  • RDMA NIC operations
  
This works because the GPU graph replays the same Push/Check-completion
pattern to the same FIFO addresses. The CPU proxy processes commands
regardless of whether they come from a live kernel or a replayed graph.
```

The caveat: the FIFO head/tail indices advance with each graph replay. Since UCCL-EP uses a ring buffer with `kMaxInflight` slots and wrapping arithmetic, this works correctly as long as the CPU proxy keeps up with consumption (which it will, since the allreduce is a blocking operation from the GPU's perspective due to the barrier).

---

## 9. Build and Packaging Considerations

Since the guidance emphasizes **wrapping it into an easy-to-use python wheel**:

```
uccl-vllm-allreduce/
├── CMakeLists.txt              # Build system
├── csrc/
│   ├── hierarchical_allreduce.cuh    # CUDA kernel + orchestrator
│   ├── hierarchical_allreduce.cpp    # Python bindings
│   ├── custom_all_reduce.cuh         # Existing vLLM custom AR (vendored)
│   └── custom_all_reduce.cpp         # Existing vLLM custom AR (vendored)
├── third_party/
│   └── uccl-ep/                      # UCCL-EP library (submodule)
│       ├── include/
│       │   ├── uccl_ep/fifo_channel.h
│       │   ├── uccl_ep/proxy.h
│       │   └── uccl_ep/symmetric_mem.h
│       └── lib/
│           └── libuccl_ep.so
├── python/
│   ├── uccl_vllm/
│   │   ├── __init__.py
│   │   ├── hierarchical_allreduce.py
│   │   └── _ops.py                   # torch.ops registration
│   └── setup.py                      # pip install .
└── README.md
```

The wheel would be installed alongside vLLM and activated via an environment variable or config:

```python
# vLLM config
VLLM_USE_HIERARCHICAL_AR=1
VLLM_UCCL_EP_PROXY_THREADS=4
```