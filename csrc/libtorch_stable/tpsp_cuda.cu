// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "torch_utils.h"
#include "cub_helpers.h"

#include <cuda_bf16.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

using bf16 = __nv_bfloat16;

// Each rank exposes separate inbox and gather slots to every peer.
constexpr int kTpspDataRegions = 2;
// P2P flags are cleared by the receiver after each phase.
constexpr int kTpspInboxReady = 0;
constexpr int kTpspGatherReady = 1;
constexpr int kTpspGatherConsumed = 2;
constexpr int kTpspFlagPhases = kTpspGatherConsumed + 1;

struct alignas(16) TpspBf16Vec {
  bf16 data[8];
};

__global__ void pack_tpsp_chunk(const bf16* input, bf16* packed, int tokens,
                                int width, int rows, int chunk_rows, int offset,
                                int tp_size) {
  // Group the same row range from each rank, padding the final shard with zero.
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t count = int64_t(tp_size) * chunk_rows * width;
  if (index >= count) {
    return;
  }
  int row = index / width;
  int rank = row / chunk_rows;
  int source_row = rank * rows + offset + row % chunk_rows;
  packed[index] = source_row < tokens
                      ? input[int64_t(source_row) * width + index % width]
                      : bf16(0.0f);
}

__global__ void unpack_tpsp_chunk(const bf16* packed, bf16* output, int tokens,
                                  int width, int rows, int chunk_rows,
                                  int offset) {
  // Undo the chunk-major layout and discard padded rows.
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t count = int64_t((tokens + rows - 1) / rows) * chunk_rows * width;
  if (index >= count) {
    return;
  }
  int packed_row = index / width;
  int row = packed_row / chunk_rows * rows + offset + packed_row % chunk_rows;
  if (row < tokens) {
    output[int64_t(row) * width + index % width] = packed[index];
  }
}

__global__ void wait_and_clear_tpsp_signals(unsigned int* signals, int rank,
                                            int tp_size) {
  int peer = blockIdx.x * blockDim.x + threadIdx.x;
  if (peer >= tp_size || peer == rank) {
    return;
  }
  auto start = clock64();
  while (atomicCAS(signals + peer, 1, 0) != 1) {
    if (clock64() - start > 30000000000ULL) {
      printf("TPSP P2P signal timed out\n");
      asm volatile("trap;");
    }
  }
}

template <bool LayerNorm>
__global__ void tpsp_add_norm(const bf16* reduced, const bf16* remote,
                              int64_t remote_stride, int rank, int tp_size,
                              const bf16* residual, const bf16* projection_bias,
                              const bf16* weight, const bf16* norm_bias,
                              bf16* new_residual, bf16* normalized,
                              int hidden_size, float eps) {
  int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    int index = row * hidden_size + col;
    bf16 projected = reduced[index];
    if (remote) {
      float sum = float(projected);
      for (int source = 0; source < tp_size; ++source) {
        if (source != rank) {
          sum += float(remote[int64_t(source) * remote_stride + index]);
        }
      }
      projected = bf16(sum);
    }
    if (projection_bias) {
      projected = bf16(float(projected) + float(projection_bias[col]));
    }
    bf16 value = bf16(float(projected) + float(residual[index]));
    new_residual[index] = value;
    float x = float(value);
    sum += LayerNorm ? x : x * x;
  }
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  sum = BlockReduce(reduce_store).Reduce(sum, CubAddOp{}, blockDim.x);
  __shared__ float mean;
  if constexpr (LayerNorm) {
    if (threadIdx.x == 0) {
      mean = sum / hidden_size;
    }
    __syncthreads();
    sum = 0.0f;
    for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
      float centered = float(new_residual[row * hidden_size + col]) - mean;
      sum += centered * centered;
    }
    sum = BlockReduce(reduce_store).Reduce(sum, CubAddOp{}, blockDim.x);
  }
  __shared__ float inverse_std;
  if (threadIdx.x == 0) {
    inverse_std = rsqrtf(sum / hidden_size + eps);
  }
  __syncthreads();
  for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    int index = row * hidden_size + col;
    float value = float(new_residual[index]);
    if constexpr (LayerNorm) {
      value -= mean;
    }
    value *= inverse_std;
    value *= float(weight[col]);
    if constexpr (LayerNorm) {
      if (norm_bias) {
        value += float(norm_bias[col]);
      }
    }
    normalized[index] = bf16(value);
  }
}

template <bool LayerNorm>
__global__ void tpsp_add_norm_vector(const bf16* reduced, const bf16* remote,
                                     int64_t remote_stride, int rank,
                                     int tp_size, const bf16* residual,
                                     const bf16* projection_bias,
                                     const bf16* weight, const bf16* norm_bias,
                                     bf16* new_residual, bf16* normalized,
                                     int hidden_size, float eps) {
  using Vec = TpspBf16Vec;
  const int row = blockIdx.x;
  const int vec_hidden = hidden_size / 8;
  const auto* reduced_v =
      reinterpret_cast<const Vec*>(reduced + int64_t(row) * hidden_size);
  const auto* residual_v =
      reinterpret_cast<const Vec*>(residual + int64_t(row) * hidden_size);
  const auto* projection_bias_v = reinterpret_cast<const Vec*>(projection_bias);
  const auto* weight_v = reinterpret_cast<const Vec*>(weight);
  const auto* norm_bias_v = reinterpret_cast<const Vec*>(norm_bias);
  auto* new_residual_v =
      reinterpret_cast<Vec*>(new_residual + int64_t(row) * hidden_size);
  auto* normalized_v =
      reinterpret_cast<Vec*>(normalized + int64_t(row) * hidden_size);

  float variance = 0.0f;
  for (int idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
    Vec value = reduced_v[idx];
    Vec other = residual_v[idx];
    Vec bias;
    if (projection_bias) {
      bias = projection_bias_v[idx];
    }
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      __nv_bfloat162 pair{value.data[j], value.data[j + 1]};
      if (remote) {
        float2 sum = __bfloat1622float2(pair);
        for (int source = 0; source < tp_size; ++source) {
          if (source != rank) {
            auto* incoming = reinterpret_cast<const Vec*>(
                remote + int64_t(source) * remote_stride +
                int64_t(row) * hidden_size);
            float2 x = __bfloat1622float2(__nv_bfloat162{
                incoming[idx].data[j], incoming[idx].data[j + 1]});
            sum.x += x.x;
            sum.y += x.y;
          }
        }
        pair = __floats2bfloat162_rn(sum.x, sum.y);
      }
      if (projection_bias) {
        pair += __nv_bfloat162{bias.data[j], bias.data[j + 1]};
      }
      pair += __nv_bfloat162{other.data[j], other.data[j + 1]};
      value.data[j] = pair.x;
      value.data[j + 1] = pair.y;
    }
    new_residual_v[idx] = value;
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      float2 pair =
          __bfloat1622float2(__nv_bfloat162{value.data[j], value.data[j + 1]});
      if constexpr (LayerNorm) {
        variance += pair.x + pair.y;
      } else {
        variance += pair.x * pair.x + pair.y * pair.y;
      }
    }
  }

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  variance = BlockReduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);
  __shared__ float mean;
  if constexpr (LayerNorm) {
    if (threadIdx.x == 0) {
      mean = variance / hidden_size;
    }
    __syncthreads();
    variance = 0.0f;
    for (int idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
      Vec value = new_residual_v[idx];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float centered = __bfloat162float(value.data[j]) - mean;
        variance += centered * centered;
      }
    }
    variance =
        BlockReduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);
  }
  __shared__ float inverse_std;
  if (threadIdx.x == 0) {
    inverse_std = rsqrtf(variance / hidden_size + eps);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
    Vec value = new_residual_v[idx];
    Vec weights = weight_v[idx];
    Vec bias;
    if constexpr (LayerNorm) {
      if (norm_bias) {
        bias = norm_bias_v[idx];
      }
    }
    Vec output;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float x = __bfloat162float(value.data[j]);
      if constexpr (LayerNorm) {
        x -= mean;
      }
      float w = __bfloat162float(weights.data[j]);
      float y = x * inverse_std * w;
      if constexpr (LayerNorm) {
        if (norm_bias) {
          y += __bfloat162float(bias.data[j]);
        }
      }
      output.data[j] = __float2bfloat16(y);
    }
    normalized_v[idx] = output;
  }
}

struct CudaEvent {
  cudaEvent_t handle{};

  CudaEvent() {
    STD_CUDA_CHECK(cudaEventCreateWithFlags(&handle, cudaEventDisableTiming));
  }
  ~CudaEvent() { cudaEventDestroy(handle); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
};

struct PipelineState {
  // Reuse a scratch slot only after its communication has finished.
  std::array<CudaEvent, 2> gemm_ready;
  std::array<CudaEvent, 2> comm_done;
  cudaStream_t comm_stream{};

  PipelineState() {
    int least_priority;
    int greatest_priority;
    STD_CUDA_CHECK(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    STD_CUDA_CHECK(cudaStreamCreateWithPriority(
        &comm_stream, cudaStreamNonBlocking, greatest_priority));
  }
  ~PipelineState() { cudaStreamDestroy(comm_stream); }
};

thread_local std::unordered_map<int, std::unique_ptr<PipelineState>>
    pipeline_states;

struct ChunkBuffers {
  torch::stable::Tensor packed;
  torch::stable::Tensor partial;
  torch::stable::Tensor local;
  torch::stable::Tensor gathered;
};

struct TpspP2pContext {
  int device_index;
  int rank;
  int tp_size;
  int64_t max_chunk_rows;
  int hidden_size;
  int64_t slot_bytes;
  int64_t flag_offset;
  ncclComm_t comm;
  void* workspace = nullptr;
  void* signal_one = nullptr;
  std::vector<void*> peers;

  void close_peer_mappings() {
    for (int source = 0; source < tp_size; ++source) {
      if (source != rank && peers[source]) {
        STD_CUDA_CHECK(cudaIpcCloseMemHandle(peers[source]));
        peers[source] = nullptr;
      }
    }
  }

  void free_local() {
    if (signal_one) {
      STD_CUDA_CHECK(cudaFree(signal_one));
      signal_one = nullptr;
    }
    if (workspace) {
      STD_CUDA_CHECK(cudaFree(workspace));
      workspace = nullptr;
    }
  }

  ~TpspP2pContext() {
    for (int source = 0; source < tp_size; ++source) {
      if (source != rank && peers[source]) {
        cudaIpcCloseMemHandle(peers[source]);
      }
    }
    if (signal_one) {
      cudaFree(signal_one);
    }
    if (workspace) {
      cudaFree(workspace);
    }
  }
};

std::mutex tpsp_context_mutex;
std::unordered_map<int64_t, std::shared_ptr<TpspP2pContext>> tpsp_contexts;
int64_t next_tpsp_handle = 1;

bool all_tpsp_ranks_ready(bool ready, ncclComm_t comm, cudaStream_t stream) {
  int local = ready ? 1 : 0;
  int global = 0;
  int* device_status = nullptr;
  STD_CUDA_CHECK(cudaMalloc(&device_status, 2 * sizeof(int)));
  auto status =
      std::unique_ptr<int, decltype(&cudaFree)>(device_status, cudaFree);
  STD_CUDA_CHECK(cudaMemcpyAsync(device_status, &local, sizeof(int),
                                 cudaMemcpyHostToDevice, stream));
  STD_TORCH_CHECK(ncclAllReduce(device_status, device_status + 1, 1, ncclInt32,
                                ncclMin, comm, stream) == ncclSuccess,
                  "TPSP P2P readiness exchange failed");
  STD_CUDA_CHECK(cudaMemcpyAsync(&global, device_status + 1, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
  STD_CUDA_CHECK(cudaStreamSynchronize(stream));
  return global != 0;
}

torch::stable::Tensor make_bf16(const torch::stable::Tensor& a, int64_t rows,
                                int64_t cols) {
  return torch::stable::empty({rows, cols}, a.scalar_type(), std::nullopt,
                              a.device());
}

}  // namespace

// Initialization and destruction must be called by all ranks in the same order.
int64_t init_tpsp_p2p(int64_t device_index, int64_t comm_address,
                      int64_t tp_size, int64_t rank, int64_t max_chunk_rows,
                      int64_t hidden_size) {
  STD_TORCH_CHECK(
      comm_address != 0 && tp_size >= 2 && tp_size <= INT32_MAX && rank >= 0 &&
          rank < tp_size && max_chunk_rows > 0 && max_chunk_rows <= INT32_MAX &&
          hidden_size > 0 && hidden_size <= INT32_MAX &&
          max_chunk_rows <= (std::numeric_limits<int64_t>::max() -
                             kTpspFlagPhases * tp_size * int64_t(sizeof(int))) /
                                tp_size / hidden_size /
                                (kTpspDataRegions * int64_t(sizeof(bf16))),
      "Invalid TPSP P2P configuration");
  const torch::stable::accelerator::DeviceGuard guard(device_index);
  auto stream = get_current_cuda_stream();
  auto comm = reinterpret_cast<ncclComm_t>(comm_address);
  auto context = std::make_shared<TpspP2pContext>();
  context->device_index = device_index;
  context->rank = rank;
  context->tp_size = tp_size;
  context->max_chunk_rows = max_chunk_rows;
  context->hidden_size = hidden_size;
  context->comm = comm;
  context->slot_bytes = max_chunk_rows * hidden_size * sizeof(bf16);
  context->flag_offset = kTpspDataRegions * tp_size * context->slot_bytes;
  context->peers.resize(tp_size, nullptr);
  size_t bytes =
      context->flag_offset + kTpspFlagPhases * tp_size * sizeof(unsigned int);

  cudaError_t allocated = cudaMalloc(&context->workspace, bytes);
  if (allocated == cudaSuccess) {
    allocated = cudaMalloc(&context->signal_one, sizeof(unsigned int));
  }
  cudaIpcMemHandle_t own_handle{};
  bool exported =
      allocated == cudaSuccess &&
      cudaIpcGetMemHandle(&own_handle, context->workspace) == cudaSuccess;
  if (!all_tpsp_ranks_ready(exported, comm, stream)) {
    context->free_local();
    return 0;
  }

  unsigned char* send = nullptr;
  unsigned char* recv = nullptr;
  STD_CUDA_CHECK(cudaMalloc(&send, sizeof(own_handle)));
  auto send_buffer =
      std::unique_ptr<unsigned char, decltype(&cudaFree)>(send, cudaFree);
  STD_CUDA_CHECK(cudaMalloc(&recv, tp_size * sizeof(own_handle)));
  auto recv_buffer =
      std::unique_ptr<unsigned char, decltype(&cudaFree)>(recv, cudaFree);
  STD_CUDA_CHECK(cudaMemcpyAsync(send, &own_handle, sizeof(own_handle),
                                 cudaMemcpyHostToDevice, stream));
  // Bootstrap IPC mappings over the existing NCCL communicator.
  STD_TORCH_CHECK(ncclAllGather(send, recv, sizeof(own_handle), ncclUint8, comm,
                                stream) == ncclSuccess,
                  "TPSP P2P handle exchange failed");
  std::vector<cudaIpcMemHandle_t> handles(tp_size);
  STD_CUDA_CHECK(cudaMemcpyAsync(handles.data(), recv,
                                 tp_size * sizeof(own_handle),
                                 cudaMemcpyDeviceToHost, stream));
  STD_CUDA_CHECK(cudaStreamSynchronize(stream));
  context->peers[rank] = context->workspace;
  bool mapped = true;
  for (int source = 0; source < tp_size; ++source) {
    if (source != rank &&
        cudaIpcOpenMemHandle(&context->peers[source], handles[source],
                             cudaIpcMemLazyEnablePeerAccess) != cudaSuccess) {
      context->peers[source] = nullptr;
      mapped = false;
      break;
    }
  }
  // An inaccessible peer (including one on another host) falls back on
  // every rank, not only on the rank where opening the handle failed.
  if (!all_tpsp_ranks_ready(mapped, comm, stream)) {
    context->close_peer_mappings();
    all_tpsp_ranks_ready(true, comm, stream);
    context->free_local();
    return 0;
  }
  STD_CUDA_CHECK(cudaMemsetAsync(context->workspace, 0, bytes, stream));
  unsigned int one = 1;
  STD_CUDA_CHECK(cudaMemcpyAsync(context->signal_one, &one, sizeof(one),
                                 cudaMemcpyHostToDevice, stream));
  all_tpsp_ranks_ready(true, comm, stream);
  std::lock_guard<std::mutex> lock(tpsp_context_mutex);
  int64_t handle = next_tpsp_handle++;
  tpsp_contexts.emplace(handle, std::move(context));
  return handle;
}

void destroy_tpsp_p2p(int64_t handle) {
  std::shared_ptr<TpspP2pContext> context;
  {
    std::lock_guard<std::mutex> lock(tpsp_context_mutex);
    auto it = tpsp_contexts.find(handle);
    STD_TORCH_CHECK(it != tpsp_contexts.end(), "Invalid TPSP P2P context");
    context = it->second;
  }
  const torch::stable::accelerator::DeviceGuard guard(context->device_index);
  STD_CUDA_CHECK(cudaDeviceSynchronize());
  // Peers must finish their writes before any rank frees its workspace.
  all_tpsp_ranks_ready(true, context->comm, get_current_cuda_stream());
  context->close_peer_mappings();
  all_tpsp_ranks_ready(true, context->comm, get_current_cuda_stream());
  context->free_local();
  std::lock_guard<std::mutex> lock(tpsp_context_mutex);
  tpsp_contexts.erase(handle);
}

// Project all tokens, reduce to local shards, normalize, then gather the
// normalized result. Return the local residual and normalized shards as well.
std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
tpsp_fused_matmul_reduce_scatter_norm_all_gather(
    const torch::stable::Tensor& a, const torch::stable::Tensor& b,
    const torch::stable::Tensor& weight, const torch::stable::Tensor& residual,
    const std::optional<torch::stable::Tensor>& projection_bias,
    const std::optional<torch::stable::Tensor>& norm_bias, double eps,
    int64_t norm_kind, int64_t microchunk_rows, int64_t comm_address,
    int64_t tp_size, int64_t p2p_handle) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(
      a.is_cuda() && b.is_cuda() && weight.is_cuda() && residual.is_cuda(),
      "TPSP requires CUDA tensors");
  STD_TORCH_CHECK(a.scalar_type() == ScalarType::BFloat16 &&
                      b.scalar_type() == ScalarType::BFloat16 &&
                      weight.scalar_type() == ScalarType::BFloat16 &&
                      residual.scalar_type() == ScalarType::BFloat16,
                  "TPSP requires BF16 tensors");
  STD_TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && weight.dim() == 1 &&
                      residual.dim() == 2 && a.is_contiguous() &&
                      b.is_contiguous() && weight.is_contiguous() &&
                      residual.is_contiguous(),
                  "TPSP requires contiguous matrices and vectors");
  STD_TORCH_CHECK(a.device() == b.device() && a.device() == weight.device() &&
                      a.device() == residual.device(),
                  "TPSP tensors must reside on the same CUDA device");
  STD_TORCH_CHECK(norm_kind == 0 || norm_kind == 1,
                  "TPSP norm must be RMSNorm or LayerNorm");
  STD_TORCH_CHECK(!norm_bias || norm_kind == 1,
                  "TPSP norm bias requires LayerNorm");
  for (const auto* bias : {&projection_bias, &norm_bias}) {
    if (*bias) {
      STD_TORCH_CHECK(
          bias->value().is_cuda() &&
              bias->value().scalar_type() == ScalarType::BFloat16 &&
              bias->value().device() == a.device() &&
              bias->value().is_contiguous() && bias->value().dim() == 1 &&
              bias->value().size(0) == b.size(1),
          "TPSP bias must be a contiguous CUDA BF16 hidden-size vector");
    }
  }
  STD_TORCH_CHECK(a.size(1) == b.size(0) && b.size(1) == weight.size(0) &&
                      tp_size >= 2 && microchunk_rows > 0 &&
                      comm_address != 0 && eps > 0,
                  "Invalid TPSP shape or configuration");
  int64_t tokens = a.size(0);
  int64_t width = a.size(1);
  int64_t hidden = b.size(1);
  int64_t rows = (tokens + tp_size - 1) / tp_size;
  STD_TORCH_CHECK(
      tokens > 0 && residual.size(0) == rows && residual.size(1) == hidden,
      "TPSP residual shard has an unexpected shape");
  STD_TORCH_CHECK(tokens <= INT32_MAX && width <= INT32_MAX &&
                      hidden <= INT32_MAX && rows <= INT32_MAX &&
                      microchunk_rows <= INT32_MAX && tp_size <= INT32_MAX,
                  "TPSP dimensions exceed CUDA kernel limits");

  const torch::stable::accelerator::DeviceGuard guard(a.get_device_index());
  cudaStream_t stream = get_current_cuda_stream();
  cublasHandle_t blas = get_current_cuda_blas_handle();
  ncclComm_t comm = reinterpret_cast<ncclComm_t>(comm_address);
  int64_t max_chunk = std::min(rows, microchunk_rows);
  int64_t num_chunks = (rows + max_chunk - 1) / max_chunk;
  bool p2p = p2p_handle != 0;
  std::shared_ptr<TpspP2pContext> p2p_context;
  bf16* local_inbox = nullptr;
  bf16* local_gather = nullptr;
  unsigned int* local_flags = nullptr;
  const unsigned int* one = nullptr;
  int64_t slot_bytes = 0;
  int64_t flag_offset = 0;
  if (p2p) {
    {
      std::lock_guard<std::mutex> lock(tpsp_context_mutex);
      auto it = tpsp_contexts.find(p2p_handle);
      STD_TORCH_CHECK(it != tpsp_contexts.end(), "Invalid TPSP P2P context");
      p2p_context = it->second;
    }
    STD_TORCH_CHECK(p2p_context->device_index == a.get_device_index() &&
                        p2p_context->tp_size == tp_size &&
                        p2p_context->hidden_size == hidden &&
                        max_chunk <= p2p_context->max_chunk_rows &&
                        p2p_context->comm == comm,
                    "TPSP P2P context does not match the request");
    auto* local_base = reinterpret_cast<char*>(p2p_context->workspace);
    slot_bytes = p2p_context->slot_bytes;
    flag_offset = p2p_context->flag_offset;
    local_inbox = reinterpret_cast<bf16*>(local_base);
    local_gather = reinterpret_cast<bf16*>(local_base + tp_size * slot_bytes);
    local_flags = reinterpret_cast<unsigned int*>(local_base + flag_offset);
    one = reinterpret_cast<const unsigned int*>(p2p_context->signal_one);
  }
  STD_TORCH_CHECK(comm_address != 0, "TPSP requires an NCCL communicator");
  int rank = p2p ? p2p_context->rank : -1;
  PipelineState* pipeline = nullptr;
  if (num_chunks > 1) {
    auto& state = pipeline_states[a.get_device_index()];
    if (!state) {
      state = std::make_unique<PipelineState>();
    }
    pipeline = state.get();
  }
  std::vector<ChunkBuffers> buffers;
  for (int slot = 0; slot < (pipeline ? 2 : 1); ++slot) {
    buffers.push_back({
        make_bf16(a, max_chunk * tp_size, width),
        make_bf16(a, max_chunk * tp_size, hidden),
        make_bf16(a, max_chunk, hidden),
        make_bf16(a, max_chunk * tp_size, hidden),
    });
  }
  auto new_residual = make_bf16(a, rows, hidden);
  auto normalized = make_bf16(a, rows, hidden);
  auto gathered = make_bf16(a, tokens, hidden);

  auto* a_ptr = reinterpret_cast<const bf16*>(a.const_data_ptr());
  auto* b_ptr = reinterpret_cast<const bf16*>(b.const_data_ptr());
  auto* residual_ptr = reinterpret_cast<const bf16*>(residual.const_data_ptr());
  auto* new_residual_ptr =
      reinterpret_cast<bf16*>(new_residual.mutable_data_ptr());
  auto* normalized_ptr = reinterpret_cast<bf16*>(normalized.mutable_data_ptr());
  auto* output_ptr = reinterpret_cast<bf16*>(gathered.mutable_data_ptr());
  auto* weight_ptr = reinterpret_cast<const bf16*>(weight.const_data_ptr());
  auto* projection_bias_ptr =
      projection_bias
          ? reinterpret_cast<const bf16*>(projection_bias->const_data_ptr())
          : nullptr;
  auto* norm_bias_ptr =
      norm_bias ? reinterpret_cast<const bf16*>(norm_bias->const_data_ptr())
                : nullptr;
  constexpr int threads = 256;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (int64_t offset = 0; offset < rows; offset += max_chunk) {
    int slot = (offset / max_chunk) % buffers.size();
    if (pipeline && offset >= 2 * max_chunk) {
      STD_CUDA_CHECK(
          cudaStreamWaitEvent(stream, pipeline->comm_done[slot].handle, 0));
    }
    auto& scratch = buffers[slot];
    auto* packed_ptr =
        reinterpret_cast<bf16*>(scratch.packed.mutable_data_ptr());
    auto* partial_ptr =
        reinterpret_cast<bf16*>(scratch.partial.mutable_data_ptr());
    auto* local_ptr = reinterpret_cast<bf16*>(scratch.local.mutable_data_ptr());
    auto* chunk_ptr =
        reinterpret_cast<bf16*>(scratch.gathered.mutable_data_ptr());
    int chunk_rows = static_cast<int>(std::min(max_chunk, rows - offset));
    bool whole_input = tokens % tp_size == 0 && chunk_rows == rows;
    // Stage 1: Project each rank's input rows; pack only for uneven shards.
    if (tokens % tp_size == 0) {
      STD_TORCH_CHECK(
          cublasGemmStridedBatchedEx(
              blas, CUBLAS_OP_N, CUBLAS_OP_N, hidden, chunk_rows, width, &alpha,
              b_ptr, CUDA_R_16BF, hidden, 0, a_ptr + offset * width,
              CUDA_R_16BF, width, rows * width, &beta, partial_ptr, CUDA_R_16BF,
              hidden, int64_t(chunk_rows) * hidden, tp_size, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS,
          "TPSP BF16 batched GEMM failed");
    } else {
      int64_t pack_elems = chunk_rows * tp_size * width;
      pack_tpsp_chunk<<<(pack_elems + threads - 1) / threads, threads, 0,
                        stream>>>(a_ptr, packed_ptr, tokens, width, rows,
                                  chunk_rows, offset, tp_size);
      STD_CUDA_CHECK(cudaGetLastError());
      STD_TORCH_CHECK(
          cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, hidden,
                       chunk_rows * tp_size, width, &alpha, b_ptr, CUDA_R_16BF,
                       hidden, packed_ptr, CUDA_R_16BF, width, &beta,
                       partial_ptr, CUDA_R_16BF, hidden, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS,
          "TPSP BF16 GEMM failed");
    }
    cudaStream_t comm_stream = pipeline ? pipeline->comm_stream : stream;
    if (pipeline) {
      STD_CUDA_CHECK(
          cudaEventRecord(pipeline->gemm_ready[slot].handle, stream));
      STD_CUDA_CHECK(cudaStreamWaitEvent(comm_stream,
                                         pipeline->gemm_ready[slot].handle, 0));
    }
    int64_t shard_elems = int64_t(chunk_rows) * hidden;
    size_t shard_bytes = shard_elems * sizeof(bf16);
    // Stage 2: Exchange projected shards via P2P or reduce-scatter with NCCL.
    if (p2p) {
      // IPC mappings provide a local device pointer to each peer's workspace.
      for (int dest = 0; dest < tp_size; ++dest) {
        if (dest == rank) {
          continue;
        }
        auto* peer_base = reinterpret_cast<char*>(p2p_context->peers[dest]);
        STD_CUDA_CHECK(cudaMemcpyAsync(
            peer_base + rank * slot_bytes, partial_ptr + dest * shard_elems,
            shard_bytes, cudaMemcpyDeviceToDevice, comm_stream));
        STD_CUDA_CHECK(cudaMemcpyAsync(
            peer_base + flag_offset +
                (kTpspInboxReady * tp_size + rank) * sizeof(unsigned int),
            one, sizeof(unsigned int), cudaMemcpyDeviceToDevice, comm_stream));
      }
      wait_and_clear_tpsp_signals<<<(tp_size + 255) / 256, 256, 0,
                                    comm_stream>>>(
          local_flags + kTpspInboxReady * tp_size, rank, tp_size);
      STD_CUDA_CHECK(cudaGetLastError());
    } else {
      STD_TORCH_CHECK(
          ncclReduceScatter(partial_ptr, local_ptr, shard_elems, ncclBfloat16,
                            ncclSum, comm, comm_stream) == ncclSuccess,
          "TPSP reduce-scatter failed");
    }

    auto* residual_chunk = residual_ptr + offset * hidden;
    auto* new_residual_chunk = new_residual_ptr + offset * hidden;
    auto* normalized_chunk = normalized_ptr + offset * hidden;
    auto* reduced_ptr = p2p ? partial_ptr + rank * shard_elems : local_ptr;
    auto* remote_ptr = p2p ? local_inbox : nullptr;
    int64_t remote_stride = slot_bytes / sizeof(bf16);
    // Stage 3: Fuse the P2P sum, biases, residual add, and normalization.
    // NCCL has already reduced the shard, so remote_ptr is null in that path.
    bool aligned =
        hidden % 8 == 0 && ((reinterpret_cast<uintptr_t>(reduced_ptr) |
                             reinterpret_cast<uintptr_t>(remote_ptr) |
                             reinterpret_cast<uintptr_t>(residual_chunk) |
                             reinterpret_cast<uintptr_t>(weight_ptr) |
                             reinterpret_cast<uintptr_t>(projection_bias_ptr) |
                             reinterpret_cast<uintptr_t>(norm_bias_ptr) |
                             reinterpret_cast<uintptr_t>(new_residual_chunk) |
                             reinterpret_cast<uintptr_t>(normalized_chunk)) &
                            15) == 0;
    if (aligned) {
      if (norm_kind == 0) {
        tpsp_add_norm_vector<false><<<chunk_rows, threads, 0, comm_stream>>>(
            reduced_ptr, remote_ptr, remote_stride, rank, tp_size,
            residual_chunk, projection_bias_ptr, weight_ptr, norm_bias_ptr,
            new_residual_chunk, normalized_chunk, hidden,
            static_cast<float>(eps));
      } else {
        tpsp_add_norm_vector<true><<<chunk_rows, threads, 0, comm_stream>>>(
            reduced_ptr, remote_ptr, remote_stride, rank, tp_size,
            residual_chunk, projection_bias_ptr, weight_ptr, norm_bias_ptr,
            new_residual_chunk, normalized_chunk, hidden,
            static_cast<float>(eps));
      }
    } else {
      if (norm_kind == 0) {
        tpsp_add_norm<false><<<chunk_rows, threads, 0, comm_stream>>>(
            reduced_ptr, remote_ptr, remote_stride, rank, tp_size,
            residual_chunk, projection_bias_ptr, weight_ptr, norm_bias_ptr,
            new_residual_chunk, normalized_chunk, hidden,
            static_cast<float>(eps));
      } else {
        tpsp_add_norm<true><<<chunk_rows, threads, 0, comm_stream>>>(
            reduced_ptr, remote_ptr, remote_stride, rank, tp_size,
            residual_chunk, projection_bias_ptr, weight_ptr, norm_bias_ptr,
            new_residual_chunk, normalized_chunk, hidden,
            static_cast<float>(eps));
      }
    }
    STD_CUDA_CHECK(cudaGetLastError());
    // Stage 4: All-gather normalized shards, waiting for P2P readers before
    // peers can reuse their gather slots.
    if (p2p) {
      auto* target = whole_input ? output_ptr : chunk_ptr;
      auto* own_output = target + rank * shard_elems;
      STD_CUDA_CHECK(cudaMemcpyAsync(own_output, normalized_chunk, shard_bytes,
                                     cudaMemcpyDeviceToDevice, comm_stream));
      for (int dest = 0; dest < tp_size; ++dest) {
        if (dest == rank) {
          continue;
        }
        auto* peer_base = reinterpret_cast<char*>(p2p_context->peers[dest]);
        STD_CUDA_CHECK(cudaMemcpyAsync(
            peer_base + tp_size * slot_bytes + rank * slot_bytes,
            normalized_chunk, shard_bytes, cudaMemcpyDeviceToDevice,
            comm_stream));
        STD_CUDA_CHECK(cudaMemcpyAsync(
            peer_base + flag_offset +
                (kTpspGatherReady * tp_size + rank) * sizeof(unsigned int),
            one, sizeof(unsigned int), cudaMemcpyDeviceToDevice, comm_stream));
      }
      wait_and_clear_tpsp_signals<<<(tp_size + 255) / 256, 256, 0,
                                    comm_stream>>>(
          local_flags + kTpspGatherReady * tp_size, rank, tp_size);
      STD_CUDA_CHECK(cudaGetLastError());
      for (int source = 0; source < tp_size; ++source) {
        if (source == rank) {
          continue;
        }
        STD_CUDA_CHECK(cudaMemcpyAsync(
            target + source * shard_elems,
            reinterpret_cast<char*>(local_gather) + source * slot_bytes,
            shard_bytes, cudaMemcpyDeviceToDevice, comm_stream));
      }
      for (int dest = 0; dest < tp_size; ++dest) {
        if (dest == rank) {
          continue;
        }
        auto* peer_base = reinterpret_cast<char*>(p2p_context->peers[dest]);
        STD_CUDA_CHECK(cudaMemcpyAsync(
            peer_base + flag_offset +
                (kTpspGatherConsumed * tp_size + rank) * sizeof(unsigned int),
            one, sizeof(unsigned int), cudaMemcpyDeviceToDevice, comm_stream));
      }
      wait_and_clear_tpsp_signals<<<(tp_size + 255) / 256, 256, 0,
                                    comm_stream>>>(
          local_flags + kTpspGatherConsumed * tp_size, rank, tp_size);
      STD_CUDA_CHECK(cudaGetLastError());
    } else {
      STD_TORCH_CHECK(
          ncclAllGather(normalized_ptr + offset * hidden,
                        whole_input ? output_ptr : chunk_ptr, shard_elems,
                        ncclBfloat16, comm, comm_stream) == ncclSuccess,
          "TPSP all-gather failed");
    }
    // Stage 5: Place chunked shards back in token order; a full chunk already
    // has the output layout when the token count is divisible by tp_size.
    if (!whole_input) {
      int64_t output_elems = tp_size * chunk_rows * hidden;
      unpack_tpsp_chunk<<<(output_elems + threads - 1) / threads, threads, 0,
                          comm_stream>>>(chunk_ptr, output_ptr, tokens, hidden,
                                         rows, chunk_rows, offset);
      STD_CUDA_CHECK(cudaGetLastError());
    }
    if (pipeline) {
      STD_CUDA_CHECK(
          cudaEventRecord(pipeline->comm_done[slot].handle, comm_stream));
    }
  }
  if (pipeline) {
    STD_CUDA_CHECK(cudaStreamWaitEvent(
        stream, pipeline->comm_done[(num_chunks - 1) % 2].handle, 0));
  }
  return {new_residual, normalized, gathered};
}
