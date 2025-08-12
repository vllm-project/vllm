#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

namespace vllm {

struct Signal {
  alignas(64) union {
    uint64_t flag;
    unsigned char data[8];
  } start;
  alignas(64) union {
    uint64_t flag;
    unsigned char data[8];
  } end;
};

struct Metadata {
  alignas(128) Signal sg;
  alignas(128) int counter;
};
static_assert(offsetof(Metadata, counter) == 128);
static_assert(sizeof(Metadata) == 256);

struct __align__(16) RankData { const void *__restrict__ ptrs[8]; };

struct RankSignals {
  volatile Signal *signals[8];
};

// like std::array, but aligned
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

// use packed type to maximize memory efficiency
// goal: generate ld.128 and st.128 instructions
template <typename T>
struct packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
  // the (A)ccumulator type for reduction
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

// scalar cast functions
DINLINE float upcast_s(half val) { return __half2float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

// scalar add functions
// for some reason when compiling with Pytorch, the + operator for half and
// bfloat is disabled so we call the intrinsics directly
DINLINE half &assign_add(half &a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float &assign_add(float &a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) {
  return __float2bfloat16(val);
}
DINLINE nv_bfloat16 &assign_add(nv_bfloat16 &a, nv_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N> &packed_assign_add(array_t<T, N> &a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) {
      out.data[i] = upcast_s(val.data[i]);
    }
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) {
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    }
    return out;
  }
}

// compute flag at compile time
__host__ __device__ constexpr uint64_t compute_flag(int ngpus) {
  auto m = std::numeric_limits<uint64_t>::max();
  return m >> ((8 - ngpus) * 8);
}

template <int ngpus>
DINLINE void start_sync(const RankSignals &sg, volatile Metadata *meta,
                        int rank) {
  constexpr auto FLAG = compute_flag(ngpus);
  if (blockIdx.x == 0) {
    if (threadIdx.x < ngpus)
      // simultaneously write to the corresponding byte to all other ranks.
      // Latency = 1 p2p write
      sg.signals[threadIdx.x]->start.data[rank] = 255;
    else if (threadIdx.x == 32)
      // reset
      meta->sg.end.flag = 0;
  }
  if (threadIdx.x == 0) {
    while (meta->sg.start.flag != FLAG)
      ;
  }
  __syncthreads();
}

template <int ngpus, bool final_sync = false>
DINLINE void end_sync(const RankSignals &sg, volatile Metadata *meta,
                      int rank) {
  constexpr auto FLAG = compute_flag(ngpus);
  __syncthreads();
  __shared__ int num;
  if (threadIdx.x == 0) num = atomicAdd((int *)&meta->counter, 1);
  __syncthreads();

  // Only the last completing block can perform the end synchronization
  // This can ensures when the final busy wait ends, all ranks must have
  // finished reading each other's buffer.
  if (num == gridDim.x - 1) {
    if (threadIdx.x == 32) {
      // reset in a different warp
      meta->counter = 0;
      meta->sg.start.flag = 0;
    } else if (threadIdx.x < ngpus) {
      // simultaneously write to the corresponding byte to all other ranks.
      // Latency = 1 p2p write
      sg.signals[threadIdx.x]->end.data[rank] = 255;
    }
    // if this is the final sync, only one block needs it
    // because kernel exit can serve as sync
    if constexpr (final_sync) {
      if (threadIdx.x == 0) {
        while (meta->sg.end.flag != FLAG)
          ;
      }
    }
  }
  if constexpr (!final_sync) {
    if (threadIdx.x == 0) {
      while (meta->sg.end.flag != FLAG)
        ;
    }
    __syncthreads();
  }
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P *ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData *_dp, RankSignals sg,
                               volatile Metadata *meta, T *__restrict__ result,
                               int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  start_sync<ngpus>(sg, meta, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P *)result)[idx] =
        packed_reduce<P, ngpus, A>((const P **)&dp.ptrs[0], idx);
  }
  end_sync<ngpus, true>(sg, meta, rank);
}

template <typename P>
DINLINE P *get_tmp_buf(volatile Signal *sg) {
  return (P *)(((Metadata *)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData *_dp, RankSignals sg,
                               volatile Metadata *meta, T *__restrict__ result,
                               int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  const P *ptrs[ngpus];
  P *tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P *)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  start_sync<ngpus>(sg, meta, rank);
  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  // Maybe TODO: replace this with per-block release-acquire
  // can save about 1-2us (not a lot though)
  end_sync<ngpus>(sg, meta, rank);

  // stage 2: allgather
  for (int idx = tid; idx < part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int dst_idx = ((rank + i) % ngpus) * part + idx;
      ((P *)result)[dst_idx] = tmps[i][idx];
    }
  }
  // process the last larger partition
  int remaining = size - part * ngpus;
  if (tid < remaining) {
    int dst_idx = tid + part * ngpus;
    ((P *)result)[dst_idx] = get_tmp_buf<P>(sg.signals[ngpus - 1])[part + tid];
  }

  // faster than this
  // for (int idx = tid; idx < size; idx += stride) {
  //   int target_rank = idx / part;
  //   if (target_rank == ngpus) target_rank -= 1;
  //   ((P *)result)[idx] = tmps[target_rank][idx - target_rank * part];
  // }
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_half_butterfly(RankData *_dp, RankSignals sg,
                                       volatile Metadata *meta,
                                       T *__restrict__ result, int rank,
                                       int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto tmp_out = get_tmp_buf<P>(sg.signals[rank]);
  constexpr int hg = ngpus / 2;
  // Actually not quite half butterfly.
  // This is an all-to-all within each group containing half of the ranks
  // followed by cross-group add. Equivalent to half butterfly when there
  // are 4 GPUs, a common case for PCIe cards like T4 and A10.
  const P *ptrs[hg];
  {
    int start = rank - rank % hg;
#pragma unroll
    for (int i = 0; i < hg; i++) {
      ptrs[i] = (const P *)_dp->ptrs[i + start];
    }
  }
  start_sync<ngpus>(sg, meta, rank);
  for (int idx = tid; idx < size; idx += stride) {
    tmp_out[idx] = packed_reduce<P, hg, A>(ptrs, idx);
  }
  end_sync<ngpus>(sg, meta, rank);

  auto src = get_tmp_buf<P>(sg.signals[(ngpus - 1) - rank % ngpus]);
  // do the cross group reduction
  for (int idx = tid; idx < size; idx += stride) {
    auto tmp = tmp_out[idx];
    packed_assign_add(tmp, src[idx]);
    ((P *)result)[idx] = tmp;
  }
}

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  bool full_nvlink_;

  // below are device pointers
  RankSignals sg_;
  std::unordered_map<void *, RankData *> buffers_;
  Metadata *meta_;

  // stores the registered device pointers from all ranks
  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void *> graph_unreg_buffers_;
  // a map from IPC handles to opened IPC pointers
  std::map<IPC_KEY, char *> ipc_handles_;

  /**
   * meta is a pointer to device metadata and temporary buffer for allreduce.
   *
   * There's a total of sizeof(Metadata) of prefix before the actual data,
   * so meta + 1 points to actual temporary buffer.
   *
   * note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor
   */
  CustomAllreduce(Metadata *meta, void *rank_data, size_t rank_data_sz,
                  const cudaIpcMemHandle_t *handles,
                  const std::vector<int64_t> &offsets, int rank,
                  bool full_nvlink = true)
      : rank_(rank),
        world_size_(offsets.size()),
        full_nvlink_(full_nvlink),
        meta_(meta),
        d_rank_data_base_(reinterpret_cast<RankData *>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      Metadata *rank_meta;
      if (i != rank_) {
        char *handle = open_ipc_handle(&handles[i]);
        handle += offsets[i];
        rank_meta = (Metadata *)handle;
      } else {
        rank_meta = meta_;
      }
      sg_.signals[i] = &rank_meta->sg;
    }
  }

  char *open_ipc_handle(const void *ipc_handle) {
    auto [it, new_handle] =
        ipc_handles_.insert({*((IPC_KEY *)ipc_handle), nullptr});
    if (new_handle) {
      char *ipc_ptr;
      CUDACHECK(cudaIpcOpenMemHandle((void **)&ipc_ptr,
                                     *((const cudaIpcMemHandle_t *)ipc_handle),
                                     cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::vector<uint8_t>, std::vector<int64_t>>
  get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::vector<uint8_t> handles(handle_sz * num_buffers, 0);
    std::vector<int64_t> offsets(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void *base_ptr;
      // note: must share the base address of each allocation, or we get wrong
      // address
      if (cuPointerGetAttribute(&base_ptr,
                                CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CUDACHECK(cudaIpcGetMemHandle(
          (cudaIpcMemHandle_t *)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char *)ptr) - ((char *)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer is overflowed by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  void register_buffer(const std::vector<std::string> &handles,
                       const std::vector<int64_t> &offsets, void *self) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      if (i != rank_) {
        char *handle = open_ipc_handle(handles[i].data());
        handle += offsets[i];
        data.ptrs[i] = handle;
      } else {
        data.ptrs[i] = self;
      }
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[self] = d_data;
  }

  // note: when registering graph buffers, we intentionally choose to not
  // deduplicate the addresses. That means if the allocator reuses some
  // addresses, they will be registered again. This is to account for the remote
  // possibility of different allocation patterns between ranks. For example,
  // rank 1 may get the same input address for the second allreduce, but rank 2
  // got a different address. IPC handles have internal reference counting
  // mechanism so overhead should be small.
  void register_graph_buffers(
      const std::vector<std::string> &handles,
      const std::vector<std::vector<int64_t>> &offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto &rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char *handle =
              open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                         sizeof(RankData) * num_buffers,
                         cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  /**
   * This is the result after careful grid search. Using 36 blocks give the best
   * or close to the best runtime on the devices I tried: A100, A10, A30, T4,
   * V100. You'll notice that NCCL kernels also only take a small amount of SMs.
   * Not quite sure the underlying reason, but my guess is that too many SMs
   * will cause contention on NVLink bus.
   */
  template <typename T>
  void allreduce(cudaStream_t stream, T *input, T *output, int size,
                 int threads = 512, int block_limit = 36) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce currently requires input length to be multiple "
          "of " +
          std::to_string(d));

    RankData *ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    auto bytes = size * sizeof(typename packed_t<T>::P);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);
#define KL(ngpus, name) \
  name<T, ngpus>        \
      <<<blocks, threads, 0, stream>>>(ptrs, sg_, meta_, output, rank_, size);
#define REDUCE_CASE(ngpus)                            \
  case ngpus: {                                       \
    if (world_size_ == 2) {                           \
      KL(ngpus, cross_device_reduce_1stage);          \
    } else if (full_nvlink_) {                        \
      if ((world_size_ <= 4 && bytes < 512 * 1024) || \
          (world_size_ <= 8 && bytes < 256 * 1024)) { \
        KL(ngpus, cross_device_reduce_1stage);        \
      } else {                                        \
        KL(ngpus, cross_device_reduce_2stage);        \
      }                                               \
    } else {                                          \
      KL(ngpus, cross_device_reduce_half_butterfly);  \
    }                                                 \
    break;                                            \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce only supports num gpus in (2,4,6,8). Actual num "
            "gpus = " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

  ~CustomAllreduce() {
    for (auto [_, ptr] : ipc_handles_) {
      CUDACHECK(cudaIpcCloseMemHandle(ptr));
    }
  }
};
/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and add
 a template instantiation:
 * template void CustomAllreduce::allreduce<half>(cudaStream_t, half *, half *,
 int, int, int);
*/
}  // namespace vllm
