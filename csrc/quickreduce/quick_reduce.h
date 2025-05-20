#pragma once

#include <vector>
#include <hip/hip_runtime.h>
#include "quick_reduce_impl.cuh"

#define HIP_CHECK(err)                                                     \
  do {                                                                     \
    hipError_t err_ = (err);                                               \
    if (err_ != hipSuccess) {                                              \
      std::printf("HIP error %d at %s:%d. %s\n", err_, __FILE__, __LINE__, \
                  hipGetErrorString(err_));                                \
      throw std::runtime_error("HIP error");                               \
    }                                                                      \
  } while (0)

namespace quickreduce {
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

enum QuickReduceAlgo {
  ONESHOT_FP16 = 0,
  TWOSHOT_FP16 = 1,
  TWOSHOT_SYMM_Q8 = 2,
  TWOSHOT_SYMM_Q4 = 3,
  TWOSHOT_ASYMM_Q8 = 4,
  TWOSHOT_ASYMM_Q4 = 5,
};

template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds__ static void allreduce_prototype(
    T const* A, T* B, int N, int num_blocks, int world_size, int rank,
    uint8_t** dbuffer_list, long data_offset, int flag_color) {
  int block = blockIdx.x;
  int grid = gridDim.x;

  while (block < num_blocks) {
    AllReduceKernel::run(A, B, N, block, num_blocks, world_size, rank,
                         dbuffer_list, data_offset, flag_color);
    block += grid;
  }
}

#define TWOSHOT_DISPATCH(__codec)                                             \
  if (world_size == 2) {                                                      \
    using LineCodec = __codec<T, 2>;                                          \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec>;                   \
    hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel, T>), dim3(grid), \
                       dim3(kBlock), 0, stream, A, B, N, num_blocks,          \
                       world_size, rank, dbuffer_list, data_offset,           \
                       flag_color);                                           \
  } else if (world_size == 4) {                                               \
    using LineCodec = __codec<T, 4>;                                          \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec>;                   \
    hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel, T>), dim3(grid), \
                       dim3(kBlock), 0, stream, A, B, N, num_blocks,          \
                       world_size, rank, dbuffer_list, data_offset,           \
                       flag_color);                                           \
  } else if (world_size == 8) {                                               \
    using LineCodec = __codec<T, 8>;                                          \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec>;                   \
    hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel, T>), dim3(grid), \
                       dim3(kBlock), 0, stream, A, B, N, num_blocks,          \
                       world_size, rank, dbuffer_list, data_offset,           \
                       flag_color);                                           \
  }

struct DeviceComms {
  // Workgroup scope = Tile = (256 threads x 16B x 8 atoms)
  static long constexpr kTileSize = 256 * 16 * 8;

  // Max problem size is 8GB (in bytes)
  static long long constexpr kMaxProblemSize =
      static_cast<long long>(536870912) * 16;
  static long constexpr kMaxTiles = kMaxProblemSize / kTileSize;

  // Max TP-8
  static int constexpr kMaxWorldSize = 8;

  bool initialized = false;
  int flag_color = 1;
  int world_size;
  int rank;

  uint8_t* dbuffer;
  uint8_t** dbuffer_list;
  hipIpcMemHandle_t buffer_ipc_handle;
  std::vector<hipIpcMemHandle_t> all_buffer_ipc_handles;
  std::vector<uint8_t*> buffer_list;
  long data_offset;

  DeviceComms() : initialized(false), world_size(1), rank(0) {}
  ~DeviceComms() { destroy(); }

  void init(int world_size, int rank) {
    destroy();
    this->world_size = world_size;
    this->rank = rank;

    // Allocate buffer size for worst case: Twoshot FP16 2-stage buffer.
    long flags_buffer_size = 2 * world_size * kMaxTiles * sizeof(int);
    long data_buffer_size = 2 * kMaxProblemSize;
    long total_buffer_size = flags_buffer_size + data_buffer_size;
    data_offset = flags_buffer_size;
    HIP_CHECK(hipExtMallocWithFlags((void**)&dbuffer, total_buffer_size,
                                    hipDeviceMallocUncached));

    // Clear the flags buffer.
    HIP_CHECK(hipMemset(dbuffer, 0, flags_buffer_size));

    // Device-side list of IPC buffers.
    buffer_list.resize(world_size);
    HIP_CHECK(hipMalloc(&dbuffer_list, world_size * sizeof(uint8_t*)));

    // Create IPC handles for rank's communication buffer.
    all_buffer_ipc_handles.resize(world_size);
    HIP_CHECK(hipIpcGetMemHandle(&buffer_ipc_handle, dbuffer));

    initialized = true;
  }
  int get_world_size() { return world_size; }
  int get_rank() { return rank; }
  bool status() { return initialized; }
  hipIpcMemHandle_t const get_handle() { return buffer_ipc_handle; }

  void destroy() {
    if (initialized) {
      for (int i = 0; i < world_size; i++) {
        if (i != rank) {
          HIP_CHECK(hipIpcCloseMemHandle(dbuffer_list[i]));
        }
      }

      HIP_CHECK(hipFree(dbuffer));
      HIP_CHECK(hipFree(dbuffer_list));

      initialized = false;
    }
  }

  void open_ipc_handles(std::vector<hipIpcMemHandle_t> const& ipc_handles) {
    assert(ipc_handles.size() == all_buffer_ipc_handles.size());
    for (int i = 0; i < world_size; i++) {
      all_buffer_ipc_handles[i] = ipc_handles[i];
    }

    // Open device memory access to the IPC communication buffers.
    // Note: For our own rank, we do not need to open a handle.
    for (int i = 0; i < world_size; i++) {
      if (i != rank) {
        HIP_CHECK(hipIpcOpenMemHandle((void**)&buffer_list[i],
                                      all_buffer_ipc_handles[i],
                                      hipIpcMemLazyEnablePeerAccess));
      } else {
        buffer_list[i] = dbuffer;
      }
    }

    HIP_CHECK(hipMemcpy(dbuffer_list, buffer_list.data(),
                        world_size * sizeof(uint8_t*), hipMemcpyHostToDevice));
  }

  template <typename T>
  void allreduce(int algo_int, hipStream_t stream, T const* A, T* B, int N) {
    if (world_size != 2 && world_size != 4 && world_size != 8) {
      throw std::runtime_error("All Reduce not supported for world_size = " +
                               std::to_string(world_size));
    }

    // Configuration.
    long msg_size = N * sizeof(T);
    unsigned long num_blocks = divceil(msg_size, kTileSize);
    unsigned long grid = min(kMaxNumBlocks, num_blocks);

    // All reduce dispatch.
    QuickReduceAlgo algo = static_cast<QuickReduceAlgo>(algo_int);

    switch (algo) {
      case QuickReduceAlgo::ONESHOT_FP16:
        using AllReduceKernel = AllReduceOneshot<T>;
        hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel, T>),
                           dim3(grid), dim3(kBlock), 0, stream, A, B, N,
                           num_blocks, world_size, rank, dbuffer_list,
                           data_offset, flag_color);
        break;
      case QuickReduceAlgo::TWOSHOT_SYMM_Q8:
        TWOSHOT_DISPATCH(CodecQ8Symm)
        break;
      case QuickReduceAlgo::TWOSHOT_ASYMM_Q8:
        TWOSHOT_DISPATCH(CodecQ8Asymm)
        break;
      case QuickReduceAlgo::TWOSHOT_SYMM_Q4:
        TWOSHOT_DISPATCH(CodecQ4Symm)
        break;
      case QuickReduceAlgo::TWOSHOT_ASYMM_Q4:
        TWOSHOT_DISPATCH(CodecQ4Asymm)
        break;
      default:
        TWOSHOT_DISPATCH(CodecFP16)
        break;
    }
    HIP_CHECK(cudaGetLastError());

    // Rotate the flag color.
    flag_color++;
  }
};

}  // namespace quickreduce