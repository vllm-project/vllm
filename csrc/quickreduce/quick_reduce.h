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

template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds_two_shot__ static void
allreduce_prototype_twoshot(T const* A, T* B, uint32_t N, uint32_t num_blocks,
                            int rank, uint8_t** dbuffer_list,
                            uint32_t data_offset, uint32_t flag_color,
                            int64_t data_size_per_phase) {
  int block = blockIdx.x;
  int grid = gridDim.x;

  while (block < num_blocks) {
    AllReduceKernel::run(A, B, N, block, rank, dbuffer_list, data_offset,
                         flag_color, data_size_per_phase);
    block += grid;
    flag_color++;
  }
}

#define TWOSHOT_DISPATCH(__codec)                                           \
  if (world_size == 2) {                                                    \
    using LineCodec = __codec<T, 2>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       flag_color, this->kMaxProblemSize);                  \
  } else if (world_size == 4) {                                             \
    using LineCodec = __codec<T, 4>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       flag_color, this->kMaxProblemSize);                  \
  } else if (world_size == 8) {                                             \
    using LineCodec = __codec<T, 8>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       flag_color, this->kMaxProblemSize);                  \
  }

enum QuickReduceQuantLevel {
  F16 = 0,
  INT8 = 1,
  INT6 = 2,
  INT4 = 3,
};

struct DeviceComms {
  // Max problem size is 2GB (in bytes) or half of uint32_t max value.
  int64_t kMaxProblemSize =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;

  // Max TP-8
  static int constexpr kMaxWorldSize = 8;

  bool initialized = false;
  uint32_t flag_color = 1;
  int world_size;
  int rank;

  uint8_t* dbuffer;
  uint8_t** dbuffer_list;
  hipIpcMemHandle_t buffer_ipc_handle;
  std::vector<hipIpcMemHandle_t> all_buffer_ipc_handles;
  std::vector<uint8_t*> buffer_list;
  uint32_t data_offset;

  DeviceComms() : initialized(false), world_size(1), rank(0) {}
  ~DeviceComms() { destroy(); }

  void init(int world_size, int rank,
            std::optional<int64_t> max_problem_size = std::nullopt) {
    destroy();
    this->world_size = world_size;
    this->rank = rank;
    if (max_problem_size.has_value() && max_problem_size.value() > 0) {
      this->kMaxProblemSize = max_problem_size.value();
    }
    // Allocate buffer size for worst case: F16 2-stage buffer.
    uint32_t flags_buffer_size =
        2 * world_size * kMaxNumBlocks * sizeof(uint32_t);
    static int64_t data_buffer_size = 2 * this->kMaxProblemSize;
    int64_t total_buffer_size = flags_buffer_size + data_buffer_size;
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

  template <typename T, bool cast_bf2half>
  void allreduce(T const* A, T* B, uint32_t N, int quant_level,
                 hipStream_t stream) {
    if (world_size != 2 && world_size != 4 && world_size != 8) {
      throw std::runtime_error("All Reduce not supported for world_size = " +
                               std::to_string(world_size));
    }

    // Configuration.
    uint32_t msg_size = N * sizeof(T);
    uint32_t num_blocks = divceil(msg_size, kTileSize);
    uint32_t grid = min(kMaxNumBlocks, num_blocks);
    auto quant_level_ = static_cast<QuickReduceQuantLevel>(quant_level);
    switch (quant_level_) {
      case QuickReduceQuantLevel::INT8:
        TWOSHOT_DISPATCH(CodecQ8)
        break;
      case QuickReduceQuantLevel::INT6:
        TWOSHOT_DISPATCH(CodecQ6)
        break;
      case QuickReduceQuantLevel::INT4:
        TWOSHOT_DISPATCH(CodecQ4)
        break;
      default:
        TWOSHOT_DISPATCH(CodecFP)
        break;
    }
    HIP_CHECK(cudaGetLastError());
    // Rotate the flag color.
    flag_color += divceil(N, grid);
  }
};

}  // namespace quickreduce