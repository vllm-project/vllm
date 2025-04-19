#ifdef USE_ROCM

  #include <hip/hip_runtime.h>

  #include "quick_reduce_impl.cuh"
  #include "quick_reduce.h"

void DeviceComms::init(int world_size, int rank) {
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

void DeviceComms::destroy() {
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

void DeviceComms::open_ipc_handles(
    std::vector<hipIpcMemHandle_t> const& ipc_handles) {
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

// ============================================================
// KERNEL
// ============================================================
template <typename AllReduceKenel>
__global__ __quickreduce_launch_bounds__ static void allreduce_prototype(
    half const* A, half* B, int N, int num_blocks, int world_size, int rank,
    uint8_t** dbuffer_list, long data_offset, int flag_color) {
  int block = blockIdx.x;
  int grid = gridDim.x;

  while (block < num_blocks) {
    AllReduceKenel::run(A, B, N, block, num_blocks, world_size, rank,
                        dbuffer_list, data_offset, flag_color);
    block += grid;
  }
}

  // ============================================================
  // DISPATCH
  // ============================================================
  #define TWOSHOT_DISPATCH(__codec)                                          \
    if (world_size == 2) {                                                   \
      using LineCodec = __codec<2>;                                          \
      using AllReduceKernel = AllReduceTwoshot<LineCodec>;                   \
      hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>), dim3(grid), \
                         dim3(kBlock), 0, stream, A, B, N, num_blocks,       \
                         world_size, rank, dbuffer_list, data_offset,        \
                         flag_color);                                        \
    } else if (world_size == 4) {                                            \
      using LineCodec = __codec<4>;                                          \
      using AllReduceKernel = AllReduceTwoshot<LineCodec>;                   \
      hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>), dim3(grid), \
                         dim3(kBlock), 0, stream, A, B, N, num_blocks,       \
                         world_size, rank, dbuffer_list, data_offset,        \
                         flag_color);                                        \
    } else if (world_size == 8) {                                            \
      using LineCodec = __codec<8>;                                          \
      using AllReduceKernel = AllReduceTwoshot<LineCodec>;                   \
      hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>), dim3(grid), \
                         dim3(kBlock), 0, stream, A, B, N, num_blocks,       \
                         world_size, rank, dbuffer_list, data_offset,        \
                         flag_color);                                        \
    }

void DeviceComms::allreduce(int profile, hipStream_t stream, half const* A,
                            half* B, int N) {
  if (world_size != 2 && world_size != 4 && world_size != 8) {
    throw std::runtime_error("All Reduce not supported for world_size = " +
                             std::to_string(world_size));
  }

  // Configuration.
  long msg_size = N * sizeof(half);
  unsigned long num_blocks = divceil(msg_size, kTileSize);
  unsigned long grid = min(304 * 4, num_blocks);
  // -------------------------------------------------
  // All reduce dispatch.
  QuickReduceProfile dprofile = static_cast<QuickReduceProfile>(profile);

  switch (dprofile) {
    case QuickReduceProfile::ONESHOT_FP16:
      using AllReduceKernel = AllReduceOneshot;
      hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>), dim3(grid),
                         dim3(kBlock), 0, stream, A, B, N, num_blocks,
                         world_size, rank, dbuffer_list, data_offset,
                         flag_color);
      break;
    case QuickReduceProfile::TWOSHOT_FP8:
      throw std::runtime_error("FP8 is not supported");
      // TWOSHOT_DISPATCH(TwoshotFP8LineCodec)
      break;
    case QuickReduceProfile::TWOSHOT_Q8:
      TWOSHOT_DISPATCH(TwoshotQ8LineCodec)
      break;
    case QuickReduceProfile::TWOSHOT_MAX_MIN_Q8:
      TWOSHOT_DISPATCH(TwoshotMaxMinQ8LineCodec)
      break;
    case QuickReduceProfile::TWOSHOT_Q6:
      TWOSHOT_DISPATCH(TwoshotQ6LineCodec)
      break;
    case QuickReduceProfile::TWOSHOT_Q4:
      TWOSHOT_DISPATCH(TwoshotQ4LineCodec)
      break;
    default:
      TWOSHOT_DISPATCH(TwoshotFP16LineCodec)
      break;
  }
  HIP_CHECK(cudaGetLastError());

  // -------------------------------------------------
  // Rotate the flag color.
  flag_color++;
}

#endif  // USE_ROCM