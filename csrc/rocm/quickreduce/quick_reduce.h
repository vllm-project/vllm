#pragma once

#include <algorithm>
#include <stdexcept>
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
                            uint32_t data_offset, uint32_t* d_flag_counters,
                            int64_t data_size_per_phase) {
  int block = blockIdx.x;
  int grid = gridDim.x;

  // Load this block's counter from device memory and advance it on-device,
  // so the color keeps changing across graph replays instead of being frozen.
  uint32_t flag_color = d_flag_counters[blockIdx.x];

  while (block < num_blocks) {
    AllReduceKernel::run(A, B, N, block, rank, dbuffer_list, data_offset,
                         flag_color, data_size_per_phase);
    block += grid;
    flag_color++;
  }
  // All threads compute the same final value; one writer per block is enough.
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    d_flag_counters[blockIdx.x] = flag_color;
  }
}

template <typename Kernel, typename T>
__global__ __quickreduce_launch_bounds_two_shot__ static void
coll_chunks_prototype(T const* A, T* B, CollChunks chunks, uint32_t num_tiles,
                      int rank, uint8_t** dbuffer_list, uint64_t data_offset,
                      uint64_t flag_offset, uint32_t flag_color) {
  for (uint32_t tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    Kernel::run(A, B, chunks, tile, rank, dbuffer_list, data_offset,
                flag_offset, flag_color);
  }
}

#define TWOSHOT_DISPATCH(__codec)                                           \
  if (world_size == 2) {                                                    \
    using LineCodec = __codec<T, 2>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       d_flag_counters, this->kMaxProblemSize);             \
  } else if (world_size == 4) {                                             \
    using LineCodec = __codec<T, 4>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       d_flag_counters, this->kMaxProblemSize);             \
  } else if (world_size == 8) {                                             \
    using LineCodec = __codec<T, 8>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       d_flag_counters, this->kMaxProblemSize);             \
  }

// INT3 only retains good performance on TP2 (world_size == 2). On TP4/TP8
// the 3-bit codec's pack/unpack overhead outweighs the reduced communication
// volume, so INT3 is restricted to a TP2-only dispatch here.
#define TWOSHOT_DISPATCH_TP2_ONLY(__codec)                                  \
  if (world_size == 2) {                                                    \
    using LineCodec = __codec<T, 2>;                                        \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel, T>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       d_flag_counters, this->kMaxProblemSize);             \
  } else {                                                                  \
    throw std::runtime_error(                                               \
        "INT3 quick all-reduce is only supported for world_size == 2 "      \
        "(TP2); use INT4/NONE for larger world sizes.");                    \
  }

enum QuickReduceQuantLevel {
  // Keep these ids in sync with Python QuickReduceRegime enum.
  F16 = 0,   // full-precision fp16/bf16 communication
  INT8 = 1,  // symmetric int8 + per-block scale
  INT6 = 2,  // symmetric int6 + per-block scale
  INT4 = 3,  // symmetric int4 + per-block scale
  INT3 = 4,  // symmetric int3 + per-block scale (TP2 only)
};

struct DeviceComms {
  // Max problem size is 2GB (in bytes) or half of uint32_t max value.
  int64_t kMaxProblemSize =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;

  // Max TP-8
  static int constexpr kMaxWorldSize = 8;

  bool initialized = false;
  uint32_t* d_flag_counters = nullptr;
  int world_size;
  int rank;

  uint8_t* dbuffer;
  uint8_t** dbuffer_list;
  hipIpcMemHandle_t buffer_ipc_handle;
  std::vector<hipIpcMemHandle_t> all_buffer_ipc_handles;
  std::vector<uint8_t*> buffer_list;
  uint32_t data_offset;

  // Reduce-scatter region past the all-reduce buffers, so a rank still
  // reading its all-reduce slots never sees chunk traffic. Two copies
  // (flags, data) alternate between calls; see ReduceScatterRows.
  int64_t coll_region_size = 0;
  int64_t coll_flags_size = 0;
  uint64_t coll_base = 0;
  uint32_t coll_calls = 0;

  DeviceComms() : initialized(false), world_size(1), rank(0) {}
  ~DeviceComms() { destroy(); }

  void init(int world_size, int rank,
            std::optional<int64_t> max_problem_size = std::nullopt,
            std::optional<int64_t> coll_max_size = std::nullopt) {
    destroy();
    this->world_size = world_size;
    this->rank = rank;
    if (max_problem_size.has_value() && max_problem_size.value() > 0) {
      this->kMaxProblemSize = max_problem_size.value();
    }
    // Allocate buffer size for worst case: F16 2-stage buffer.
    uint32_t flags_buffer_size =
        2 * world_size * kMaxNumBlocks * sizeof(uint32_t);
    int64_t data_buffer_size = 2 * this->kMaxProblemSize;
    coll_region_size = 0;
    coll_flags_size = 0;
    if (coll_max_size.has_value() && coll_max_size.value() > 0) {
      coll_region_size = divceil(coll_max_size.value(), kTileSize) * kTileSize;
      // One flag per (tile, source) of the smallest transmitted tile.
      coll_flags_size =
          divceil((coll_region_size / 4096 + world_size) * sizeof(uint32_t),
                  kTileSize) *
          kTileSize;
    }
    coll_base = flags_buffer_size + data_buffer_size;
    int64_t total_buffer_size =
        coll_base + 2 * (coll_flags_size + coll_region_size);
    data_offset = flags_buffer_size;
    HIP_CHECK(hipExtMallocWithFlags((void**)&dbuffer, total_buffer_size,
                                    hipDeviceMallocUncached));

    // Clear the flags buffers.
    HIP_CHECK(hipMemset(dbuffer, 0, flags_buffer_size));
    if (coll_flags_size > 0) {
      HIP_CHECK(hipMemset(dbuffer + coll_base, 0, 2 * coll_flags_size));
    }
    coll_calls = 0;

    // One flag-color counter per block, advanced by the kernel. Start at 1
    // to stay clear of the flags buffer we just zeroed.
    HIP_CHECK(hipMalloc(&d_flag_counters, kMaxNumBlocks * sizeof(uint32_t)));
    {
      std::vector<uint32_t> init_color(kMaxNumBlocks, 1u);
      HIP_CHECK(hipMemcpy(d_flag_counters, init_color.data(),
                          kMaxNumBlocks * sizeof(uint32_t),
                          hipMemcpyHostToDevice));
    }

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
    // Allocated before `initialized` flips true, so free it on its own guard
    // to avoid a leak if init fails partway through.
    if (d_flag_counters) {
      HIP_CHECK(hipFree(d_flag_counters));
      d_flag_counters = nullptr;
    }
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
      case QuickReduceQuantLevel::INT3:
        TWOSHOT_DISPATCH_TP2_ONLY(CodecQ3)
        break;
      default:
        TWOSHOT_DISPATCH(CodecFP)
        break;
    }
    HIP_CHECK(cudaGetLastError());
  }

  // Chunk layout of one variable-size call; `unit` is bytes per length unit.
  // Returns the largest per-rank tile count, or 0 if a chunk is too large.
  uint32_t make_chunks(std::vector<int64_t> const& lens, int64_t unit,
                       CollChunks& chunks) const {
    if (static_cast<int>(lens.size()) != world_size) {
      throw std::runtime_error("one chunk length per rank is required");
    }
    uint32_t max_tiles = 0;
    int64_t off = 0;
    for (int r = 0; r < world_size; r++) {
      int64_t bytes = lens[r] * unit;
      if (lens[r] < 0 || bytes % 16 != 0) {
        throw std::runtime_error("chunk bytes must be 16B multiples");
      }
      if (bytes > coll_region_size) return 0;
      chunks.off[r] = static_cast<uint32_t>(off);
      chunks.len[r] = static_cast<uint32_t>(lens[r]);
      // An empty chunk still takes a tile: its flags order the calls.
      chunks.tiles[r] = std::max<uint32_t>(
          1, static_cast<uint32_t>(divceil(bytes, kTileSize)));
      max_tiles = std::max(max_tiles, chunks.tiles[r]);
      off += lens[r];
    }
    return max_tiles;
  }

  // Next call's (data, flag) offsets and flag color, alternating the regions.
  void next_coll_slots(uint64_t& data_off, uint64_t& flag_off,
                       uint32_t& color) {
    if (coll_region_size == 0) {
      throw std::runtime_error(
          "quick reduce-scatter needs a nonzero coll_max_size");
    }
    int parity = coll_calls & 1;
    color = ++coll_calls;
    flag_off = coll_base + parity * coll_flags_size;
    data_off = coll_base + 2 * coll_flags_size + parity * coll_region_size;
  }

  template <typename T, bool cast_bf2half, template <typename, int> class Codec>
  bool launch_reduce_scatter(T const* A, T* B, CollChunks const& chunks,
                             uint32_t max_tiles, hipStream_t stream) {
    uint64_t data_off, flag_off;
    uint32_t color;
    if (max_tiles == 0 ||
        int64_t(max_tiles) * world_size * Codec<T, 2>::kTransmittedTileSize >
            coll_region_size ||
        int64_t(max_tiles) * world_size * sizeof(uint32_t) > coll_flags_size) {
      return false;
    }
    next_coll_slots(data_off, flag_off, color);
    uint32_t grid = min(kMaxNumBlocks, max_tiles);
#define QR_RS_LAUNCH(__ws)                                                    \
  hipLaunchKernelGGL(                                                         \
      (coll_chunks_prototype<                                                 \
          ReduceScatterRows<T, Codec<T, __ws>, cast_bf2half>, T>),            \
      dim3(grid), dim3(kBlockTwoShot), 0, stream, A, B, chunks, max_tiles,    \
      rank, dbuffer_list, data_off, flag_off, color)
    if (world_size == 2) {
      QR_RS_LAUNCH(2);
    } else if (world_size == 4) {
      QR_RS_LAUNCH(4);
    } else if (world_size == 8) {
      QR_RS_LAUNCH(8);
    } else {
      throw std::runtime_error("reduce-scatter needs world_size 2, 4 or 8");
    }
#undef QR_RS_LAUNCH
    HIP_CHECK(cudaGetLastError());
    return true;
  }

  // Rank r receives the sum over ranks of elements [off_r, off_r + lens[r])
  // of their inputs, off_r = sum(lens[:r]). Returns false, launching nothing,
  // if the call does not fit the chunk region or the codec is INT3; the
  // decision only depends on `lens` and the codec, so all ranks agree.
  template <typename T, bool cast_bf2half>
  bool reduce_scatterv(T const* A, T* B, std::vector<int64_t> const& lens,
                       int quant_level, hipStream_t stream) {
    CollChunks chunks;
    uint32_t max_tiles = make_chunks(lens, sizeof(T), chunks);
    switch (static_cast<QuickReduceQuantLevel>(quant_level)) {
      case QuickReduceQuantLevel::INT8:
        return launch_reduce_scatter<T, cast_bf2half, CodecQ8>(
            A, B, chunks, max_tiles, stream);
      case QuickReduceQuantLevel::INT6:
        return launch_reduce_scatter<T, cast_bf2half, CodecQ6>(
            A, B, chunks, max_tiles, stream);
      case QuickReduceQuantLevel::INT4:
        return launch_reduce_scatter<T, cast_bf2half, CodecQ4>(
            A, B, chunks, max_tiles, stream);
      case QuickReduceQuantLevel::F16:
        return launch_reduce_scatter<T, cast_bf2half, CodecFP>(
            A, B, chunks, max_tiles, stream);
      default:
        return false;
    }
  }
};

}  // namespace quickreduce