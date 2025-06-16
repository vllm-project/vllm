#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include "quick_all_reduce.cuh"

namespace quickreduce {

// ============================================================
// CONTEXT
// ============================================================
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
  hipMemset(dbuffer, 0, flags_buffer_size);

  // Device-side list of IPC buffers.
  buffer_list.resize(world_size);
  hipMalloc(&dbuffer_list, world_size * sizeof(uint8_t*));

  // Create IPC handles for rank's communication buffer.
  all_buffer_ipc_handles.resize(world_size);
  hipIpcGetMemHandle(&buffer_ipc_handle, dbuffer);

  initialized = true;
}

void DeviceComms::destroy() {
  if (initialized) {
    for (int i = 0; i < world_size; i++) {
      if (i != rank) {
        hipIpcCloseMemHandle(dbuffer_list[i]);
      }
    }

    hipFree(dbuffer);
    hipFree(dbuffer_list);

    initialized = false;
  }
}

void DeviceComms::open_ipc_handles(
    std::vector<hipIpcMemHandle_t> const& ipc_handles) {
  for (int i = 0; i < world_size; i++) {
    all_buffer_ipc_handles[i] = ipc_handles[i];
  }

  // Open device memory access to the IPC communication buffers.
  // Note: For our own rank, we do not need to open a handle.
  for (int i = 0; i < world_size; i++) {
    if (i != rank) {
      hipIpcOpenMemHandle((void**)&buffer_list[i], all_buffer_ipc_handles[i],
                          hipIpcMemLazyEnablePeerAccess);
    } else {
      buffer_list[i] = dbuffer;
    }
  }

  hipMemcpy(dbuffer_list, buffer_list.data(), world_size * sizeof(uint8_t*),
            hipMemcpyHostToDevice);
}

// ============================================================
// KERNEL
// ============================================================
template <typename AllReduceKenel, typename T>
__global__ __quickreduce_launch_bounds__ static void allreduce_prototype(
    T const* A, T* B, int N, int num_blocks, int world_size, int rank,
    uint8_t** dbuffer_list, long data_offset, int flag_color,
    bool cast_bf162half) {
  int block = blockIdx.x;
  int grid = gridDim.x;

  while (block < num_blocks) {
    AllReduceKenel::run(A, B, N, block, num_blocks, world_size, rank,
                        dbuffer_list, data_offset, flag_color, cast_bf162half);
    block += grid;
  }
}

// ============================================================
// DISPATCH
// ============================================================
#define TWOSHOT_DISPATCH(__codec)                                             \
  if (world_size == 2) {                                                      \
    using LineCodec = __codec<2, T>;                                          \
    using AllReduceKernel = AllReduceTwoshot<LineCodec, T>;                   \
    hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel, T>), dim3(grid), \
                       dim3(kBlock), 0, stream, A, B, N, num_blocks,          \
                       world_size, rank, dbuffer_list, data_offset,           \
                       flag_color, cast_bf162half);                           \
  } else if (world_size == 4) {                                               \
    using LineCodec = __codec<4, T>;                                          \
    using AllReduceKernel = AllReduceTwoshot<LineCodec, T>;                   \
    hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel, T>), dim3(grid), \
                       dim3(kBlock), 0, stream, A, B, N, num_blocks,          \
                       world_size, rank, dbuffer_list, data_offset,           \
                       flag_color, cast_bf162half);                           \
  } else if (world_size == 8) {                                               \
    using LineCodec = __codec<8, T>;                                          \
    using AllReduceKernel = AllReduceTwoshot<LineCodec, T>;                   \
    hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel, T>), dim3(grid), \
                       dim3(kBlock), 0, stream, A, B, N, num_blocks,          \
                       world_size, rank, dbuffer_list, data_offset,           \
                       flag_color, cast_bf162half);                           \
  }

template <typename T>
void DeviceComms::allreduce(int profile, hipStream_t stream, T const* A, T* B,
                            int N, bool cast_bf162half) {
  static_assert(sizeof(T) == 2,
                "Template parameter T must be 16 bits (2 bytes) in size.");
  if (world_size != 2 && world_size != 4 && world_size != 8) {
    throw std::runtime_error("All Reduce not supported for world_size = " +
                             std::to_string(world_size));
  }

  // Configuration.
  long msg_size = N * sizeof(T);
  int num_blocks = divceil(msg_size, kTileSize);
  int grid = min(304 * 4, num_blocks);

  // -------------------------------------------------
  // All reduce dispatch.
  QuickReduceProfile dprofile = static_cast<QuickReduceProfile>(profile);
  switch (dprofile) {
    case QuickReduceProfile::TWOSHOT_FP8:
      TWOSHOT_DISPATCH(TwoshotFP8LineCodec)
      break;
    case QuickReduceProfile::TWOSHOT_Q8:
      TWOSHOT_DISPATCH(TwoshotQ8LineCodec)
      break;
    case QuickReduceProfile::TWOSHOT_Q6:
      TWOSHOT_DISPATCH(TwoshotQ6LineCodec)
      break;
    case QuickReduceProfile::TWOSHOT_Q4:
      TWOSHOT_DISPATCH(TwoshotQ4LineCodec)
      break;
    default:
      TWOSHOT_DISPATCH(TwoshotF16LineCodec)
      break;
  }

  // -------------------------------------------------
  // Rotate the flag color.
  flag_color++;
}

}  // namespace quickreduce

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
bool _is_weak_contiguous(torch::Tensor const& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() ==
          t.numel() * t.element_size());
}

fptr_t init_quick_ar(int64_t world_size, int64_t rank) {
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size == 6)
    throw std::invalid_argument("world size == 6 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");

  quickreduce::DeviceComms* dev_comm = new quickreduce::DeviceComms();
  dev_comm->init(world_size, rank);
  return reinterpret_cast<fptr_t>(dev_comm);
}

torch::Tensor qr_get_comm_handle(fptr_t _fa) {
  auto fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
  hipIpcMemHandle_t handle = fa->get_handle();
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  torch::Tensor tensor_handle =
      torch::empty({static_cast<int64_t>(sizeof(hipIpcMemHandle_t))}, options);
  std::memcpy(tensor_handle.data_ptr(), &handle, sizeof(hipIpcMemHandle_t));
  return tensor_handle;
}

void qr_set_comm_handles(fptr_t _fa,
                         std::vector<torch::Tensor> const& comm_handles) {
  auto fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
  auto world_size = comm_handles.size();
  std::vector<hipIpcMemHandle_t> ipc_handles(world_size);

  for (int i = 0; i < world_size; ++i) {
    const auto& tensor = comm_handles[i];
    TORCH_CHECK(tensor.device().is_cpu(), "Comm handle tensor must be on CPU");
    TORCH_CHECK(tensor.scalar_type() == torch::kUInt8,
                "Comm handle tensor must be of type uint8");
    TORCH_CHECK(tensor.numel() == sizeof(hipIpcMemHandle_t),
                "Comm handle tensor must have ", sizeof(hipIpcMemHandle_t),
                " elements");

    std::memcpy(&(ipc_handles[i]), tensor.data_ptr(),
                sizeof(hipIpcMemHandle_t));
  }
  fa->open_ipc_handles(ipc_handles);
}

void qr_all_reduce(fptr_t _fa, int64_t profile, torch::Tensor const& inp,
                   torch::Tensor& out, bool cast_bf162half) {
  quickreduce::DeviceComms* fa =
      reinterpret_cast<quickreduce::DeviceComms*>(_fa);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();  // hipStream_t

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));

  auto input_size = inp.numel() * inp.element_size();

  if (out.scalar_type() == at::ScalarType::Half) {
    fa->allreduce<half>(
        profile, stream, reinterpret_cast<half const*>(inp.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()), inp.numel(), false);
  } else if (out.scalar_type() == at::ScalarType::BFloat16) {
    if (cast_bf162half) {
      // change dtype in thread.
      fa->allreduce<half>(
          profile, stream, reinterpret_cast<half const*>(inp.data_ptr()),
          reinterpret_cast<half*>(out.data_ptr()), inp.numel(), true);
    } else {
      fa->allreduce<nv_bfloat16>(
          profile, stream, reinterpret_cast<nv_bfloat16 const*>(inp.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), inp.numel(), false);
    }
  } else {
    throw std::runtime_error(
        "quick allreduce only supports float16 and bfloat16 for now.");
  }
}

void qr_destroy(fptr_t _fa) {
  if (_fa) {
    auto fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
    fa->destroy();
    delete fa;
  }
}

void is_quickreduce_available() {};
