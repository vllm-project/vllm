#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>

#include "custom_all_reduce.cuh"
#include "torch_utils.h"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                      torch::stable::Tensor& rank_data, int64_t rank,
                      bool fully_connected) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");

  vllm::Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<vllm::Signal*>(fake_ipc_ptrs[i]);
  }
  return (fptr_t) new vllm::CustomAllreduce(ipc_ptrs, rank_data.data_ptr(),
                                            rank_data.numel(), rank, world_size,
                                            fully_connected);
}

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
bool _is_weak_contiguous(torch::stable::Tensor& t) {
  return t.is_contiguous() ||
         (get_storage_nbytes(t) - t.storage_offset() * t.element_size() ==
          t.numel() * t.element_size());
}

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
void all_reduce(fptr_t _fa, torch::stable::Tensor& inp,
                torch::stable::Tensor& out, fptr_t _reg_buffer,
                int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  int32_t device_index = inp.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  auto stream = get_current_cuda_stream(device_index);

  STD_TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  STD_TORCH_CHECK_EQ(inp.numel(), out.numel());
  STD_TORCH_CHECK(_is_weak_contiguous(out));
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    STD_TORCH_CHECK(input_size <= reg_buffer_sz_bytes);
    STD_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size,
                                   cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }

  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::Float: {
      fa->allreduce<float>(stream, reinterpret_cast<float*>(reg_buffer),
                           reinterpret_cast<float*>(out.mutable_data_ptr()),
                           out.numel());
      break;
    }
    case torch::headeronly::ScalarType::Half: {
      fa->allreduce<half>(stream, reinterpret_cast<half*>(reg_buffer),
                          reinterpret_cast<half*>(out.mutable_data_ptr()),
                          out.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case torch::headeronly::ScalarType::BFloat16: {
      fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "custom allreduce only supports float32, float16 and bfloat16");
  }
}

void dispose(fptr_t _fa) {
  delete reinterpret_cast<vllm::CustomAllreduce*>(_fa);
}

int64_t meta_size() { return sizeof(vllm::Signal); }

void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fake_ipc_ptrs.size() == fa->world_size_);
  void* ipc_ptrs[8];
  for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (int i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  bytes.reserve(handles.size());
  fa->register_graph_buffers(bytes, offsets);
}

std::tuple<fptr_t, torch::stable::Tensor> allocate_shared_buffer_and_handle(
    int64_t size) {
  int32_t device_index = torch::stable::accelerator::getCurrentDeviceIndex();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);

  auto stream = get_current_cuda_stream(device_index);
  void* buffer;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  STD_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Allocate buffer
#if defined(USE_ROCM)
  // data buffers need to be "uncached" for signal on MI200
  STD_CUDA_CHECK(
      hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached));
#else
  STD_CUDA_CHECK(cudaMalloc((void**)&buffer, size));
#endif
  STD_CUDA_CHECK(cudaMemsetAsync(buffer, 0, size, stream));
  STD_CUDA_CHECK(cudaStreamSynchronize(stream));
  STD_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Create IPC memhandle for the allocated buffer.
  // Will use it in open_mem_handle.
  torch::stable::Tensor handle = torch::stable::empty(
      {static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))},
      torch::headeronly::ScalarType::Byte, std::nullopt,
      torch::stable::Device(torch::stable::DeviceType::CPU));

  STD_CUDA_CHECK(
      cudaIpcGetMemHandle((cudaIpcMemHandle_t*)handle.data_ptr(), buffer));

  return std::make_tuple(reinterpret_cast<fptr_t>(buffer), handle);
}

fptr_t open_mem_handle(torch::stable::Tensor& mem_handle) {
  void* ipc_ptr;
  STD_CUDA_CHECK(cudaIpcOpenMemHandle(
      (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)mem_handle.data_ptr()),
      cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<fptr_t>(ipc_ptr);
}

void free_shared_buffer(fptr_t buffer) {
  STD_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
}
