#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include "custom_all_reduce.cuh"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
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
bool _is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() ==
          t.numel() * t.element_size());
}

void* prepare_reg_buffer(cudaStream_t stream, torch::Tensor& inp,
                         fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size,
                                  cudaMemcpyDeviceToDevice, stream));
    return reg_buffer;
  }
  return inp.data_ptr();
}

template <typename Fn>
void dispatch_custom_ar_out_dtype(torch::ScalarType dtype, Fn&& fn,
                                  const char* op_name) {
  switch (dtype) {
    case at::ScalarType::Float:
      fn(static_cast<float*>(nullptr));
      return;
    case at::ScalarType::Half:
      fn(static_cast<half*>(nullptr));
      return;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      fn(static_cast<nv_bfloat16*>(nullptr));
      return;
#endif
    default:
      throw std::runtime_error(std::string("custom ") + op_name +
                               " only supports float32, float16 and bfloat16");
  }
}

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  auto reg_buffer =
      prepare_reg_buffer(stream, inp, _reg_buffer, reg_buffer_sz_bytes);
  dispatch_custom_ar_out_dtype(
      out.scalar_type(),
      [&](auto* type_tag) {
        using T = std::remove_pointer_t<decltype(type_tag)>;
        fa->allreduce<T>(stream, reinterpret_cast<T*>(reg_buffer),
                         reinterpret_cast<T*>(out.data_ptr()), out.numel());
      },
      "allreduce");
}

/**
 * Performs an out-of-place all_gather and stores result in out.
 *
 * If
 * _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 *
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 *
 * copied into _reg_buffer.
 */
void all_gather(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK_EQ(inp.numel() * fa->world_size_, out.numel());
  TORCH_CHECK(inp.numel() <= std::numeric_limits<int>::max(),
              "custom all_gather input is too large");
  int size = static_cast<int>(inp.numel());

  auto reg_buffer =
      prepare_reg_buffer(stream, inp, _reg_buffer, reg_buffer_sz_bytes);

  switch (out.scalar_type()) {
    case at::ScalarType::Byte:
    case at::ScalarType::Float8_e4m3fn: {
      fa->allgather<uint8_t>(stream, reinterpret_cast<uint8_t*>(reg_buffer),
                             reinterpret_cast<uint8_t*>(out.data_ptr()), size);
      return;
    }
    case at::ScalarType::Float: {
      fa->allgather<float>(stream, reinterpret_cast<float*>(reg_buffer),
                           reinterpret_cast<float*>(out.data_ptr()), size);
      return;
    }
    case at::ScalarType::Half: {
      fa->allgather<half>(stream, reinterpret_cast<half*>(reg_buffer),
                          reinterpret_cast<half*>(out.data_ptr()), size);
      return;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
      fa->allgather<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), size);
      return;
    }
#endif
    default:
      throw std::runtime_error(
          "custom all_gather only supports uint8, float8_e4m3fn, float32, "
          "float16 and bfloat16");
  }
}

/**
 * Performs an out-of-place reduce_scatter and stores result in out.
 *
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 *
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 *
 * copied into _reg_buffer.
 */
void reduce_scatter(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                    fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK_EQ(inp.numel() % fa->world_size_, 0);
  TORCH_CHECK_EQ(inp.numel() / fa->world_size_, out.numel());

  auto reg_buffer =
      prepare_reg_buffer(stream, inp, _reg_buffer, reg_buffer_sz_bytes);
  dispatch_custom_ar_out_dtype(
      out.scalar_type(),
      [&](auto* type_tag) {
        using T = std::remove_pointer_t<decltype(type_tag)>;
        fa->reduce_scatter<T>(stream, reinterpret_cast<T*>(reg_buffer),
                              reinterpret_cast<T*>(out.data_ptr()),
                              inp.numel());
      },
      "reduce_scatter");
}

void reduce_scatter_with_publish(fptr_t _fa, torch::Tensor& inp,
                                 torch::Tensor& out, fptr_t _reg_buffer,
                                 int64_t reg_buffer_sz_bytes,
                                 int64_t ready_slot, int64_t ready_gen) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(_reg_buffer != 0,
              "reduce_scatter_with_publish requires a registered buffer");
  TORCH_CHECK(ready_slot >= 0 && ready_slot < vllm::kReadySlots,
              "ready_slot out of range");
  TORCH_CHECK(ready_gen > 0, "ready_gen must be > 0");
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK_EQ(inp.numel() % fa->world_size_, 0);
  TORCH_CHECK_EQ(inp.numel() / fa->world_size_, out.numel());

  auto reg_buffer =
      prepare_reg_buffer(stream, inp, _reg_buffer, reg_buffer_sz_bytes);
  fa->publish_slot_ready(stream, static_cast<int>(ready_slot),
                         static_cast<vllm::FlagType>(ready_gen));
  dispatch_custom_ar_out_dtype(
      out.scalar_type(),
      [&](auto* type_tag) {
        using T = std::remove_pointer_t<decltype(type_tag)>;
        fa->reduce_scatter<T>(stream, reinterpret_cast<T*>(reg_buffer),
                              reinterpret_cast<T*>(out.data_ptr()), inp.numel(),
                              static_cast<int>(ready_slot),
                              static_cast<vllm::FlagType>(ready_gen));
      },
      "reduce_scatter_with_publish");
}

void dispose(fptr_t _fa) {
  delete reinterpret_cast<vllm::CustomAllreduce*>(_fa);
}

int64_t meta_size() { return sizeof(vllm::Signal); }

void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs,
                     int64_t buffer_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  TORCH_CHECK(fake_ipc_ptrs.size() == fa->world_size_);
  TORCH_CHECK(buffer_bytes > 0, "buffer_bytes must be > 0");
  void* ipc_ptrs[8];
  for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs, static_cast<size_t>(buffer_bytes));
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

std::tuple<fptr_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size) {
  auto device_index = c10::cuda::current_device();
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  void* buffer;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Allocate buffer
#if defined(USE_ROCM)
  // data buffers need to be "uncached" for signal on MI200
  AT_CUDA_CHECK(
      hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached));
#else
  AT_CUDA_CHECK(cudaMalloc((void**)&buffer, size));
#endif
  AT_CUDA_CHECK(cudaMemsetAsync(buffer, 0, size, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Create IPC memhandle for the allocated buffer.
  // Will use it in open_mem_handle.
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto handle =
      torch::empty({static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options);
  AT_CUDA_CHECK(
      cudaIpcGetMemHandle((cudaIpcMemHandle_t*)handle.data_ptr(), buffer));

  return std::make_tuple(reinterpret_cast<fptr_t>(buffer), handle);
}

fptr_t open_mem_handle(torch::Tensor& mem_handle) {
  void* ipc_ptr;
  AT_CUDA_CHECK(cudaIpcOpenMemHandle(
      (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)mem_handle.data_ptr()),
      cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<fptr_t>(ipc_ptr);
}

void free_shared_buffer(fptr_t buffer) {
  AT_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
}
