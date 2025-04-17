#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#ifdef USE_ROCM

  #include "quickreduce/quick_reduce.h"

fptr_t init_custom_qr(int64_t rank, int64_t world_size) {
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");
  DeviceComms* fptr = new DeviceComms();
  fptr->init(world_size, rank);
  return (fptr_t)fptr;
}

void qr_destroy(fptr_t _fa) {
  auto fa = reinterpret_cast<DeviceComms*>(_fa);
  fa->destroy();
  delete fa;
}

torch::Tensor qr_get_handle(fptr_t _fa) {
  auto fa = reinterpret_cast<DeviceComms*>(_fa);
  hipIpcMemHandle_t handle = fa->get_handle();
  auto device_index = c10::cuda::current_device();
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto data_handle =
      torch::empty({static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options);
  std::memcpy(data_handle.data_ptr(), &handle, sizeof(hipIpcMemHandle_t));
  return data_handle;
}

void qr_open_handles(fptr_t _fa, const std::vector<torch::Tensor>& handles) {
  auto fa = reinterpret_cast<DeviceComms*>(_fa);
  std::vector<hipIpcMemHandle_t> ipc_handles;
  ipc_handles.reserve(handles.size());
  for (auto& handle : handles) {
    // Ensure the tensor is on the same device as the current device.
    hipIpcMemHandle_t ipc_handle;
    std::memcpy(&ipc_handle, handle.data_ptr(), sizeof(hipIpcMemHandle_t));
    ipc_handles.push_back(ipc_handle);
  }
  fa->open_ipc_handles(ipc_handles);
}

void qr_all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                   int64_t algo_int) {
  auto fa = reinterpret_cast<DeviceComms*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.scalar_type(), at::ScalarType::Half)
      << "QR only supports half precision for now.";
  TORCH_CHECK_EQ(inp.numel(), out.numel());

  auto algo = static_cast<QuickReduceProfile>(algo_int);
  fa->allreduce(algo_int, stream, reinterpret_cast<half*>(inp.data_ptr()),
                reinterpret_cast<half*>(out.data_ptr()), out.numel());
}

int64_t qr_max_size() {
  return static_cast<int64_t>(DeviceComms::kMaxProblemSize);
}

#endif  // USE_ROCM