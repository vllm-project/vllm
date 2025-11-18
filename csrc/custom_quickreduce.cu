#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#ifdef USE_ROCM

  #include "quickreduce/quick_reduce.h"

quickreduce::fptr_t init_custom_qr(int64_t rank, int64_t world_size,
                                   std::optional<int64_t> qr_max_size) {
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size == 6)
    throw std::invalid_argument("world size == 6 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");
  quickreduce::DeviceComms* fptr = new quickreduce::DeviceComms();
  fptr->init(world_size, rank, qr_max_size);
  return (quickreduce::fptr_t)fptr;
}

void qr_destroy(quickreduce::fptr_t _fa) {
  if (_fa) {
    auto fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
    fa->destroy();
    delete fa;
  }
}

torch::Tensor qr_get_handle(quickreduce::fptr_t _fa) {
  auto fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
  hipIpcMemHandle_t handle = fa->get_handle();
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto data_handle =
      torch::empty({static_cast<int64_t>(sizeof(hipIpcMemHandle_t))}, options);
  std::memcpy(data_handle.data_ptr(), &handle, sizeof(hipIpcMemHandle_t));
  return data_handle;
}

void qr_open_handles(quickreduce::fptr_t _fa,
                     const std::vector<torch::Tensor>& handles) {
  auto fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
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

void qr_all_reduce(quickreduce::fptr_t _fa, torch::Tensor& inp,
                   torch::Tensor& out, int64_t quant_level, bool cast_bf2half) {
  auto fa = reinterpret_cast<quickreduce::DeviceComms*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK_LE(out.numel(), fa->kMaxProblemSize);
  if (out.scalar_type() == at::ScalarType::Half) {
    fa->allreduce<half, false>(reinterpret_cast<half*>(inp.data_ptr()),
                               reinterpret_cast<half*>(out.data_ptr()),
                               out.numel(), quant_level, stream);
  } else if (out.scalar_type() == at::ScalarType::BFloat16) {
    if (cast_bf2half) {
      fa->allreduce<half, true>(reinterpret_cast<half*>(inp.data_ptr()),
                                reinterpret_cast<half*>(out.data_ptr()),
                                out.numel(), quant_level, stream);
    } else {
      fa->allreduce<quickreduce::nv_bfloat16, false>(
          reinterpret_cast<quickreduce::nv_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<quickreduce::nv_bfloat16*>(out.data_ptr()),
          out.numel(), quant_level, stream);
    }
  } else {
    throw std::runtime_error(
        "quick allreduce only supports float16 and bfloat16");
  }
}

int64_t qr_max_size() {
  // The default is 2GB (2,147,483,648 bytes)
  return static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
}

  #define INSTANTIATE_FOR_WORLDSIZE(T, Codec, cast_bf2half)       \
    template struct quickreduce::AllReduceTwoshot<T, Codec<T, 2>, \
                                                  cast_bf2half>;  \
    template struct quickreduce::AllReduceTwoshot<T, Codec<T, 4>, \
                                                  cast_bf2half>;  \
    template struct quickreduce::AllReduceTwoshot<T, Codec<T, 8>, cast_bf2half>;

INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecFP, false)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ4, false)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ6, false)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ8, false)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecFP, true)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ4, true)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ6, true)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ8, true)

INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecFP, false)
INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecQ4, false)
INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecQ6, false)
INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecQ8, false)

#endif  // USE_ROCM