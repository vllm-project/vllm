#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <cuda_runtime.h>

// This function assumes that `cpu_tensor` is a CPU tensor allocated with pinned
// memory, and that UVA (Unified Virtual Addressing) is enabled.
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor) {
  STD_TORCH_CHECK(cpu_tensor.is_cpu(), "Input tensor must be on CPU");

  // Get raw host pointer from CPU tensor
  void* host_ptr = cpu_tensor.data_ptr();

  // Get a device pointer corresponding to the pinned host memory
  void* device_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));

  // We'll use the same sizes, strides, and dtype as the CPU tensor.
  // TODO: check if layout is respected.
  torch::stable::Device cuda_device(torch::stable::DeviceType::CUDA);

  // use default no-op deleter, since the memory is owned by the original CPU
  // tensor
  torch::stable::Tensor cuda_tensor = torch::stable::from_blob(
      device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(), cuda_device,
      cpu_tensor.scalar_type());

  STD_TORCH_CHECK(cuda_tensor.is_cuda(),
                  "Resulting tensor is not on CUDA device");

  return cuda_tensor;
}
