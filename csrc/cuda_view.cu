#include <torch/all.h>
#include <torch/cuda.h>

// This function assumes that `cpu_tensor` is a CPU tensor allocated with pinned
// memory, and that UVA (Unified Virtual Addressing) is enabled.
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");
  TORCH_CHECK(cpu_tensor.is_contiguous(), "Input tensor must be contiguous");

  // Get raw host pointer from CPU tensor
  void* host_ptr = cpu_tensor.data_ptr();

  // Get a device pointer corresponding to the pinned host memory
  void* device_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  TORCH_CHECK(err == cudaSuccess,
              "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));

  // Construct a CUDA tensor from the device pointer.
  // We'll use the same sizes, strides, and dtype as the CPU tensor.
  auto sizes = cpu_tensor.sizes();
  auto strides = cpu_tensor.strides();
  auto options =
      cpu_tensor.options().device(torch::kCUDA);  // Change device to CUDA

  // from_blob signature: from_blob(void *data, IntArrayRef sizes, ..., Deleter,
  // const TensorOptions &) Provide a no-op deleter. The CPU tensor holds the
  // memory, so we don't free it here.
  auto deleter = [](void*) {
    // no-op, since the memory is owned by the original CPU tensor
  };

  torch::Tensor cuda_tensor =
      torch::from_blob(device_ptr, sizes, strides, deleter, options);

  TORCH_CHECK(cuda_tensor.device().is_cuda(),
              "Resulting tensor is not on CUDA device");
  TORCH_CHECK(cuda_tensor.sizes().equals(sizes), "Size mismatch");
  TORCH_CHECK(cuda_tensor.strides().equals(strides), "Stride mismatch");
  TORCH_CHECK(cuda_tensor.dtype() == cpu_tensor.dtype(), "Dtype mismatch");

  return cuda_tensor;
}
