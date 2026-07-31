#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <torch/headeronly/version.h>
#include <cuda_runtime.h>

// This function assumes that `cpu_tensor` is a CPU tensor,
// and that UVA (Unified Virtual Addressing) is enabled.
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor) {
  STD_TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  const auto dtype = cpu_tensor.scalar_type();
  const auto layout = cpu_tensor.layout();
  const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA);

  // handle empty tensor
  if (cpu_tensor.numel() == 0) {
    return torch::stable::empty(cpu_tensor.sizes(), dtype, layout, cuda_dev);
  }

  // Try to obtain a zero-copy device pointer directly.  This succeeds for
  // any host allocation that the CUDA runtime recognises as registered or
  // pinned, regardless of what torch's is_pinned() reports.  Under GPU
  // Confidential Computing the runtime may classify the pointer as
  // "Managed" instead of "Host", causing is_pinned() to return false even
  // though the mapping is valid -- so we let the CUDA API be the arbiter.
  void* host_ptr = const_cast<void*>(cpu_tensor.mutable_data_ptr());
  void* device_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err == cudaSuccess) {
    return torch::stable::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(), cuda_dev, dtype,
        [base = cpu_tensor](void*) {});  // keep cpu tensor alive
  }

  // Zero-copy failed -- the memory is truly not pinned/registered.
  // Allocate a new pinned+mapped buffer and copy the data once.
  // NOTE: this path produces a *detached* copy; subsequent writes to the
  // original cpu_tensor will NOT be visible through the returned view.
  cudaGetLastError();  // clear the non-fatal error from the failed call above
  torch::stable::Tensor contiguous_cpu = torch::stable::contiguous(cpu_tensor);
  size_t nbytes = contiguous_cpu.numel() * contiguous_cpu.element_size();

  void* new_host_ptr = nullptr;
  err = cudaHostAlloc(&new_host_ptr, nbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaHostAlloc failed: ", cudaGetErrorString(err));
  }

  err = cudaMemcpy(new_host_ptr, contiguous_cpu.const_data_ptr(), nbytes,
                   cudaMemcpyDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(new_host_ptr);
    STD_TORCH_CHECK(false, "cudaMemcpy failed: ", cudaGetErrorString(err));
  }

  device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, new_host_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(new_host_ptr);
    STD_TORCH_CHECK(
        false, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

  auto deleter = [new_host_ptr](void*) { cudaFreeHost(new_host_ptr); };

  return torch::stable::from_blob(device_ptr, contiguous_cpu.sizes(),
                                  contiguous_cpu.strides(), cuda_dev,
                                  contiguous_cpu.scalar_type(), deleter);
}
