#include <torch/all.h>
#include <torch/cuda.h>
#include <cuda_runtime.h>
#include <sys/mman.h>

// This function assumes that `cpu_tensor` is a CPU tensor,
// and that UVA (Unified Virtual Addressing) is enabled.
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  if (cpu_tensor.is_pinned()) {
    // If CPU tensor is pinned, directly get the device pointer.
    void* host_ptr = const_cast<void*>(cpu_tensor.data_ptr());
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));

    return torch::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(),
        [base = cpu_tensor](void*) {},  // keep cpu tensor alive
        cpu_tensor.options().device(torch::kCUDA));
  }

  // If CPU tensor is not pinned, allocate a new pinned memory buffer.
  torch::Tensor contiguous_cpu = cpu_tensor.contiguous();
  size_t nbytes = contiguous_cpu.nbytes();
  long page_size = sysconf(_SC_PAGESIZE);
  size_t aligned_size = (nbytes + page_size - 1) & ~(page_size - 1);

  void* host_ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  if (host_ptr == MAP_FAILED) {
    AT_ERROR("mmap failed to allocate ", aligned_size, " bytes");
  }

  std::memcpy(host_ptr, contiguous_cpu.data_ptr(), nbytes);

  cudaError_t err =
      cudaHostRegister(host_ptr, aligned_size, cudaHostRegisterDefault);
  if (err != cudaSuccess) {
    munmap(host_ptr, aligned_size);
    AT_ERROR("cudaHostRegister failed: ", cudaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);

  auto deleter = [host_ptr, aligned_size](void*) {
    cudaHostUnregister(host_ptr);
    munmap(host_ptr, aligned_size);
  };

  return torch::from_blob(device_ptr, contiguous_cpu.sizes(),
                          contiguous_cpu.strides(), deleter,
                          contiguous_cpu.options().device(torch::kCUDA));
}