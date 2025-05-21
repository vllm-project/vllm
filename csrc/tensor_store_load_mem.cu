#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Template-based CUDA kernel: Copy from device memory to pinned host memory
template <typename scalar_t>
__global__ void store_kernel(const scalar_t* device_ptr, scalar_t* host_ptr,
                             size_t num_elements) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    host_ptr[idx] = device_ptr[idx];
  }
}

// Templated CUDA kernel: Copy from pinned host memory to device memory
template <typename scalar_t>
__global__ void load_kernel(const scalar_t* host_ptr, scalar_t* device_ptr,
                            size_t num_elements) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    device_ptr[idx] = host_ptr[idx];
  }
}

// Templated wrapper function: Store Tensor to pinned memory
template <typename scalar_t>
void store_tensor_impl(torch::Tensor& device_tensor, torch::Tensor& host_tensor) {
  const auto num_elements = device_tensor.numel();
  const int threads = 256;
  const int blocks = (num_elements + threads - 1) / threads;

  auto device_ptr = device_tensor.data_ptr<scalar_t>();
  auto host_ptr = host_tensor.data_ptr<scalar_t>();

  store_kernel<scalar_t>
      <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          device_ptr, host_ptr, num_elements);
}

// Templated wrapper function: Load Tensor from pinned memory
template <typename scalar_t>
void load_tensor_impl(torch::Tensor& host_tensor, torch::Tensor& device_tensor) {
  const auto num_elements = host_tensor.numel();
  const int threads = 256;
  const int blocks = (num_elements + threads - 1) / threads;

  auto host_ptr = host_tensor.data_ptr<scalar_t>();
  auto device_ptr = device_tensor.data_ptr<scalar_t>();

  load_kernel<scalar_t>
      <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          host_ptr, device_ptr, num_elements);
}

// Type-dispatched wrapper function
void store_tensor(torch::Tensor& device_tensor, torch::Tensor& host_tensor) {
  // Validate arguments
  AT_ASSERT(device_tensor.is_cuda(), "Input tensor must be a CUDA tensor");
  AT_ASSERT(host_tensor.is_pinned(), "Output tensor must be pinned memory");
  AT_ASSERT(device_tensor.numel() == host_tensor.numel(),
            "Tensors must have same number of elements");
  AT_ASSERT(device_tensor.dtype() == host_tensor.dtype(),
            "Tensors must have same dtype");

  // Type-based dispatch to different implementations
  switch (device_tensor.scalar_type()) {
    case torch::kFloat:
      store_tensor_impl<float>(device_tensor, host_tensor);
      break;
    case torch::kHalf:
      store_tensor_impl<at::Half>(device_tensor, host_tensor);
      break;
    case torch::kBFloat16:
      store_tensor_impl<at::BFloat16>(device_tensor, host_tensor);
      break;
    default:
      AT_ERROR("Unsupported data type: ", device_tensor.scalar_type());
  }
}

void load_tensor(torch::Tensor& host_tensor, torch::Tensor& device_tensor) {
  // Validate arguments
  AT_ASSERT(device_tensor.is_cuda(), "Output tensor must be a CUDA tensor");
  AT_ASSERT(host_tensor.is_pinned(), "Input tensor must be pinned memory");
  AT_ASSERT(device_tensor.numel() == host_tensor.numel(),
            "Tensors must have same number of elements");
  AT_ASSERT(device_tensor.dtype() == host_tensor.dtype(),
            "Tensors must have same dtype");

  // Type-based dispatch to different implementations
  switch (host_tensor.scalar_type()) {
    case torch::kFloat:
      load_tensor_impl<float>(host_tensor, device_tensor);
      break;
    case torch::kHalf:
      load_tensor_impl<at::Half>(host_tensor, device_tensor);
      break;
    case torch::kBFloat16:
      load_tensor_impl<at::BFloat16>(host_tensor, device_tensor);
      break;
    default:
      AT_ERROR("Unsupported data type: ", host_tensor.scalar_type());
  }
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("store_tensor", &store_tensor,
//           "Store CUDA tensor to pinned memory
//           (supports float32, float16, bfloat16)");
//     m.def("load_tensor", &load_tensor,
//           "Load CUDA tensor from pinned memory
//           (supports float32, float16, bfloat16)");
// }
