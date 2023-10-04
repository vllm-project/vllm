#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include "ATen/core/TensorBody.h"

void vllm_sort_cuda(const float *src, float *dst, int batch_size, int len,
                    bool desending);

void vllm_sort(const torch::Tensor src, torch::Tensor dst, bool desending) {
  float *src_ptr = reinterpret_cast<float *>(src.data_ptr());
  float *dst_ptr = reinterpret_cast<float *>(dst.data_ptr());
  auto shape = src.sizes();
  std::cout << "shape: " << shape << std::endl;
  auto shape_len = shape.size();
  assert(shape_len == 2);
  unsigned int batch_size = src.sizes()[0];
  unsigned int len = shape[shape_len - 1];
  vllm_sort_cuda(src_ptr, dst_ptr, batch_size, len, desending);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vllm_sort", &vllm_sort, "Applty vllm sort");
}
