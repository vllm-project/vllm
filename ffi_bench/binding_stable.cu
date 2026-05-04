// Variant 2: STABLE_TORCH_LIBRARY (PyTorch >= 2.10 stable ABI). Goes through
// the boxed dispatcher path via TORCH_BOX. This is the path vLLM is migrating
// to in csrc/libtorch_stable/.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include "scale_kernel.cuh"

static inline cudaStream_t stable_current_stream(int32_t dev) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(dev, &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

namespace bench_stable {

void scale(torch::stable::Tensor& out, torch::stable::Tensor const& in,
           double factor) {
  STD_TORCH_CHECK(
      out.scalar_type() == torch::headeronly::ScalarType::Float &&
          in.scalar_type() == torch::headeronly::ScalarType::Float,
      "float32 only");
  const int32_t dev = in.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(dev);
  cudaStream_t stream = stable_current_stream(dev);
  launch_scale_f32(static_cast<float*>(out.mutable_data_ptr()),
                   static_cast<const float*>(in.const_data_ptr()),
                   static_cast<float>(factor), in.numel(), stream);
}

}  // namespace bench_stable

STABLE_TORCH_LIBRARY(bench_stable, m) {
  m.def("scale(Tensor(a!) out, Tensor src, float factor) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(bench_stable, CUDA, m) {
  m.impl("scale", TORCH_BOX(&bench_stable::scale));
}
