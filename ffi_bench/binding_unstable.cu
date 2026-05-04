// Baseline: classic TORCH_LIBRARY registration with at::Tensor (unboxed
// dispatch path). This is what most of vLLM's csrc still uses today.
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#include "scale_kernel.cuh"

namespace bench_unstable {

void scale(at::Tensor& out, at::Tensor const& in, double factor) {
  TORCH_CHECK(out.is_cuda() && in.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "float32 only");
  auto stream = c10::cuda::getCurrentCUDAStream(in.device().index()).stream();
  launch_scale_f32(out.data_ptr<float>(), in.data_ptr<float>(),
                   static_cast<float>(factor), in.numel(), stream);
}

}  // namespace bench_unstable

TORCH_LIBRARY(bench_unstable, m) {
  m.def("scale(Tensor(a!) out, Tensor src, float factor) -> ()");
}

TORCH_LIBRARY_IMPL(bench_unstable, CUDA, m) {
  m.impl("scale", &bench_unstable::scale);
}
