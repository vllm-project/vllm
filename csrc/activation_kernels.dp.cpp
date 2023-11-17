#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ipex.h>
#include "dispatch_utils.h"

namespace at{

namespace cuda{

dpct::queue_ptr getCurrentCUDAStream(){

  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto& queue = xpu::get_queue_from_stream(c10_stream);
  return &queue;

}

}

}

namespace vllm {

template <typename T> __dpct_inline__ T silu(const T &x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + sycl::exp((float)-x)));
}

template<typename scalar_t>
void silu_and_mul_kernel(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., 2, d]
  const int d,
  const sycl::nd_item<3> &item_ct1) {
  const int64_t token_idx = item_ct1.get_group(2);
  for (int64_t idx = item_ct1.get_local_id(2); idx < d;
       idx += item_ct1.get_local_range(2)) {
    const scalar_t x = __ldg(&input[token_idx * 2 * d + idx]);
    const scalar_t y = __ldg(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = silu(x) * y;
  }
}

} // namespace vllm

void silu_and_mul(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input)    // [..., 2 * d]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;

  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(std::min(d, 1024));
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  stream.submit([&](sycl::handler &cgh){
  
  cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
   [=](sycl::nd_item<3> item_ct1){
   vllm::silu_and_mul_kernel<scalar_t>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),d, item_ct1);
   });
  
  });
  
  /*
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "silu_and_mul_kernel",
    [&] {
    
      
      vllm::silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        d);
    });
    */
}

namespace vllm {

// Element-wise activation kernel template.
template<typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
void activation_kernel(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., d]
  const int d,
  const sycl::nd_item<3> &item_ct1) {
  const int64_t token_idx = item_ct1.get_group(2);
  for (int64_t idx = item_ct1.get_local_id(2); idx < d;
       idx += item_ct1.get_local_range(2)) {
    const scalar_t x = __ldg(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = ACT_FN(x);
  }
}

} // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                       \
  int d = input.size(-1);                                                      \
  int64_t num_tokens = input.numel() / d;                                      \
  sycl::range<3> grid(1, 1, num_tokens);                                       \
  sycl::range<3> block(std::min(d, 1024));                                     \
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();             \
  stream.submit([&](sycl::handler &cgh){                                       \
  cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                     \
   [=](sycl::nd_item<3> item_ct1){                                             \
   vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>(out.data_ptr<scalar_t>(),\
   input.data_ptr<scalar_t>(),d, item_ct1);                                     \
   });                                                                          \
   });                                                                          \                                                                          
  /*
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activation_kernel", [&] { \
    vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>                        \
        <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),                 \
                                     input.data_ptr<scalar_t>(), d);           \
  });
  */

namespace vllm {

template <typename T> __dpct_inline__ T gelu_new_kernel(const T &x) {
  const float x3 = (float) (x * x * x);
  const T t = (T) tanhf((T) (0.79788456f * (float) (x + (T) (0.044715f * x3))));
  return ((T) 0.5) * x * (((T) 1.0) + t);
}

template <typename T> __dpct_inline__ T gelu_fast_kernel(const T &x) {
  const float f = (float) x;
  const T t = (T) tanhf(((T) (f * 0.79788456f)) * (((T) 1.0) + (T) (0.044715f * f) * x));
  return ((T) 0.5) * x * (((T) 1.0) + t);
}

} // namespace vllm

void gelu_new(
  torch::Tensor& out,     // [..., d]
  torch::Tensor& input)   // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(
  torch::Tensor& out,     // [..., d]
  torch::Tensor& input)   // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}
