#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ipex.h>

#include "dispatch_utils.h"
#include "reduction_utils.dp.hpp"



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

// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
void rms_norm_kernel(
  scalar_t* __restrict__ out,             // [..., hidden_size]
  const scalar_t* __restrict__ input,     // [..., hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size,
  const sycl::nd_item<3> &item_ct1,
  float &s_variance) {

  float variance = 0.0f;

  for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
       idx += item_ct1.get_local_range(2)) {
    const float x = (float)input[item_ct1.get_group(2) * hidden_size + idx];
    variance += x * x;
  }
  
  variance = blockReduceSum<float>(variance, item_ct1);
  if (item_ct1.get_local_id(2) == 0) {
    s_variance = sycl::rsqrt(variance / hidden_size + epsilon);
  }
  
  //item_ct1.barrier();
  item_ct1.barrier(sycl::access::fence_space::local_space);

  for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
       idx += item_ct1.get_local_range(2)) {
    float x = (float)input[item_ct1.get_group(2) * hidden_size + idx];
    out[item_ct1.get_group(2) * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

} // namespace vllm

void rms_norm(
  torch::Tensor& out,      // [..., hidden_size]
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(std::min(hidden_size, 1024));
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  stream.submit([&](sycl::handler &cgh){
  
  cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
   [=](sycl::nd_item<3> item_ct1){
   vllm::rms_norm_kernel<scalar_t>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size,
        item_ct1);
   });
    }
  );
  
  /*
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "rms_norm_kernel",
    [&] {
      vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size);
    });
    */
}
