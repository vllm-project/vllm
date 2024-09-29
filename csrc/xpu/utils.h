#pragma once

#include <sycl/sycl.hpp>
#include <memory>
#include <ipex.h>
#include <ATen/ATen.h>

#define VLLM_LDG(arg) *(arg)
namespace vllm {
namespace xpu {

static inline sycl::queue& vllmGetQueue() {
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto& queue = ::xpu::get_queue_from_stream(c10_stream);
  return queue;
}
template <typename T>
struct SyclTypeTrait{
  using Type = T;
};

template <>
struct SyclTypeTrait<c10::Half>{
  using Type = sycl::half;
};

template <>
struct SyclTypeTrait<c10::BFloat16>{
  using Type = sycl::ext::oneapi::bfloat16;
};


} // namespace xpu

} // namespace vllm

SYCL_EXTERNAL sycl::half sycl_half_mul(sycl::half a, sycl::half b);
SYCL_EXTERNAL sycl::half sycl_half_add(sycl::half a, sycl::half b);
SYCL_EXTERNAL sycl::half sycl_half_sub(sycl::half a, sycl::half b);
SYCL_EXTERNAL sycl::half sycl_half_fma(sycl::half a, sycl::half b, sycl::half c);

SYCL_EXTERNAL sycl::half2 sycl_half_mul2(sycl::half2 a, sycl::half2 b);
SYCL_EXTERNAL sycl::half2 sycl_half_add2(sycl::half2 a, sycl::half2 b);
SYCL_EXTERNAL sycl::half2 sycl_half_sub2(sycl::half2 a, sycl::half2 b);
SYCL_EXTERNAL sycl::half2 sycl_half_fma2(sycl::half2 a, sycl::half2 b, sycl::half2 c);

int get_max_shared_memory_per_block_device_attribute(int device_id);
