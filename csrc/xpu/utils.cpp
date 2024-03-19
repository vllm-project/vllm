#include "utils.h"
#include <sycl/ext/intel/math.hpp>

sycl::half sycl_half_mul(sycl::half a, sycl::half b) {
  return sycl::ext::intel::math::hmul(a, b);
}
sycl::half sycl_half_add(sycl::half a, sycl::half b) {
  return sycl::ext::intel::math::hadd(a, b);
}
sycl::half sycl_half_sub(sycl::half a, sycl::half b) {
  return sycl::ext::intel::math::hsub(a, b);
}
sycl::half sycl_half_fma(sycl::half a, sycl::half b, sycl::half c) {
  return sycl::ext::intel::math::hfma(a, b, c);
}

sycl::half2 sycl_half_mul2(sycl::half2 a, sycl::half2 b) {
  return sycl::ext::intel::math::hmul2(a, b);
}
sycl::half2 sycl_half_add2(sycl::half2 a, sycl::half2 b) {
  return sycl::ext::intel::math::hadd2(a, b);
}
sycl::half2 sycl_half_sub2(sycl::half2 a, sycl::half2 b) {
  return sycl::ext::intel::math::hsub2(a, b);
}

sycl::half2 sycl_half_fma2(sycl::half2 a, sycl::half2 b, sycl::half2 c) {
  return sycl::ext::intel::math::hfma2(a, b, c);
}

int get_max_shared_memory_per_block_device_attribute(int device_id) {
  const sycl::device& device = vllm::xpu::vllmGetQueue().get_device();
  return device.get_info<sycl::info::device::local_mem_size>();
}
