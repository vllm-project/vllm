#include <mscclpp/proxy_channel_device.hpp>

extern "C" __global__ void __launch_bounds__(1024, 1)
    nw_cache_in_kernel(mscclpp::ProxyChannelDeviceHandle* proxyChannel) {
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIndex == 0) {
    proxyChannel[0].wait(100000000);
  }
}

extern "C" __global__ void __launch_bounds__(1024, 1)
    nw_cache_out_kernel(mscclpp::ProxyChannelDeviceHandle* proxyChannel, int dst_mem, int src_mem, int kv_block_offset, int dataSize, int flush) {
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIndex == 0) {
    proxyChannel[0].put(dst_mem, kv_block_offset, src_mem, kv_block_offset, dataSize);
    if (flush) {
      proxyChannel[0].flush();
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024, 1)
    nw_cache_out_signal_kernel(mscclpp::ProxyChannelDeviceHandle* proxyChannel) {
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIndex == 0) {
    proxyChannel[0].signal();
    proxyChannel[0].flush();
  }
}
