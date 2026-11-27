#pragma once
// Device mechanisms from native-fp32-boundary-loop/binding.cu.
namespace DOSA_NAMESPACE::native_fp32_boundary_map {
// Every physical row contributes at most one task; state and tasks are device
// resident. E occupies the first uncertain words of each contiguous backing
// row.
__global__ void tasks(const int* partition, int* error, int* out, int* state,
                      int rows, int stride) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || error[row]) return;
  const unsigned p = unsigned(partition[row]);
  if (!p) return;
  const int certain = p >> 13, uncertain = p & 8191;
  if (certain > 2048 || uncertain > 4096 || uncertain > stride ||
      uncertain < 2048 - certain) {
    atomicOr(error + row, 1);
    return;
  }
  if (certain == 2048) return;
  const int slot = atomicAdd(state, 1);
  out[slot * 3] = row;
  out[slot * 3 + 1] = row * stride;
  out[slot * 3 + 2] = uncertain;
}
}  // namespace DOSA_NAMESPACE::native_fp32_boundary_map
