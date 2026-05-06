#include "w4a8_utils.cuh"

#include <array>
#include <cuda_runtime.h>
#include <cstdio>

namespace vllm::cutlass_w4a8_utils {

/*
  GPU-accelerated implementation of cutlass::unified_encode_int4b.
  Constructs a lookup table in constant memory to map 8 bits
  (two 4-bit values) at a time. Assumes memory is contiguous
  and pointers are 16-byte aligned.
*/
__constant__ uint8_t kNibbleLUT[256];

__global__ void unified_encode_int4b_device(const uint8_t* in, uint8_t* out,
                                            size_t nbytes) {
  constexpr size_t V = sizeof(uint4);  // 16 bytes
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t nthreads = size_t(gridDim.x) * blockDim.x;
  const size_t nvec = nbytes / V;

  // 1-D grid-stride loop over 16-byte chunks
  for (size_t vec = tid; vec < nvec; vec += nthreads) {
    uint4 v = reinterpret_cast<const uint4*>(in)[vec];
    uint8_t* b = reinterpret_cast<uint8_t*>(&v);
#pragma unroll
    for (int i = 0; i < int(V); ++i) b[i] = kNibbleLUT[b[i]];
    reinterpret_cast<uint4*>(out)[vec] = v;
  }
}

static bool upload_lut() {
  std::array<uint8_t, 256> lut{};
  auto map_nib = [](uint8_t v) -> uint8_t {
    // 1..7 -> (8 - v); keep 0 and 8..15
    return (v == 0 || (v & 0x8)) ? v : uint8_t(8 - v);
  };
  for (int b = 0; b < 256; ++b) {
    uint8_t lo = b & 0xF;
    uint8_t hi = (b >> 4) & 0xF;
    lut[b] = uint8_t((map_nib(hi) << 4) | map_nib(lo));
  }
  cudaError_t e = cudaMemcpyToSymbol(kNibbleLUT, lut.data(), lut.size(),
                                     /*offset=*/0, cudaMemcpyHostToDevice);

  return (e == cudaSuccess);
}

bool unified_encode_int4b(cutlass::int4b_t const* in, cutlass::int4b_t* out,
                          size_t num_int4_elems) {
  // Build/upload LUT
  if (!upload_lut()) return false;

  static_assert(sizeof(typename cutlass::int4b_t::Storage) == 1,
                "int4 storage must be 1 byte");
  const size_t nbytes = num_int4_elems >> 1;

  auto* in_bytes = reinterpret_cast<uint8_t const*>(in);
  auto* out_bytes = reinterpret_cast<uint8_t*>(out);

  // kernel launch params
  constexpr int block = 256;
  const size_t nvec = nbytes / sizeof(uint4);  // # of 16B vectors
  int grid = int((nvec + block - 1) / block);
  if (grid == 0) grid = 1;  // ensure we still cover the tail in the kernel

  unified_encode_int4b_device<<<grid, block>>>(in_bytes, out_bytes, nbytes);

  // launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("unified_encode_int4b_device launch error: %s (%d)\n",
           cudaGetErrorString(err), err);
    return false;
  }

  // runtime errors
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("unified_encode_int4b_device runtime error: %s (%d)\n",
           cudaGetErrorString(err), err);
    return false;
  }

  return true;
}

}  // namespace vllm::cutlass_w4a8_utils