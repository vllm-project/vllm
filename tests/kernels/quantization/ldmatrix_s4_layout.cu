// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if CUDART_VERSION < 13040
  #error "ldmatrix.s8.s4 requires CUDA 13.4 or newer"
#endif

namespace {

constexpr int kWarpSize = 32;
constexpr int kMatrixRows = 8;
constexpr int kMatrixColumns = 16;
constexpr int kPackedRowBytes = kMatrixColumns / 2;
constexpr int kMaxMatrices = 4;

#define CUDA_CHECK(call)                             \
  do {                                               \
    cudaError_t error = call;                        \
    if (error != cudaSuccess) {                      \
      std::fprintf(stderr, "%s failed: %s\n", #call, \
                   cudaGetErrorString(error));       \
      std::exit(EXIT_FAILURE);                       \
    }                                                \
  } while (0)

template <int NumMatrices>
__device__ inline void load_s4(uint32_t (&registers)[NumMatrices],
                               const void* shared_ptr) {
  uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr));
  if constexpr (NumMatrices == 1) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x1.shared::cta.s8.s4 {%0}, [%1];\n"
        : "=r"(registers[0])
        : "r"(address));
  } else if constexpr (NumMatrices == 2) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x2.shared::cta.s8.s4 {%0,%1}, [%2];\n"
        : "=r"(registers[0]), "=r"(registers[1])
        : "r"(address));
  } else {
    static_assert(NumMatrices == 4);
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x4.shared::cta.s8.s4 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]),
          "=r"(registers[3])
        : "r"(address));
  }
}

template <int NumMatrices>
__global__ void load_layout_kernel(const uint8_t* packed, uint32_t* output) {
  __shared__ __align__(16)
      uint8_t shared[NumMatrices][kMatrixRows][kPackedRowBytes];

  int lane = threadIdx.x;
  for (int index = lane; index < NumMatrices * kMatrixRows * kPackedRowBytes;
       index += kWarpSize) {
    reinterpret_cast<uint8_t*>(shared)[index] = packed[index];
  }
  __syncthreads();

  int address_lane = lane < NumMatrices * kMatrixRows ? lane : 0;
  int matrix = address_lane / kMatrixRows;
  int row = address_lane % kMatrixRows;

  uint32_t registers[NumMatrices];
  load_s4(registers, &shared[matrix][row][0]);

#pragma unroll
  for (int i = 0; i < NumMatrices; ++i) {
    output[lane * kMaxMatrices + i] = registers[i];
  }
}

__host__ __device__ int8_t a_value(int row, int column) {
  return static_cast<int8_t>((row * 3 + column * 5) % 7 - 3);
}

__host__ __device__ int8_t b_value(int row, int column) {
  return static_cast<int8_t>((row * 7 + column * 3) % 16 - 8);
}

__global__ void mma_layout_kernel(const uint8_t* packed_b, int32_t* output) {
  __shared__ __align__(16) uint8_t shared_b[2][kMatrixRows][kPackedRowBytes];

  int lane = threadIdx.x;
  for (int index = lane; index < 2 * kMatrixRows * kPackedRowBytes;
       index += kWarpSize) {
    reinterpret_cast<uint8_t*>(shared_b)[index] = packed_b[index];
  }
  __syncthreads();

  int address_lane = lane < 2 * kMatrixRows ? lane : 0;
  int matrix = address_lane / kMatrixRows;
  int row = address_lane % kMatrixRows;

  uint32_t b[2];
  load_s4(b, &shared_b[matrix][row][0]);

  int group = lane / 4;
  int thread_in_group = lane % 4;
  uint32_t a[4] = {};
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    int a_row = (i < 4 || (i >= 8 && i < 12)) ? group : group + 8;
    int a_column = thread_in_group * 4 + (i & 3) + (i >= 8 ? 16 : 0);
    a[i / 4] |=
        static_cast<uint32_t>(static_cast<uint8_t>(a_value(a_row, a_column)))
        << (8 * (i & 3));
  }

  int32_t c[4] = {};
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    output[lane * 4 + i] = c[i];
  }
}

uint8_t get_nibble(const std::vector<uint8_t>& values, int matrix, int row,
                   int column) {
  int index = (matrix * kMatrixRows + row) * kMatrixColumns + column;
  return values[index];
}

std::vector<uint8_t> pack_values(const std::vector<uint8_t>& values,
                                 int num_matrices) {
  std::vector<uint8_t> packed(num_matrices * kMatrixRows * kPackedRowBytes);
  for (int matrix = 0; matrix < num_matrices; ++matrix) {
    for (int row = 0; row < kMatrixRows; ++row) {
      for (int column = 0; column < kMatrixColumns; column += 2) {
        uint8_t low = get_nibble(values, matrix, row, column) ^ 0x8;
        uint8_t high = get_nibble(values, matrix, row, column + 1) ^ 0x8;
        int index = (matrix * kMatrixRows + row) * kPackedRowBytes + column / 2;
        packed[index] = low | (high << 4);
      }
    }
  }
  return packed;
}

template <int NumMatrices>
std::vector<uint32_t> run_layout(const std::vector<uint8_t>& values) {
  std::vector<uint8_t> packed = pack_values(values, NumMatrices);
  std::vector<uint32_t> output(kWarpSize * kMaxMatrices);

  uint8_t* device_packed;
  uint32_t* device_output;
  CUDA_CHECK(cudaMalloc(&device_packed, packed.size()));
  CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(device_packed, packed.data(), packed.size(),
                        cudaMemcpyHostToDevice));

  load_layout_kernel<NumMatrices>
      <<<1, kWarpSize>>>(device_packed, device_output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output,
                        output.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_packed));
  return output;
}

template <int NumMatrices>
bool validate_layout(const std::vector<uint8_t>& values) {
  std::vector<uint32_t> output = run_layout<NumMatrices>(values);
  for (int matrix = 0; matrix < NumMatrices; ++matrix) {
    for (int row = 0; row < kMatrixRows; ++row) {
      for (int column = 0; column < kMatrixColumns; ++column) {
        int lane = row * 4 + column / 4;
        int byte = column % 4;
        int8_t actual = static_cast<int8_t>(
            output[lane * kMaxMatrices + matrix] >> (8 * byte));
        int8_t expected =
            static_cast<int8_t>(get_nibble(values, matrix, row, column) - 8);
        if (actual != expected) {
          std::fprintf(
              stderr,
              "x%d mismatch: matrix=%d row=%d column=%d lane=%d byte=%d "
              "expected=%d actual=%d\n",
              NumMatrices, matrix, row, column, lane, byte, expected, actual);
          return false;
        }
      }
    }
  }
  return true;
}

template <int NumMatrices>
bool validate_pattern() {
  std::vector<uint8_t> values(NumMatrices * kMatrixRows * kMatrixColumns);
  for (int matrix = 0; matrix < NumMatrices; ++matrix) {
    for (int row = 0; row < kMatrixRows; ++row) {
      for (int column = 0; column < kMatrixColumns; ++column) {
        int index = (matrix * kMatrixRows + row) * kMatrixColumns + column;
        values[index] =
            static_cast<uint8_t>((column + row * 3 + matrix * 5) & 0xF);
      }
    }
  }
  return validate_layout<NumMatrices>(values);
}

template <int NumMatrices>
bool validate_every_source_element() {
  constexpr int num_values = NumMatrices * kMatrixRows * kMatrixColumns;
  std::vector<uint8_t> values(num_values, 8);
  for (int index = 0; index < num_values; ++index) {
    values[index] = 15;
    if (!validate_layout<NumMatrices>(values)) {
      std::fprintf(stderr, "source element %d failed\n", index);
      return false;
    }
    values[index] = 8;
  }
  return true;
}

bool validate_mma() {
  std::vector<uint8_t> packed(2 * kMatrixRows * kPackedRowBytes);
  for (int matrix = 0; matrix < 2; ++matrix) {
    for (int row = 0; row < kMatrixRows; ++row) {
      for (int column = 0; column < kMatrixColumns; column += 2) {
        int k0 = matrix * kMatrixColumns + column;
        int k1 = k0 + 1;
        uint8_t low = static_cast<uint8_t>(b_value(k0, row)) & 0xF;
        uint8_t high = static_cast<uint8_t>(b_value(k1, row)) & 0xF;
        int index = (matrix * kMatrixRows + row) * kPackedRowBytes + column / 2;
        packed[index] = low | (high << 4);
      }
    }
  }

  std::vector<int32_t> fragments(kWarpSize * 4);
  uint8_t* device_packed;
  int32_t* device_fragments;
  CUDA_CHECK(cudaMalloc(&device_packed, packed.size()));
  CUDA_CHECK(cudaMalloc(&device_fragments, fragments.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(device_packed, packed.data(), packed.size(),
                        cudaMemcpyHostToDevice));

  mma_layout_kernel<<<1, kWarpSize>>>(device_packed, device_fragments);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(fragments.data(), device_fragments,
                        fragments.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_fragments));
  CUDA_CHECK(cudaFree(device_packed));

  for (int lane = 0; lane < kWarpSize; ++lane) {
    int group = lane / 4;
    int thread_in_group = lane % 4;
    for (int i = 0; i < 4; ++i) {
      int row = i < 2 ? group : group + 8;
      int column = thread_in_group * 2 + (i & 1);
      int32_t expected = 0;
      for (int k = 0; k < 32; ++k) {
        expected += static_cast<int32_t>(a_value(row, k)) *
                    static_cast<int32_t>(b_value(k, column));
      }
      int32_t actual = fragments[lane * 4 + i];
      if (actual != expected) {
        std::fprintf(stderr,
                     "mma mismatch: row=%d column=%d lane=%d register=%d "
                     "expected=%d actual=%d\n",
                     row, column, lane, i, expected, actual);
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  bool supported = (properties.major == 9 && properties.minor == 0) ||
                   (properties.major == 10 &&
                    (properties.minor == 0 || properties.minor == 3 ||
                     properties.minor == 7)) ||
                   (properties.major == 11 && properties.minor == 0) ||
                   (properties.major == 12 &&
                    (properties.minor == 0 || properties.minor == 1));
  if (!supported) {
    std::fprintf(stderr, "unsupported compute capability %d.%d\n",
                 properties.major, properties.minor);
    return EXIT_FAILURE;
  }

  if (!validate_pattern<1>() || !validate_pattern<2>() ||
      !validate_pattern<4>() || !validate_every_source_element<1>() ||
      !validate_every_source_element<2>() ||
      !validate_every_source_element<4>() || !validate_mma()) {
    return EXIT_FAILURE;
  }

  std::printf("ldmatrix.s8.s4 layout and MMA operand-B mapping passed\n");
  return EXIT_SUCCESS;
}
