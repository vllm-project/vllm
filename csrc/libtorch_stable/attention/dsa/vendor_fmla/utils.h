#pragma once

#include <cstdint>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(1);                                                        \
    }                                                                 \
  } while (0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

#define FLASH_ASSERT(cond)                                                  \
  do {                                                                      \
    if (not(cond)) {                                                        \
      fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, \
              #cond);                                                       \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

#define FLASH_DEVICE_ASSERT(cond)                                          \
  do {                                                                     \
    if (not(cond)) {                                                       \
      printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
      asm("trap;");                                                        \
    }                                                                      \
  } while (0)

#define println(fmt, ...)      \
  {                            \
    print(fmt, ##__VA_ARGS__); \
    print("\n");               \
  }

template <typename T>
__inline__ __host__ __device__ T ceil_div(const T& a, const T& b) {
  return (a + b - 1) / b;
}

#ifndef TRAP_ONLY_DEVICE_ASSERT
  #define TRAP_ONLY_DEVICE_ASSERT(cond) \
    do {                                \
      if (not(cond)) asm("trap;");      \
    } while (0)
#endif

#ifndef TRAP_ONLY_DEVICE_ASSERT
  #define TRAP_ONLY_DEVICE_ASSERT(cond) \
    do {                                \
      if (not(cond)) asm("trap;");      \
    } while (0)
#endif

struct RingBufferState {
  uint32_t cur_block_idx = 0u;

  __device__ __forceinline__ void update() { cur_block_idx += 1; }

  template <uint32_t NUM_STAGES>
  __device__ __forceinline__ std::pair<uint32_t, bool> get() const {
    uint32_t stage_idx = cur_block_idx % NUM_STAGES;
    bool phase = (cur_block_idx / NUM_STAGES) & 1;
    return {stage_idx, phase};
  }

  __device__ __forceinline__ RingBufferState offset_by(const int offset) const {
    // Must guarantee no underflow
    uint32_t new_block_idx =
        static_cast<uint32_t>(static_cast<int>(cur_block_idx) + offset);
    RingBufferState new_state;
    new_state.cur_block_idx = new_block_idx;
    return new_state;
  }
};
