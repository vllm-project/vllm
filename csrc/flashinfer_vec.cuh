#ifndef FLASHINFER_VEC_MINIMAL_CUH_
#define FLASHINFER_VEC_MINIMAL_CUH_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

namespace vllm {

// Generic vec_t template (for float32, half, bfloat16)
template <typename T, size_t N>
struct vec_t {
  T data[N];

  FLASHINFER_INLINE T& operator[](size_t i) { return data[i]; }
  FLASHINFER_INLINE const T& operator[](size_t i) const { return data[i]; }

  FLASHINFER_INLINE void load(const T* ptr) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      data[i] = ptr[i];
    }
  }

  FLASHINFER_INLINE void store(T* ptr) const {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      ptr[i] = data[i];
    }
  }

  template <typename U>
  FLASHINFER_INLINE void cast_load(const U* ptr) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      data[i] = static_cast<T>(ptr[i]);
    }
  }

  FLASHINFER_INLINE T* ptr() { return data; }
};

}  // namespace vllm

#undef FLASHINFER_INLINE

#endif  // FLASHINFER_VEC_MINIMAL_CUH_
