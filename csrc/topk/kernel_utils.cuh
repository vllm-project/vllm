#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <type_traits>
static const float HALF_FLT_MAX = 65504.F;
#define FINAL_MASK 0xffffffff

#define checkErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
struct TopK {
  int p = -1;
  T u = -((std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX);

  __device__ __forceinline__ void insert(T elem, int elem_id) {
    if (elem > u) {
      u = elem;
      p = elem_id;
    }
  }

  __device__ __forceinline__ void init() {
    u = -((std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX);
    p = -1;
  }
};

template <typename T>
__device__ __forceinline__ TopK<T> reduce_topk_op(const TopK<T> &a,
                                                  const TopK<T> &b) {
  return a.u > b.u ? a : b;
}

template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}
