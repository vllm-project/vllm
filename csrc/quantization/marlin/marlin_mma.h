
#include "marlin_dtypes.cuh"

namespace MARLIN_NAMESPACE_NAME {

// m16n8k16 tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
template <vllm::ScalarTypeId type_id, bool use_fp16_accum, int k_size = 16>
__device__ inline void mma(
    const typename MarlinScalarType<type_id>::FragA& a_frag,
    const typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::FragC& frag_c, int idx = 0) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  if constexpr (!std::is_same<scalar_t, half>::value || k_size != 16) {
    static_assert(!use_fp16_accum);
  }

  if constexpr (k_size == 16) {
    if constexpr (std::is_same<scalar_t, half>::value && !use_fp16_accum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(b[0]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[2]), "r"(a[3]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));
#else
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
    } else if constexpr (std::is_same<scalar_t, half>::value &&
                         use_fp16_accum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[2]), "r"(a[3]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
#else
      uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
            "r"(c[0]), "r"(c[1]));
#endif
    } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    } else if constexpr (std::is_same<scalar_t, __nv_fp8_e4m3>::value) {
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[idx * 2]), "r"(a[idx * 2 + 1]), "r"(b[idx]), "f"(c[0]),
            "f"(c[1]), "f"(c[2]), "f"(c[3]));
    } else if constexpr (std::is_same<scalar_t, int8_t>::value) {
      int32_t* c = reinterpret_cast<int32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
          : "r"(a[idx * 2]), "r"(a[idx * 2 + 1]), "r"(b[idx]), "r"(c[0]),
            "r"(c[1]), "r"(c[2]), "r"(c[3]));
    }
  } else if (k_size == 32) {
    if constexpr (std::is_same<scalar_t, __nv_fp8_e4m3>::value) {
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    } else if constexpr (std::is_same<scalar_t, int8_t>::value) {
      int32_t* c = reinterpret_cast<int32_t*>(&frag_c);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[0]), "r"(b[0]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[2]), "=r"(c[3])
          : "r"(a[1]), "r"(b[0]), "r"(c[2]), "r"(c[3]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(a[2]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[2]), "=r"(c[3])
          : "r"(a[3]), "r"(b[1]), "r"(c[2]), "r"(c[3]));
#else
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
            "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#endif
    }
  }
}

template <vllm::ScalarTypeId type_id, bool use_fp16_accum, int k_size = 16>
__device__ inline void mma_trans(
    const typename MarlinScalarType<type_id>::FragA& a_frag,
    const typename MarlinScalarType<type_id>::FragB& frag_b,
    const typename MarlinScalarType<type_id>::FragB& frag_b2,
    typename MarlinScalarType<type_id>::FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  const uint32_t* b2 = reinterpret_cast<const uint32_t*>(&frag_b2);
  float* c = reinterpret_cast<float*>(&frag_c);
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  if constexpr (!std::is_same<scalar_t, half>::value || k_size != 16) {
    static_assert(!use_fp16_accum);
  }

  if constexpr (k_size == 16) {
    if constexpr (std::is_same<scalar_t, half>::value && !use_fp16_accum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(a[0]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[1]), "r"(b2[1]), "r"(a[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));
#else
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
    } else if constexpr (std::is_same<scalar_t, half>::value &&
                         use_fp16_accum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[0]), "r"(b2[0]), "r"(a[0]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[1]), "r"(b2[1]), "r"(a[1]), "r"(c[0]), "r"(c[1]));
#else
      uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
          "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
            "r"(c[0]), "r"(c[1]));
#endif
    } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    } else if constexpr (std::is_same<scalar_t, __nv_fp8_e4m3>::value) {
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(a[0]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
            "f"(c[3]));
    } else if constexpr (std::is_same<scalar_t, int8_t>::value) {
      int32_t* c = reinterpret_cast<int32_t*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
          : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(a[0]), "r"(c[0]), "r"(c[1]), "r"(c[2]),
            "r"(c[3]));
    }
  } else {
    if constexpr (std::is_same<scalar_t, __nv_fp8_e4m3>::value) {
      float* c = reinterpret_cast<float*>(&frag_c);
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
            "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    } else if constexpr (std::is_same<scalar_t, int8_t>::value) {
      int32_t* c = reinterpret_cast<int32_t*>(&frag_c);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[0]), "r"(a[0]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[2]), "=r"(c[3])
          : "r"(b2[1]), "r"(a[0]), "r"(c[2]), "r"(c[3]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[0]), "=r"(c[1])
          : "r"(b[0]), "r"(a[1]), "r"(c[0]), "r"(c[1]));
      asm volatile(
          "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
          : "=r"(c[2]), "=r"(c[3])
          : "r"(b2[1]), "r"(a[1]), "r"(c[2]), "r"(c[3]));
#else
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
          : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
          : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
            "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#endif
    }
  }
}

}  // namespace MARLIN_NAMESPACE_NAME