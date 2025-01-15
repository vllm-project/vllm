/*
 * Adapted from https://github.com/turboderp/exllamav2
 * Copyright (c) 2024 turboderp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef _qdq_util_cuh
#define _qdq_util_cuh

namespace vllm {
namespace exl2 {

union half2_uint32 {
  uint32_t as_uint32;
  half2 as_half2;
  __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
  __device__ half2_uint32(half2 val) : as_half2(val) {}
  __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16 {
  uint16_t as_uint16;
  half as_half;
  __device__ half_uint16(uint16_t val) : as_uint16(val) {}
  __device__ half_uint16(half val) : as_half(val) {}
  __device__ half_uint16() : as_uint16(0) {}
};

// Max_scale premultiplied by 1/256

__forceinline__ __device__ half dq_scale(const int qs, const half max_scale) {
  int qs_i = qs + 1;
  half qs_h = __int2half_rn(qs_i * qs_i);
  qs_h = __hmul(qs_h, max_scale);
  return qs_h;
}

__forceinline__ __device__ half dq(const int q, const int qzero,
                                   const half scale) {
  return __hmul(__int2half_rn(q - qzero), scale);
}

__forceinline__ __device__ half dq_ns(const int q, const int qzero) {
  // return __hsub(__int2half_rn(q), __int2half_rn(qzero));
  return __int2half_rn(q - qzero);
}

__forceinline__ __device__ int exb(const uint32_t q, const int shift,
                                   const int mask) {
  return (int)((q >> shift) & mask);
}

__forceinline__ __device__ int exb(const uint32_t q1, const uint32_t q0,
                                   const int shift, const int mask) {
  return (int)(__funnelshift_rc(q0, q1, shift) & mask);
}

}  // namespace exl2
}  // namespace vllm
#endif