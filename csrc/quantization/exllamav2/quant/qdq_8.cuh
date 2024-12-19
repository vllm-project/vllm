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
#ifndef _qdq_8_cuh
#define _qdq_8_cuh

#include "qdq_util.cuh"

namespace vllm {
namespace exl2 {

__forceinline__ __device__ void shuffle_8bit_4(uint32_t* q, int stride) {}

__forceinline__ __device__ void dequant_8bit_8(const uint32_t q_0,
                                               const uint32_t q_1,
                                               half2 (&dq)[4], int stride) {
  half dqh[8];
  for (int i = 0; i < 4; i++) dqh[i] = dq_ns(exb(q_0, i * 8, 0xff), 128);
  for (int i = 0; i < 4; i++) dqh[i + 4] = dq_ns(exb(q_1, i * 8, 0xff), 128);

  for (int i = 0; i < 4; i++)
    dq[i] = __halves2half2(dqh[i * 2], dqh[i * 2 + 1]);
}

}  // namespace exl2
}  // namespace vllm

#endif