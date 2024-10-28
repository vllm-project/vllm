/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Reference implementation for GEMM in host-side code.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

namespace cutlass {
namespace reference {
namespace detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Template function to compute an inner product.
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate with a
                            // host-only type
template <typename Atype, typename Btype, typename Ctype>
CUTLASS_HOST_DEVICE
Ctype inner_product(Atype a, Btype b, Ctype c) {
  return Ctype(a) * Ctype(b) + c;
}

/// Specialization for matrix multiplication with binary operands
template <>
CUTLASS_HOST_DEVICE
int inner_product<Array<bin1_t, 32>, Array<bin1_t, 32>, int>(
    Array<bin1_t, 32> a,
    Array<bin1_t, 32> b,
    int c) {

  int accum = 0;
  for (int bit = 0; bit < 32; bit++) {
    accum += a[bit] ^ b[bit];
  }
  return accum + c;
}

/*
/// Specialization for matrix multiplication with signed 4-bit integer operands
template <>
CUTLASS_HOST_DEVICE
int inner_product<Array<int4b_t, 8>, Array<int4b_t, 8>, int>(
    Array<int4b_t, 8> a,
    Array<int4b_t, 8> b,
    int c) {

  int accum = 0;
  for (int k = 0; k < 8; k++) {
    accum += a[k] * b[k];
  }
  return accum + c;
}

/// Specialization for matrix multiplication with unsigned 4-bit integer operands
template <>
CUTLASS_HOST_DEVICE
int inner_product<Array<uint4b_t, 8>, Array<uint4b_t, 8>, int>(
    Array<uint4b_t, 8> a,
    Array<uint4b_t, 8> b,
    int c) {

  int accum = 0;
  for (int k = 0; k < 8; k++) {
    accum += a[k] * b[k];
  }
  return accum + c;
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType>
struct Cast {
  // Default behavior: convert to the destination type
#pragma hd_warning_disable  // Suppresses warnings when attempting to instantiate complex<T> with a
                            // host-only type
  CUTLASS_HOST_DEVICE
  static DstType apply(SrcType src) { return static_cast<DstType>(src); };
};

template <>
struct Cast<float, int8_t> {
  CUTLASS_HOST_DEVICE
  static int8_t apply(float src) {
    // Clamp to the range of signed 8-bit integers.
    return static_cast<int8_t>(fmaxf(-128.f, fminf(127.f, src)));
  };
};

template <>
struct Cast<float, uint8_t> {
  CUTLASS_HOST_DEVICE
  static uint8_t apply(float src) {
    // Clamp to the range of signed 8-bit integers.
    return static_cast<uint8_t>(fmaxf(0.f, fminf(255.f, src)));
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail
} // namespace reference
} // namespace cutlass

