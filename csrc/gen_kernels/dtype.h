/***************************************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are not permit- ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cuda_utils.h"

enum class Dtype : uint32_t {

// We use the following encoding for the types:
//
// Byte 0: Identifier for the type (going from 0 to the number of data types -
// 1, Byte 1: Number of bits in the type, Byte 2: Bit 0: Is it an integer? 0x1
// if true, 0x0 otherwise;
//         Bit 4: is it signed?  0x1 if true, 0x0 otherwise.
// Byte 3: Is it a block format? 0x1 if true, 0x0 otherwise.

#define ENCODE_DTYPE(BlockFormatBit, SignedBit, IntegerBit, NumBits, Uid) \
  uint32_t {                                                              \
    (BlockFormatBit << 24) | (SignedBit << 20) | (IntegerBit << 16) |     \
        (NumBits << 8) | (Uid)                                            \
  }

  // clang-format off
    Bfloat16 = ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u, /*bits*/  16u, /*uid*/  0u),
    E4m3     = ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u, /*bits*/   8u, /*uid*/  1u),
    Fp16     = ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u, /*bits*/  16u, /*uid*/  2u),
    Fp32     = ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u, /*bits*/  32u, /*uid*/  3u),
    Void     = ENCODE_DTYPE(/*block*/ 0u, /*signed*/ 1u, /*int*/ 0u, /*bits*/   0u, /*uid*/ 4u),
// clang-format on

#undef ENCODE_DTYPE
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// The number of bits in a data type?
inline int dtypeGetNumBits(Dtype dtype) {
  constexpr uint32_t kMask = 0xffu << 8;
  return static_cast<int>((static_cast<uint32_t>(dtype) & kMask) >> 8);
}

// For logging and error reporting
inline std::string dtypeToString(Dtype dtype) {
  switch (dtype) {
    case Dtype::Bfloat16:
      return "Bfloat16";
    case Dtype::E4m3:
      return "E4m3";
    case Dtype::Fp16:
      return "Fp16";
    case Dtype::Fp32:
      return "Fp32";
    case Dtype::Void:
      return "Void";
    default:
      TORCH_CHECK(false, "Unsupported type");
      return "Error";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string dtypeToPtxString(Dtype dtype) {
  switch (dtype) {
    case Dtype::Bfloat16:
      return "bf16";
    case Dtype::E4m3:
      return "e4m3";
    case Dtype::Fp16:
      return "f16";
    case Dtype::Fp32:
      return "f32";
    case Dtype::Void:
    default:
      TORCH_CHECK(false, "Unsupported type");
      return "Error";
  }
}

// The number of bytes in a data type?
inline int dtypeGetNumBytes(Dtype dtype) {
  TORCH_CHECK(dtypeGetNumBits(dtype) % 8 == 0, "Sub-byte types not supported");
  return dtypeGetNumBits(dtype) / 8;
}

class Type {
 public:
  // Ctor.
  Type(Dtype dtype, int numElts = 1) : mDtype{dtype}, mNumElts{numElts} {}

  // The data type.
  inline Dtype getDtype() const { return mDtype; }
  // The number of elements.
  inline int getNumElts() const { return mNumElts; }

 private:
  // The data type.
  Dtype const mDtype;
  // The number of elements.
  int const mNumElts;
};
