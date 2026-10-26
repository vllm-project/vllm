// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Candidate records of the gated scan: a uint16 value (the low 16 bits of the
// fp24 score code) and an int32 word (key index in bits 0-19, the code's high
// 8 bits in bits 20-27).
#pragma once
#include <cstdint>
#include <cutlass/cutlass.h>

#include "common.cuh"
#include "constants.h"
namespace fused_mqa_topk {
using CandidateValue = uint16_t;
constexpr uint32_t kCandidateIndexBits = 20;
constexpr uint32_t kCandidateIndexMask = (1u << kCandidateIndexBits) - 1u;
static_assert(kMaxKeys <= (1u << kCandidateIndexBits),
              "key indices must fit the candidate index field");

CUTLASS_DEVICE uint32_t candidate_pack_index(const uint32_t payload,
                                             const uint32_t kv_index) {
  return (kv_index & kCandidateIndexMask) |
         ((payload >> 16) << kCandidateIndexBits);
}

CUTLASS_DEVICE uint32_t candidate_fp24_code(const float value) {
  return ordered_code(value) >> 8;
}

CUTLASS_DEVICE uint32_t candidate_load_score_code(const CandidateValue value,
                                                  const int32_t packed_idx) {
  return (static_cast<uint32_t>(packed_idx) >> kCandidateIndexBits) << 16 |
         static_cast<uint32_t>(value);
}

CUTLASS_DEVICE int32_t candidate_decode_index(const int32_t packed_idx) {
  return static_cast<int32_t>(static_cast<uint32_t>(packed_idx) &
                              kCandidateIndexMask);
}

// Stores a candidate as its fp24 code (low 16 bits in value_dst, high 8 bits
// in bits 20-27 of the index word) and key index.
CUTLASS_DEVICE void store_candidate_record(CandidateValue* value_dst,
                                           int32_t* index_dst,
                                           const uint32_t payload,
                                           const uint32_t kv_index) {
  __stcg(value_dst, static_cast<CandidateValue>(payload));
  __stcg(index_dst,
         static_cast<int32_t>(candidate_pack_index(payload, kv_index)));
}

CUTLASS_DEVICE void store_candidate(CandidateValue* value_dst,
                                    int32_t* index_dst, const float bq,
                                    const uint32_t kv_index) {
  store_candidate_record(value_dst, index_dst, candidate_fp24_code(bq),
                         kv_index);
}

}  // namespace fused_mqa_topk
