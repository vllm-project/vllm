// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
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
