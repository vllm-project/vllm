// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cstdint>
#include <limits>

namespace vllm::kernel_buckets {

// The operator row count is nonnegative. Stop is exclusive, as in Python.
constexpr int64_t kUnbounded = std::numeric_limits<int64_t>::max();

constexpr bool m_in_range(int64_t m, int64_t start, int64_t stop = kUnbounded) {
  return m >= start && (stop == kUnbounded || m < stop);
}

}  // namespace vllm::kernel_buckets
