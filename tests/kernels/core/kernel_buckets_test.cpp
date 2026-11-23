// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "core/kernel_buckets.h"

#include <cassert>
#include <initializer_list>

using vllm::kernel_buckets::m_in_range;

static_assert(m_in_range(16, 0, 17));
static_assert(!m_in_range(17, 0, 17));
static_assert(m_in_range(1024, 1024));

int main() {
  for (int stop : {5, 9, 13, 17, 25, 33, 34, 65, 129, 257, 513, 1025, 1600,
                   2049, 4097, 8192, 1048576}) {
    for (int m : {0, 1, stop - 1, stop, stop + 1}) {
      assert(m_in_range(m, 0, stop) == (m < stop));
      assert(m_in_range(m, stop) == (m >= stop));
    }
  }
}
