/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/int_set.h
 * \brief The header for utilities used in grammar-guided generation.
 */
#ifndef XGRAMMAR_SUPPORT_INT_SET_H_
#define XGRAMMAR_SUPPORT_INT_SET_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace xgrammar {

/*!
 * \brief Let lhs be the union of lhs and rhs. Suppose that both sets are sorted.
 * \note No additional vectors are allocated, and the time complexity is O(n)
 */
inline void IntsetUnion(std::vector<int32_t>* lhs, const std::vector<int32_t>& rhs) {
  int original_lhs_size = lhs->size();
  int rhs_size = rhs.size();

  lhs->resize(original_lhs_size + rhs_size);

  auto it_lhs = lhs->rbegin() + rhs_size;
  auto it_rhs = rhs.rbegin();
  auto it_result = lhs->rbegin();

  while (it_lhs != lhs->rend() && it_rhs != rhs.rend()) {
    if (*it_lhs > *it_rhs) {
      *it_result = *it_lhs;
      ++it_lhs;
    } else if (*it_lhs < *it_rhs) {
      *it_result = *it_rhs;
      ++it_rhs;
    } else {
      *it_result = *it_lhs;
      ++it_lhs;
      ++it_rhs;
    }
    ++it_result;
  }

  while (it_rhs != rhs.rend()) {
    *it_result = *it_rhs;
    ++it_result;
    ++it_rhs;
  }

  auto last = std::unique(lhs->begin(), lhs->end());
  lhs->erase(last, lhs->end());
}

/*!
 * \brief Let lhs be the intersection of lhs and rhs. Suppose that both sets are sorted.
 * \note No additional vector is allocated, and the time complexity is O(n).
 * \note Support the case where lhs is the universal set by setting lhs to {-1}. The result will be
 * rhs then.
 */
inline void IntsetIntersection(std::vector<int32_t>* lhs, const std::vector<int32_t>& rhs) {
  if (lhs->size() == 1 && (*lhs)[0] == -1) {
    *lhs = rhs;
    return;
  }

  auto it_lhs = lhs->begin();
  auto it_rhs = rhs.begin();
  auto it_result = lhs->begin();

  while (it_lhs != lhs->end() && it_rhs != rhs.end()) {
    if (*it_lhs < *it_rhs) {
      ++it_lhs;
    } else if (*it_lhs > *it_rhs) {
      ++it_rhs;
    } else {
      *it_result = *it_lhs;
      ++it_lhs;
      ++it_rhs;
      ++it_result;
    }
  }
  lhs->erase(it_result, lhs->end());
}

/*!
 * \brief Let lhs = lhs - rhs. Both sets must be sorted.
 * \note In-place, no additional vector allocated, O(n) time.
 */
inline void IntsetDifference(std::vector<int32_t>* lhs, const std::vector<int32_t>& rhs) {
  auto it_lhs = lhs->begin();
  auto it_rhs = rhs.begin();
  auto it_result = lhs->begin();

  while (it_lhs != lhs->end() && it_rhs != rhs.end()) {
    if (*it_lhs < *it_rhs) {
      *it_result++ = *it_lhs++;
    } else if (*it_lhs > *it_rhs) {
      ++it_rhs;
    } else {
      ++it_lhs;
      ++it_rhs;
    }
  }
  while (it_lhs != lhs->end()) {
    *it_result++ = *it_lhs++;
  }
  lhs->erase(it_result, lhs->end());
}

/*!
 * \brief Compute result = [0, n) - excluded. excluded must be sorted with values in [0, n).
 * \note O(n) time.
 */
inline void IntsetComplement(
    std::vector<int32_t>* result, int32_t n, const std::vector<int32_t>& excluded
) {
  result->clear();
  result->reserve(n - static_cast<int32_t>(excluded.size()));
  auto it = excluded.begin();
  for (int32_t i = 0; i < n; ++i) {
    if (it != excluded.end() && *it == i) {
      ++it;
    } else {
      result->push_back(i);
    }
  }
}

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_INT_SET_H_
