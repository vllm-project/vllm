/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/support/union_find_set.h
 */
#ifndef XGRAMMAR_SUPPORT_UNION_FIND_SET_H_
#define XGRAMMAR_SUPPORT_UNION_FIND_SET_H_

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "logging.h"

namespace xgrammar {

template <typename T>
class UnionFindSet {
 private:
  std::unordered_map<T, std::pair<T, size_t>> element_to_parent_and_size_;

 public:
  UnionFindSet() = default;

  /*!
   * \brief Add a new element to the union-find set.
   * \param element The element to add.
   * \return True if the element was added successfully, false if it already exists.
   */
  bool Add(const T& element) {
    if (element_to_parent_and_size_.find(element) != element_to_parent_and_size_.end()) {
      return false;  // Element already exists.
    }
    element_to_parent_and_size_[element] = {element, 1};
    return true;
  }

  /*! \brief Clear the union find set.*/
  void Clear() { element_to_parent_and_size_.clear(); }

  /*!
   * \brief Find the representative of the set containing the element.
   * \param element The element to find.
   * \return The representative of the set containing the element.
   */
  T Find(const T& element) {
    XGRAMMAR_CHECK(element_to_parent_and_size_.find(element) != element_to_parent_and_size_.end())
        << "Element not found in union-find set.";
    if (element_to_parent_and_size_[element].first != element) {
      // Path compression.
      element_to_parent_and_size_[element].first = Find(element_to_parent_and_size_[element].first);
    }
    return element_to_parent_and_size_[element].first;
  }

  /*!
   * \brief Union two elements into the same set.
   * \param a The first element.
   * \param b The second element.
   */
  void Union(const T& a, const T& b) {
    XGRAMMAR_CHECK(element_to_parent_and_size_.find(a) != element_to_parent_and_size_.end())
        << "Element " << a << " not found in union-find set.";
    XGRAMMAR_CHECK(element_to_parent_and_size_.find(b) != element_to_parent_and_size_.end())
        << "Element " << b << " not found in union-find set.";
    T root_a = Find(a);
    T root_b = Find(b);
    if (root_a == root_b) {
      return;
    }
    if (element_to_parent_and_size_[root_a].second < element_to_parent_and_size_[root_b].second) {
      std::swap(root_a, root_b);
      // Make sure root_a is the larger set.
    }
    element_to_parent_and_size_[root_b].first = root_a;
    element_to_parent_and_size_[root_a].second += element_to_parent_and_size_[root_b].second;
  }

  int Count(const T& element) const { return element_to_parent_and_size_.count(element); }

  std::vector<std::vector<T>> GetAllSets() {
    std::vector<std::vector<T>> result;
    std::unordered_map<T, size_t> root_to_set;
    for (const auto& [value, _] : element_to_parent_and_size_) {
      auto root = Find(value);
      if (root_to_set.find(root) == root_to_set.end()) {
        result.emplace_back();
        root_to_set[root] = result.size() - 1;
      }
      result[root_to_set[root]].push_back(value);
    }
    // Sort result to make it deterministic
    for (auto& vec : result) {
      std::sort(vec.begin(), vec.end());
    }
    std::sort(result.begin(), result.end(), [](const std::vector<T>& v1, const std::vector<T>& v2) {
      return v1.front() < v2.front();
    });
    return result;
  }
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_UNION_FIND_SET_H_
