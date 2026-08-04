#pragma once

#include <cstddef>
#include <vector>

namespace lmcache {
namespace utils {

/**
 * @brief Pattern matcher for integer vectors
 *
 * This class performs pattern matching on a vector of integers.
 * It finds all positions where a given pattern occurs in the input data.
 */
class ParallelPatternMatcher {
 public:
  /**
   * @brief Construct a new Pattern Matcher object
   *
   * @param pattern The pattern to search for
   */
  ParallelPatternMatcher(const std::vector<int>& pattern);

  /**
   * @brief Match the pattern in the given data
   *
   * @param data The data to search in
   * @return std::vector<int> Sorted vector of positions where pattern starts
   */
  std::vector<int> match(const std::vector<int>& data);

 private:
  std::vector<int> pattern_;
};

/**
 * @brief Range pattern matcher for integer vectors
 *
 * This class performs range pattern matching on a vector of integers.
 * It finds ranges that start with a start pattern and end with an end pattern.
 * When multiple end patterns exist after a start pattern, it matches the first
 * one (minimal range).
 */
class RangePatternMatcher {
 public:
  /**
   * @brief Construct a new Range Pattern Matcher object
   *
   * @param start_pattern The pattern marking the start of a range (1-5
   * elements)
   * @param end_pattern The pattern marking the end of a range (1-5 elements)
   */
  RangePatternMatcher(const std::vector<int>& start_pattern,
                      const std::vector<int>& end_pattern);

  /**
   * @brief Match ranges in the given data
   *
   * @param data The data to search in
   * @return std::vector<std::pair<int, int>> Vector of (start_pos, end_pos)
   * pairs where start_pos is the beginning of the start pattern and end_pos is
   * the exclusive index after the end pattern
   */
  std::vector<std::pair<int, int>> match(const std::vector<int>& data);

 private:
  std::vector<int> start_pattern_;
  std::vector<int> end_pattern_;

  /**
   * @brief Check if a pattern matches at a specific position
   *
   * @param data The data to search in
   * @param pos The position to check
   * @param pattern The pattern to match
   * @return true if pattern matches at pos, false otherwise
   */
  bool matchesAt(const std::vector<int>& data, size_t pos,
                 const std::vector<int>& pattern) const;
};

}  // namespace utils
}  // namespace lmcache
