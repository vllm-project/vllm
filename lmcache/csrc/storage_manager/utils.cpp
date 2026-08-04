#include "utils.h"
#include <algorithm>
#include <stdexcept>

namespace lmcache {
namespace utils {

ParallelPatternMatcher::ParallelPatternMatcher(const std::vector<int>& pattern)
    : pattern_(pattern) {
  if (pattern_.empty()) {
    throw std::invalid_argument("Pattern cannot be empty");
  }
}

std::vector<int> ParallelPatternMatcher::match(const std::vector<int>& data) {
  // Handle edge cases
  if (data.size() < pattern_.size()) {
    return std::vector<int>();
  }

  std::vector<int> matches;

  // Search for pattern in the data
  size_t search_end = data.size() - pattern_.size() + 1;

  for (size_t i = 0; i < search_end; ++i) {
    bool match = true;
    for (size_t j = 0; j < pattern_.size(); ++j) {
      if (data[i + j] != pattern_[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      matches.push_back(static_cast<int>(i));
    }
  }

  return matches;
}

RangePatternMatcher::RangePatternMatcher(const std::vector<int>& start_pattern,
                                         const std::vector<int>& end_pattern)
    : start_pattern_(start_pattern), end_pattern_(end_pattern) {
  if (start_pattern_.empty()) {
    throw std::invalid_argument("Start pattern cannot be empty");
  }
  if (end_pattern_.empty()) {
    throw std::invalid_argument("End pattern cannot be empty");
  }
}

bool RangePatternMatcher::matchesAt(const std::vector<int>& data, size_t pos,
                                    const std::vector<int>& pattern) const {
  if (pos + pattern.size() > data.size()) {
    return false;
  }
  for (size_t i = 0; i < pattern.size(); ++i) {
    if (data[pos + i] != pattern[i]) {
      return false;
    }
  }
  return true;
}

std::vector<std::pair<int, int>> RangePatternMatcher::match(
    const std::vector<int>& data) {
  std::vector<std::pair<int, int>> ranges;

  // Handle edge cases
  if (data.size() < start_pattern_.size() + end_pattern_.size()) {
    return ranges;
  }

  size_t i = 0;
  while (i <= data.size() - start_pattern_.size()) {
    // Check if start pattern matches at position i
    if (matchesAt(data, i, start_pattern_)) {
      int start_pos = static_cast<int>(i);

      // Look for the first end pattern after the start pattern
      size_t search_start = i + start_pattern_.size();
      bool found_end = false;

      for (size_t j = search_start; j <= data.size() - end_pattern_.size();
           ++j) {
        if (matchesAt(data, j, end_pattern_)) {
          int end_pos = static_cast<int>(j + end_pattern_.size());
          ranges.push_back({start_pos, end_pos});

          // Move to the position after this range to continue searching
          i = j + end_pattern_.size();
          found_end = true;
          break;
        }
      }

      // If no end pattern was found, move past the start pattern
      if (!found_end) {
        i += start_pattern_.size();
      }
    } else {
      ++i;
    }
  }

  return ranges;
}

}  // namespace utils
}  // namespace lmcache
