/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/support/recursion_guard.h
 * \brief The header for recursion depth guard.
 */

#ifndef XGRAMMAR_SUPPORT_RECURSION_GUARD_H_
#define XGRAMMAR_SUPPORT_RECURSION_GUARD_H_

#include <atomic>
#include <optional>
#include <stdexcept>

#include "logging.h"

namespace xgrammar {

/*!
 * \brief Thread-safe recursion guard to prevent stack overflow
 *
 * This class provides a RAII-style guard that tracks recursion depth
 * and prevents excessive recursion that could lead to stack overflow.
 * It uses atomic operations for thread safety and supports configurable
 * maximum recursion depth.
 */
class RecursionGuard {
 public:
  /*!
   * \brief Constructor that increments recursion depth
   * \param current_recursion_depth Pointer to the current recursion depth counter
   * \throws Logs fatal error if max recursion depth is exceeded
   */
  explicit RecursionGuard(int* current_recursion_depth)
      : current_depth_ptr_(current_recursion_depth) {
    auto error = AddRecursionDepth(current_depth_ptr_);
    XGRAMMAR_CHECK(error == std::nullopt) << error.value().what();
  }

  /*!
   * \brief Reset the recursion depth to 0
   * \param current_recursion_depth Pointer to the current recursion depth counter
   */
  static void ResetRecursionDepth(int* current_recursion_depth) {
    XGRAMMAR_DCHECK(current_recursion_depth != nullptr);
    *current_recursion_depth = 0;
  }

  /*!
   * \brief Destructor that decrements recursion depth
   */
  ~RecursionGuard() { SubtractRecursionDepth(current_depth_ptr_); }

  /*!
   * \brief Get the maximum allowed recursion depth
   * \return Current maximum recursion depth limit
   */
  static int GetMaxRecursionDepth() { return max_recursion_depth_.load(std::memory_order_relaxed); }

  /*!
   * \brief Set the maximum allowed recursion depth
   * \param max_depth New maximum recursion depth limit (must be positive)
   */
  static void SetMaxRecursionDepth(int max_depth) {
    if (max_depth <= 0 || max_depth > kMaxReasonableDepth) {
      XGRAMMAR_LOG(FATAL
      ) << "RecursionGuard: Maximum recursion depth must be positive and less than "
        << kMaxReasonableDepth << ", got: " << max_depth;
    }
    max_recursion_depth_.store(max_depth, std::memory_order_relaxed);
  }

  static std::optional<std::runtime_error> AddRecursionDepth(int* current_recursion_depth) {
    XGRAMMAR_DCHECK(current_recursion_depth != nullptr);
    int current_depth = ++(*current_recursion_depth);
    int max_depth = max_recursion_depth_.load(std::memory_order_relaxed);
    if (current_depth > max_depth) {
      return std::runtime_error(
          "RecursionGuard: Maximum recursion depth exceeded. "
          "Current depth: " +
          std::to_string(current_depth) + ", Max allowed: " + std::to_string(max_depth)
      );
    }
    return std::nullopt;
  }

  static void SubtractRecursionDepth(int* current_recursion_depth) {
    XGRAMMAR_DCHECK(current_recursion_depth != nullptr && *current_recursion_depth > 0);
    --(*current_recursion_depth);
  }

 private:
  /*!
   * \brief Get the maximum allowed recursion depth from the environment variable. Used to
   * initialize max_recursion_depth_.
   * \return Current maximum recursion depth limit
   */
  static int LoadMaxRecursionDepthFromEnv();

  /*!
   * \brief Pointer to the recursion depth counter
   */
  int* current_depth_ptr_;

  /*!
   * \brief Thread-safe global configuration
   */
  static std::atomic<int> max_recursion_depth_;

  /*!
   * \brief Environment variable name for the maximum recursion depth
   */
  inline constexpr static char kMaxRecursionDepthEnvVar[] = "XGRAMMAR_MAX_RECURSION_DEPTH";

  /*!
   * \brief Default maximum recursion depth
   */
  inline constexpr static int kDefaultMaxRecursionDepth = 10000;

  /*!
   * \brief Maximum reasonable recursion depth
   */
  inline constexpr static int kMaxReasonableDepth = 1000000;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_RECURSION_GUARD_H_
