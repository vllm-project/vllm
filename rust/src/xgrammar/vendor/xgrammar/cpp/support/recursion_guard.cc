/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/support/recursion_guard.cc
 */

#include "recursion_guard.h"

#include <charconv>
#include <cstdlib>
#include <string_view>
#include <system_error>

#include "logging.h"

namespace xgrammar {

int RecursionGuard::LoadMaxRecursionDepthFromEnv() {
  const char* env_value = std::getenv(kMaxRecursionDepthEnvVar);
  if (env_value == nullptr) {
    return kDefaultMaxRecursionDepth;
  }

  int value = 0;
  std::string_view sv(env_value);

  // Convert the string to an integer
  auto result = std::from_chars(sv.data(), sv.data() + sv.size(), value);

  // Check if the conversion is successful
  if (result.ec == std::errc::invalid_argument || result.ec == std::errc::result_out_of_range ||
      result.ptr != sv.data() + sv.size() || value <= 0) {
    XGRAMMAR_LOG(WARNING) << "Env variable XGRAMMAR_MAX_RECURSION_DEPTH is not a valid "
                             "integer or out of range: '"
                          << env_value << "', using default " << kDefaultMaxRecursionDepth;
    return kDefaultMaxRecursionDepth;
  }

  // Check if the value is too large
  if (value > kMaxReasonableDepth) {
    XGRAMMAR_LOG(WARNING) << "Env variable XGRAMMAR_MAX_RECURSION_DEPTH too large: " << value
                          << ", clamping to " << kMaxReasonableDepth;
    return kMaxReasonableDepth;
  }

  return value;
}

std::atomic<int> RecursionGuard::max_recursion_depth_{LoadMaxRecursionDepthFromEnv()};

}  // namespace xgrammar
