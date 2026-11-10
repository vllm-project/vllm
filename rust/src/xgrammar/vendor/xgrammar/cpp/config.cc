/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/config.cc
 */

#include <xgrammar/config.h>

#include "support/json_serializer.h"
#include "support/recursion_guard.h"

namespace xgrammar {

void SetMaxRecursionDepth(int max_recursion_depth) {
  RecursionGuard::SetMaxRecursionDepth(max_recursion_depth);
}

int GetMaxRecursionDepth() { return RecursionGuard::GetMaxRecursionDepth(); }

std::string GetSerializationVersion() { return std::string(SerializeVersion::GetVersion()); }

}  // namespace xgrammar
