// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <string_view>

namespace octave {

enum class Target { Unsupported, GFX1100, GFX942 };

// Preserve the prefix matching used by Octave dispatch.
constexpr Target target_from_name(std::string_view name) noexcept {
  if (name.rfind("gfx1100", 0) == 0) return Target::GFX1100;
  if (name.rfind("gfx942", 0) == 0) return Target::GFX942;
  return Target::Unsupported;
}

}  // namespace octave
