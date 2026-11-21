// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <string_view>

namespace octave {

enum class Target { Unsupported, GFX1100, GFX942 };

// Match the name before the first ':' exactly; other names are unsupported.
constexpr Target target_from_name(std::string_view name) noexcept {
  name = name.substr(0, name.find(':'));
  if (name == "gfx1100") return Target::GFX1100;
  if (name == "gfx942") return Target::GFX942;
  return Target::Unsupported;
}

}  // namespace octave
