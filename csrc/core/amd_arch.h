// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <string_view>

namespace vllm::amd {

// GPU targets include discrete GPUs and the GPU portion of APUs.
enum class GPUTarget {
  UNKNOWN,
  GFX906,
  GFX908,
  GFX90A,
  GFX942,
  GFX950,
  GFX1250,
  GFX1030,
  GFX1100,
  GFX1101,
  GFX1102,
  GFX1103,
  GFX1150,
  GFX1151,
  GFX1152,
  GFX1153,
  GFX1200,
  GFX1201,
};

enum class GPUArchitecture {
  UNKNOWN,
  GCN5,
  CDNA1,
  CDNA2,
  CDNA3,
  CDNA4,
  CDNA5,
  RDNA2,
  RDNA3,
  RDNA35,
  RDNA4,
};

struct GPUArch {
  GPUTarget target = GPUTarget::UNKNOWN;
  GPUArchitecture architecture = GPUArchitecture::UNKNOWN;
};

namespace detail {

struct GPUArchEntry {
  std::string_view name;
  GPUArch arch;
};

// Classification reference: rocprim::detail::gen_from_target_arch() in
// ROCm/rocm-libraries at commit e84109fead1dc6db99387ad691cd5dcb1f424cec:
// projects/rocprim/rocprim/include/rocprim/device/config_types.hpp.
// This table covers only HIP_SUPPORTED_ARCHS in CMakeLists.txt and classifies
// gfx1150-gfx1153 as RDNA35; upstream groups them under rdna3.
// Architecture classification does not imply support by any particular kernel.
inline constexpr GPUArchEntry kGPUArchEntries[] = {
    {"gfx906", {GPUTarget::GFX906, GPUArchitecture::GCN5}},

    {"gfx908", {GPUTarget::GFX908, GPUArchitecture::CDNA1}},

    {"gfx90a", {GPUTarget::GFX90A, GPUArchitecture::CDNA2}},

    {"gfx942", {GPUTarget::GFX942, GPUArchitecture::CDNA3}},

    {"gfx950", {GPUTarget::GFX950, GPUArchitecture::CDNA4}},

    {"gfx1250", {GPUTarget::GFX1250, GPUArchitecture::CDNA5}},

    {"gfx1030", {GPUTarget::GFX1030, GPUArchitecture::RDNA2}},

    {"gfx1100", {GPUTarget::GFX1100, GPUArchitecture::RDNA3}},
    {"gfx1101", {GPUTarget::GFX1101, GPUArchitecture::RDNA3}},
    {"gfx1102", {GPUTarget::GFX1102, GPUArchitecture::RDNA3}},
    {"gfx1103", {GPUTarget::GFX1103, GPUArchitecture::RDNA3}},

    {"gfx1150", {GPUTarget::GFX1150, GPUArchitecture::RDNA35}},
    {"gfx1151", {GPUTarget::GFX1151, GPUArchitecture::RDNA35}},
    {"gfx1152", {GPUTarget::GFX1152, GPUArchitecture::RDNA35}},
    {"gfx1153", {GPUTarget::GFX1153, GPUArchitecture::RDNA35}},

    {"gfx1200", {GPUTarget::GFX1200, GPUArchitecture::RDNA4}},
    {"gfx1201", {GPUTarget::GFX1201, GPUArchitecture::RDNA4}},
};

}  // namespace detail

// Parsing references: rocprim::detail::parse_gcn_arch() and
// get_target_arch_from_name(), in the same upstream file at commit
// e84109fead1dc6db99387ad691cd5dcb1f424cec.
// Like upstream, strips ':' target features (e.g. "gfx942:sramecc+:xnack-").
// This HIP-independent constexpr implementation honors string_view's length
// instead of scanning a HIP property buffer for ':' or '\0', and returns both
// target and architecture through exact table lookup instead of macros.
// Only "gfx1250-strict" is normalized to "gfx1250"; upstream also accepts the
// enum spelling "gfx1250_strict" via its '-'/'_' matching rule.
// Unlisted processors remain UNKNOWN.
constexpr GPUArch parse_gcn_arch(std::string_view name) noexcept {
  name = name.substr(0, name.find(':'));
  if (name == "gfx1250-strict") name = "gfx1250";
  for (const auto& entry : detail::kGPUArchEntries) {
    if (name == entry.name) return entry.arch;
  }
  return {};
}

}  // namespace vllm::amd
