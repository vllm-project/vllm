// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "core/amd_arch.h"

using namespace vllm::amd;

constexpr bool parses_as(std::string_view name, GPUTarget target,
                         GPUArchitecture architecture) {
  const auto arch = parse_gcn_arch(name);
  return arch.target == target && arch.architecture == architecture;
}

static_assert(parses_as("gfx906", GPUTarget::GFX906, GPUArchitecture::GCN5));
static_assert(parses_as("gfx908", GPUTarget::GFX908, GPUArchitecture::CDNA1));
static_assert(parses_as("gfx90a", GPUTarget::GFX90A, GPUArchitecture::CDNA2));
static_assert(parses_as("gfx942", GPUTarget::GFX942, GPUArchitecture::CDNA3));
static_assert(parses_as("gfx950", GPUTarget::GFX950, GPUArchitecture::CDNA4));
static_assert(parses_as("gfx1250", GPUTarget::GFX1250, GPUArchitecture::CDNA5));
static_assert(parses_as("gfx1030", GPUTarget::GFX1030, GPUArchitecture::RDNA2));
static_assert(parses_as("gfx1100", GPUTarget::GFX1100, GPUArchitecture::RDNA3));
static_assert(parses_as("gfx1101", GPUTarget::GFX1101, GPUArchitecture::RDNA3));
static_assert(parses_as("gfx1102", GPUTarget::GFX1102, GPUArchitecture::RDNA3));
static_assert(parses_as("gfx1103", GPUTarget::GFX1103, GPUArchitecture::RDNA3));
static_assert(parses_as("gfx1150", GPUTarget::GFX1150,
                        GPUArchitecture::RDNA35));
static_assert(parses_as("gfx1151", GPUTarget::GFX1151,
                        GPUArchitecture::RDNA35));
static_assert(parses_as("gfx1152", GPUTarget::GFX1152,
                        GPUArchitecture::RDNA35));
static_assert(parses_as("gfx1153", GPUTarget::GFX1153,
                        GPUArchitecture::RDNA35));
static_assert(parses_as("gfx1200", GPUTarget::GFX1200, GPUArchitecture::RDNA4));
static_assert(parses_as("gfx1201", GPUTarget::GFX1201, GPUArchitecture::RDNA4));

static_assert(parses_as("gfx942:sramecc+:xnack-", GPUTarget::GFX942,
                        GPUArchitecture::CDNA3));
static_assert(parses_as("gfx942:xnack+", GPUTarget::GFX942,
                        GPUArchitecture::CDNA3));
static_assert(parses_as("gfx90a:xnack-", GPUTarget::GFX90A,
                        GPUArchitecture::CDNA2));
static_assert(parses_as("gfx1250-strict:xnack+", GPUTarget::GFX1250,
                        GPUArchitecture::CDNA5));

// Honor string_view's length; the input need not be null terminated.
static_assert(parses_as(std::string_view("gfx1200suffix", 7),
                        GPUTarget::GFX1200, GPUArchitecture::RDNA4));

constexpr bool unlisted_or_ambiguous_names_stay_unknown() {
  constexpr std::string_view names[] = {
      "",         "gfx",      "gfx12",    "gfx120",
      "gfx12000", "gfx9420",  "gfx9999",  "gfx1200garbage",
      "GFX1200",  " gfx1200", "gfx1200 ", ":gfx1200",
  };
  for (const auto name : names) {
    const auto arch = parse_gcn_arch(name);
    if (arch.target != GPUTarget::UNKNOWN ||
        arch.architecture != GPUArchitecture::UNKNOWN)
      return false;
  }
  return true;
}

static_assert(unlisted_or_ambiguous_names_stay_unknown());
