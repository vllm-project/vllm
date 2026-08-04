// SPDX-License-Identifier: Apache-2.0

#pragma once

// Backend-agnostic transfer descriptors, free of any vendor runtime headers so
// every backend (CUDA, ROCm, MUSA, SYCL/XPU, ...) shares one definition.

#include "engine_kv_format.h"  // EngineKVFormat + its classification predicates

enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};
