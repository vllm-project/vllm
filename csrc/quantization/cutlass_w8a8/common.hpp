#pragma once

#include "cutlass/cutlass.h"

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                        \
  {                                                  \
    TORCH_CHECK(status == cutlass::Status::kSuccess, \
                cutlassGetStatusString(status))      \
  }
