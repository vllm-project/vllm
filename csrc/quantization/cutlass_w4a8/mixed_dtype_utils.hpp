/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "cute/tensor.hpp"

#include <cuda.h>
#include <numeric>
#include "cutlass_extensions/common.hpp"

enum MixedDtypeGemmMode { ConvertOnly, ScaleOnly, ScaleWithZeroPoint };

/// Command line options parsing
struct MixedDtypeOptions {
  bool help = false;

  float alpha = 1.0f;
  float beta = 0.0f;
  int iterations = 100;
  int warmup = 10;
  int mode = 1;
  int m = 5120, n = 4096, k = 4096;
  int g = 128;
  int l = 1;

  // Parses the command line
  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("g", g);
    cmd.get_cmd_line_argument("mode", mode);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("warmup", warmup);
  }

  /// Prints the usage statement.
  std::ostream& print_usage(std::ostream& out) const {
    out << "55_hopper_mixed_dtype_gemm\n\n"
        << "  Hopper Mixed Data Type GEMM using a Warp Specialized kernel.\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage "
           "statement\n\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --l=<int>                   The number of independent gemm "
           "problems with mnk shape\n"
        << "  --g=<int>                   The size of each group for the "
           "scales and zeros. To broadcast a vector of scales or zeros, set "
           "the group size to K.\n"
        << "  --mode=<int>                The mode to run the gemm. 0 does (A "
           "@ B), 1 means A @ (scale * B), 2 means A @ (scale * B + "
           "zero-point).\n"
        << "  --alpha=<f32>               Epilogue scalar alpha\n"
        << "  --beta=<f32>                Epilogue scalar beta\n\n"
        << "  --iterations=<int>          Number of profiling iterations to "
           "perform.\n\n"
        << "  --warmup=<int>              Number of warmup iterations to "
           "perform.\n\n";

    out << "\n\nExamples:\n\n"
        << "$ " << "55_hopper_mixed_dtype_gemm"
        << " --m=1024 --n=512 --k=1024 -g=1024 --l=10 --alpha=2 --mode=2 "
           "--beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k * l;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct MixedDtypeResult {
  double avg_runtime_ms = 0.0;
  double gflops = 0.0;
  cutlass::Status status = cutlass::Status::kSuccess;
  cudaError_t error = cudaSuccess;
  bool passed = false;
};

/// Profiling Loop
template <class Gemm>
void mixed_dtype_profiling(Gemm& gemm, MixedDtypeOptions const& options,
                           MixedDtypeResult& result) {
  if (options.iterations <= 0) return;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<float> runtimes;
  runtimes.reserve(options.iterations);

  for (int iter = 0; iter < options.warmup + options.iterations; ++iter) {
    cudaEventRecord(start);
    CUTLASS_CHECK(gemm.run());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (iter >= options.warmup) {
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      runtimes.push_back(milliseconds);
    }
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Compute average setup and runtime and GFLOPs.
  result.avg_runtime_ms =
      std::accumulate(runtimes.begin(), runtimes.end(), 0.0f) / runtimes.size();
  result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

  std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x'
            << options.k << 'x' << options.l << std::endl;
  std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
  std::cout << "  GFLOPS: " << result.gflops << std::endl;
}

/// Helpers to initialize a block of device data
template <class Element>
bool initialize_tensor(cutlass::DeviceAllocation<Element>& block,
                       uint64_t seed = 2023) {
  double scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  int bits_output = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  } else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  } else if (bits_output == 16) {
    scope_max = 5;
    scope_min = -5;
  } else {
    scope_max = 8;
    scope_min = -8;
  }
  cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(scope_max), Element(scope_min));

  return true;
}

template <class Element>
bool initialize_scale(cutlass::DeviceAllocation<Element>& block,
                      MixedDtypeOptions const& options, uint64_t seed = 2023) {
  // If no scales, initialize with 1 so we can use the same kernel to dequantize
  // the data
  float scope_max = 1.0f, scope_min = 1.0f;
  if (options.mode != MixedDtypeGemmMode::ConvertOnly) {
    float elt_max_f = float(cutlass::platform::numeric_limits<Element>::max());
    scope_max = 2.f;
    scope_min = 0.1f;
  }
  cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(scope_max), Element(scope_min));

  return true;
}

template <class Element>
bool initialize_zero(cutlass::DeviceAllocation<Element>& block,
                     MixedDtypeOptions const& options, uint64_t seed = 2023) {
  // If no bias, initialize with 0 so we can use the same kernel to dequantize
  // the data
  float scope_max = 0.0f, scope_min = 0.0f;
  if (options.mode == MixedDtypeGemmMode::ScaleWithZeroPoint) {
    scope_max = 2.0f;
    scope_min = -2.0f;
  }
  cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(scope_max), Element(scope_min));

  return true;
}